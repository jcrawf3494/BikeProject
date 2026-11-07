import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.UUID
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.storage.StorageLevel

/** Computes inverse-frequency weights separately inside EVERY CV training fold.
  * Returns a standard PipelineModel, so saved models need no custom model loader.
  * The weight column is used by fit only; scoring does not require labels or weights.
  */
class WeightedPipeline(override val uid: String, val pipeline: Pipeline)
    extends Estimator[PipelineModel] {
  def this(pipeline: Pipeline) = this(Identifiable.randomUID("weightedPipeline"), pipeline)
  override def fit(data: Dataset[_]): PipelineModel = {
    val counts = data.groupBy("label").count().collect().map(r => r.getDouble(0) -> r.getLong(1)).toMap
    require(counts.keySet == Set(0.0, 1.0), "Each fitting subset must contain both classes")
    val total = counts.values.sum.toDouble
    val weighted = data.withColumn("class_weight",
      when(col("label") === 1.0, lit(total / (2.0 * counts(1.0))))
        .otherwise(lit(total / (2.0 * counts(0.0)))))
    pipeline.fit(weighted)
  }
  override def transformSchema(schema: StructType): StructType =
    pipeline.transformSchema(schema.add("class_weight", DoubleType, false))
  override def copy(extra: ParamMap): WeightedPipeline =
    new WeightedPipeline(uid, pipeline.copy(extra))
}

object TrainBikeShareModels {
  private val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
  private val defaultNumeric = Seq("duration_sec", "is_weekend", "is_holiday", "is_round_trip")
  private val defaultCategorical = Seq("start_hour", "day_of_week", "month")
  private val blocked = Set("id", "label", "subscription_type", "duration_min", "trip_date", "start_ts",
    "class_weight", "cv_fold", "prediction", "probability", "rawPrediction", "features")

  def writeJson(spark: SparkSession, path: String, value: Any): Unit = {
    val p = new Path(path)
    val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
    val out = fs.create(p, false)
    try out.write(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(value).getBytes(StandardCharsets.UTF_8))
    finally out.close()
  }
  private def elapsed(start: Long): Double = (System.nanoTime() - start) / 1e9
  private def csv(value: String): Seq[String] = value.split(",").map(_.trim).filter(_.nonEmpty).toSeq

  private def validate(data: DataFrame, name: String, numeric: Seq[String], categorical: Seq[String]): Long = {
    val required = Seq("id", "label") ++ numeric ++ categorical
    require(required.distinct.size == required.size, "Duplicate feature names, or numeric/categorical overlap")
    require(required.forall(data.columns.contains), s"$name missing columns: ${required.filterNot(data.columns.contains).mkString(",")}")
    require(data.filter(col("id").isNull || col("label").isNull || !col("label").isin(0.0, 1.0)).limit(1).count() == 0,
      s"$name contains missing IDs or labels other than 0/1")
    val n = data.count()
    require(n > 0 && data.select("id").distinct().count() == n, s"$name is empty or has duplicate IDs")
    require(data.select("label").distinct().count() == 2, s"$name must contain both classes")
    if (data.columns.contains("subscription_type")) {
      require(data.filter(col("subscription_type").isNull || !col("subscription_type").isin("Customer", "Subscriber") ||
        (col("label") =!= when(col("subscription_type") === "Customer", 1.0).otherwise(0.0))).limit(1).count() == 0,
        s"$name has a text/numeric label mismatch")
    }
    numeric.foreach { c =>
      val x = col(c).cast("double")
      require(data.filter(col(c).isNotNull && (x.isNull || isnan(x) || abs(x) === Double.PositiveInfinity)).limit(1).count() == 0,
        s"$name.$c contains nonnumeric or infinite values; clean it first")
      require(data.filter(x.isNotNull).limit(1).count() > 0, s"$name.$c is entirely missing")
    }
    n
  }

  private def prepare(data: DataFrame, numeric: Seq[String], categorical: Seq[String]): DataFrame = {
    data.select((Seq(col("id"), col("label").cast("double").as("label")) ++
      numeric.map(c => col(c).cast("double").as(c)) ++
      categorical.map(c => coalesce(col(c).cast("string"), lit("__MISSING__")).as(c))): _*)
  }

  // Stable hash order makes the sample and folds independent of Parquet partition layout.
  private def sample(data: DataFrame, fraction: Double, seed: Long): DataFrame = {
    if (fraction == 1.0) data else data.filter(
      pmod(xxhash64(col("id"), lit(seed)), lit(1000000000L)) < lit((fraction * 1000000000L).toLong))
  }
  private def folds(data: DataFrame, seed: Long): DataFrame = data.withColumn("cv_fold",
    pmod(row_number().over(Window.partitionBy("label").orderBy(xxhash64(col("id"), lit(seed)), col("id"))) - 1, lit(3)).cast("int"))

  private def recipe(model: String, seed: Long, numeric: Seq[String], categorical: Seq[String],
                     mode: String, gbtDepth: Int, gbtIterations: Int, durationCap: Double): (Pipeline, Array[ParamMap]) = {
    val stages = scala.collection.mutable.ArrayBuffer.empty[PipelineStage]
    if (durationCap > 0 && numeric.contains("duration_sec")) {
      // A fixed, declared cap; rows are never discarded and the test split stays intact.
      stages += new SQLTransformer().setStatement(
        s"SELECT *, CASE WHEN duration_sec IS NULL THEN CAST(NULL AS DOUBLE) ELSE LEAST(duration_sec, $durationCap) END AS duration_sec__capped FROM __THIS__")
    }
    if (numeric.nonEmpty) stages += new Imputer().setStrategy("median")
      .setInputCols(numeric.map(c => if (c == "duration_sec" && durationCap > 0) "duration_sec__capped" else c).toArray)
      .setOutputCols(numeric.map(_ + "__imputed").toArray)
    categorical.foreach { c => stages += new StringIndexer().setInputCol(c).setOutputCol(c + "__index")
      .setHandleInvalid("keep").setStringOrderType("alphabetAsc") }
    if (categorical.nonEmpty) stages += new OneHotEncoder().setInputCols(categorical.map(_ + "__index").toArray)
      .setOutputCols(categorical.map(_ + "__encoded").toArray).setHandleInvalid("keep").setDropLast(false)
    stages += new VectorAssembler().setInputCols((numeric.map(_ + "__imputed") ++ categorical.map(_ + "__encoded")).toArray)
      .setOutputCol("features").setHandleInvalid("error")
    val tiny = mode == "smoke"
    val grids = model match {
      case "P1" =>
        val gbt = new GBTClassifier().setWeightCol("class_weight").setSeed(seed)
          .setMaxDepth(if (tiny) 2 else gbtDepth).setMaxIter(if (tiny) 3 else gbtIterations).setMaxBins(64)
        stages += gbt
        if (mode == "tune") new ParamGridBuilder().addGrid(gbt.maxDepth, Array(3, 5))
          .addGrid(gbt.maxIter, Array(20, 40)).build() else Array(ParamMap.empty)
      case "P2" =>
        val rf = new RandomForestClassifier().setWeightCol("class_weight").setSeed(seed)
          .setMaxDepth(if (tiny) 3 else 5).setNumTrees(if (tiny) 5 else 50).setMaxBins(64)
        stages += rf
        if (mode == "tune") new ParamGridBuilder().addGrid(rf.maxDepth, Array(3, 5))
          .addGrid(rf.numTrees, Array(30, 60)).build() else Array(ParamMap.empty)
      case "P3" =>
        val lr = new LogisticRegression().setWeightCol("class_weight").setMaxIter(if (tiny) 15 else 100)
          .setRegParam(0.01).setElasticNetParam(0.0).setStandardization(true)
        stages += lr
        if (mode == "tune") new ParamGridBuilder().addGrid(lr.regParam, Array(0.001, 0.01, 0.1))
          .addGrid(lr.elasticNetParam, Array(0.0, 0.5)).build() else Array(ParamMap.empty)
    }
    (new Pipeline().setStages(stages.toArray), grids)
  }

  private def metrics(predictions: DataFrame): Map[String, Any] = {
    val r = predictions.agg(
      sum(when(col("label") === 1.0 && col("prediction") === 1.0, 1L).otherwise(0L)).as("tp"),
      sum(when(col("label") === 0.0 && col("prediction") === 1.0, 1L).otherwise(0L)).as("fp"),
      sum(when(col("label") === 0.0 && col("prediction") === 0.0, 1L).otherwise(0L)).as("tn"),
      sum(when(col("label") === 1.0 && col("prediction") === 0.0, 1L).otherwise(0L)).as("fn")
    ).head()
    val tp = r.getLong(0); val fp = r.getLong(1); val tn = r.getLong(2); val fn = r.getLong(3)
    def ratio(a: Double, b: Double): Double = if (b == 0) 0.0 else a / b
    val precision = ratio(tp, tp + fp); val recall = ratio(tp, tp + fn)
    val manual = ratio(1.25 * precision * recall, 0.25 * precision + recall)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("fMeasureByLabel").setMetricLabel(1.0).setBeta(0.5)
    val f05 = evaluator.evaluate(predictions)
    require(math.abs(manual - f05) < 1e-10, "F0.5 does not match confusion-matrix calculation")
    val auprc = new BinaryClassificationEvaluator().setMetricName("areaUnderPR").evaluate(predictions)
    Map("customer_f0_5" -> f05, "customer_precision" -> precision, "customer_recall" -> recall,
      "accuracy" -> ratio(tp + tn, tp + fp + tn + fn), "area_under_pr" -> auprc,
      "tp" -> tp, "fp" -> fp, "tn" -> tn, "fn" -> fn, "evaluation_rows" -> (tp + fp + tn + fn))
  }

  def main(args: Array[String]): Unit = {
    if (args.contains("--help")) {
      println("TrainBikeShareModels --data-dir data --output-dir results --mode smoke|train|tune|volume " +
        "--models P3,P2,P1 --seeds 42 --fraction 1.0 --numeric-features a,b --categorical-features c,d " +
        "--parallelism 1 --partitions 8 --gbt-depth 5 --gbt-iterations 20 --duration-cap-seconds 0")
      return
    }
    require(args.length % 2 == 0, "Options must be --name value pairs; see --help")
    val opt = args.grouped(2).map(a => a(0) -> a(1)).toMap
    val allowed = Set("--data-dir", "--output-dir", "--mode", "--models", "--seeds", "--fraction",
      "--numeric-features", "--categorical-features", "--parallelism", "--partitions", "--gbt-depth", "--gbt-iterations", "--duration-cap-seconds")
    require(opt.keySet.subsetOf(allowed), s"Unknown options: ${opt.keySet -- allowed}")
    require(opt.size * 2 == args.length, "Duplicate options")
    val mode = opt.getOrElse("--mode", "smoke")
    require(Set("smoke", "train", "tune", "volume").contains(mode), "Invalid mode")
    val models = csv(opt.getOrElse("--models", if (mode == "volume") "P1" else "P3,P2,P1")).map(_.toUpperCase)
    require(models.nonEmpty && models.distinct.size == models.size && models.forall(Set("P1", "P2", "P3")), "Invalid models")
    require(mode != "volume" || models == Seq("P1"), "Volume mode runs P1 only")
    val seeds = csv(opt.getOrElse("--seeds", "42")).map(_.toLong)
    require(seeds.nonEmpty && seeds.distinct.size == seeds.size, "Seeds must be nonempty and unique")
    val fraction = opt.getOrElse("--fraction", if (mode == "smoke") "0.02" else "1.0").toDouble
    require(fraction > 0 && fraction <= 1, "Fraction must be in (0,1]")
    require(mode != "volume" || !opt.contains("--fraction"), "Volume mode sets fractions automatically")
    val numeric = csv(opt.getOrElse("--numeric-features", defaultNumeric.mkString(",")))
    val categorical = csv(opt.getOrElse("--categorical-features", defaultCategorical.mkString(",")))
    require((numeric ++ categorical).nonEmpty && !(numeric ++ categorical).exists(blocked), "Empty or forbidden feature list")
    require((numeric ++ categorical).forall(_.matches("[A-Za-z][A-Za-z0-9_]*")), "Use ordinary column names")
    val parallelism = opt.getOrElse("--parallelism", "1").toInt
    val partitions = opt.getOrElse("--partitions", "8").toInt
    val gbtDepth = opt.getOrElse("--gbt-depth", "5").toInt
    val gbtIterations = opt.getOrElse("--gbt-iterations", "20").toInt
    val cap = opt.getOrElse("--duration-cap-seconds", "0").toDouble
    require(parallelism > 0 && partitions > 0 && gbtDepth >= 1 && gbtDepth <= 30 && gbtIterations > 0 && cap >= 0 && !cap.isInfinity && !cap.isNaN,
      "Invalid numeric setting")
    val dataDir = opt.getOrElse("--data-dir", "data").stripSuffix("/")
    val runDir = opt.getOrElse("--output-dir", "results").stripSuffix("/") + "/" +
      mode + "-" + Instant.now().toString.replaceAll("[:.]", "-") + "-" + UUID.randomUUID().toString.take(8)
    val spark = SparkSession.builder().appName("Bike Share Model Training " + mode).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", partitions)
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try {
      require(spark.version.startsWith("3.5."), s"Use Spark 3.5.x; found ${spark.version}")
      val rawTrain = spark.read.parquet(dataDir + "/train.parquet")
      val rawTest = spark.read.parquet(dataDir + "/test.parquet")
      val trainRows = validate(rawTrain, "train", numeric, categorical)
      val testRows = validate(rawTest, "test", numeric, categorical)
      require(rawTrain.select("id").join(rawTest.select("id"), Seq("id"), "inner").limit(1).count() == 0, "Train/test IDs overlap")
      val train = prepare(rawTrain, numeric, categorical).persist(StorageLevel.MEMORY_AND_DISK)
      val test = prepare(rawTest, numeric, categorical)
      train.count()
      val manifest = Map[String, Any]("started_utc" -> Instant.now().toString, "spark_version" -> spark.version,
        "scala_version" -> util.Properties.versionNumberString, "java_version" -> System.getProperty("java.version"),
        "master" -> spark.sparkContext.master, "application_id" -> spark.sparkContext.applicationId,
        "mode" -> mode, "data_dir" -> new Path(dataDir).getName, "input_path_scope" -> "Directory name only; local absolute paths omitted", "train_rows" -> trainRows, "test_rows" -> testRows,
        "numeric_features" -> numeric, "categorical_features" -> categorical, "seeds" -> seeds, "models" -> models,
        "fraction" -> fraction, "duration_cap_seconds" -> cap, "cv_parallelism" -> parallelism,
        "shuffle_partitions" -> partitions, "test_split_recreated" -> false,
        "feature_scope" -> "Explicit feature lists only; default is preliminary trip/time features, without weather or clusters",
        "timing_definition" -> "CV seconds includes all folds plus its best-model refit; final_fit_seconds measures a separate fit of selected settings; excludes prediction and file writing")
      writeJson(spark, runDir + "/manifest.json", manifest)
      println(s"Validated $trainRows training and $testRows test rows. Output: $runDir")
      val results = scala.collection.mutable.ArrayBuffer.empty[Map[String, Any]]
      seeds.foreach { seed =>
        val fractions = if (mode == "volume") Seq(0.10, 0.25, 0.50, 1.0) else Seq(fraction)
        fractions.foreach { f =>
          println(s"Preparing training subset: seed=$seed fraction=$f")
          val selected = sample(train, f, seed)
          val folded = folds(selected, seed).persist(StorageLevel.MEMORY_AND_DISK)
          val selectedRows = folded.count()
          val classFolds = folded.groupBy("label", "cv_fold").count().collect()
          require(classFolds.length == 6, "Subset too small for three stratified folds; increase --fraction")
          val fitting = if (mode == "smoke") folded.filter(col("cv_fold") =!= 0) else folded
          val scoring = if (mode == "smoke") folded.filter(col("cv_fold") === 0) else test
          val fittingRows = fitting.count()
          val suffix = s"seed-$seed/fraction-${math.round(f * 10000)}"
          folded.select("id", "label", "cv_fold").write.mode("errorifexists").parquet(runDir + "/folds/" + suffix)
          try models.foreach { modelName =>
            val modelDir = runDir + "/" + modelName + "/" + suffix
            val (pipeline, grid) = recipe(modelName, seed, numeric, categorical, mode, gbtDepth, gbtIterations, cap)
            val weighted = new WeightedPipeline(pipeline)
            var cvSeconds = 0.0
            var chosen = ParamMap.empty
            if (mode == "tune") {
              println(s"$modelName seed=$seed: ${grid.length} parameter sets x 3 folds")
              val cv = new CrossValidator().setEstimator(weighted).setEstimatorParamMaps(grid)
                .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("fMeasureByLabel").setMetricLabel(1.0).setBeta(0.5))
                .setNumFolds(3).setFoldCol("cv_fold").setSeed(seed).setParallelism(parallelism).setCollectSubModels(false)
              val started = System.nanoTime()
              val fittedCV = cv.fit(fitting)
              cvSeconds = elapsed(started)
              val best = fittedCV.avgMetrics.indices.maxBy(i => fittedCV.avgMetrics(i))
              chosen = grid(best)
              val scores = grid.indices.map(i => Map[String, Any]("candidate" -> i,
                "mean_customer_f0_5" -> fittedCV.avgMetrics(i), "selected" -> (i == best),
                "parameters" -> grid(i).toSeq.map(p => p.param.name -> p.value.toString).toMap))
              writeJson(spark, modelDir + "/cross_validation.json", scores)
            }
            println(s"$modelName seed=$seed: fitting $fittingRows rows")
            val started = System.nanoTime()
            val fitted = weighted.copy(chosen).fit(fitting)
            val fitSeconds = elapsed(started)
            fitted.write.save(modelDir + "/model")
            println(s"$modelName: saved model; evaluating and checking reload")
            val predictions = fitted.transform(scoring).persist(StorageLevel.MEMORY_AND_DISK)
            try {
              val scores = metrics(predictions)
              require(predictions.count() == scoring.count(), "Prediction row count changed")
              val output = predictions.select(col("id"), col("label"), col("prediction"),
                vector_to_array(col("probability")).getItem(1).as("customer_probability"))
              output.write.mode("errorifexists").parquet(modelDir + "/predictions.parquet")
              // Verify standard model persistence and score without any label or weight column.
              val reloaded = PipelineModel.load(modelDir + "/model")
              val probe = scoring.orderBy("id").limit(25).drop("label", "cv_fold")
              val before = fitted.transform(probe).select("id", "prediction", "probability")
              val after = reloaded.transform(probe).select("id", "prediction", "probability")
              require(before.except(after).count() == 0 && after.except(before).count() == 0, "Reloaded predictions differ")
              val record = scores ++ Map[String, Any]("model" -> modelName, "seed" -> seed,
                "requested_fraction" -> f, "sample_rows" -> selectedRows, "training_rows" -> fittingRows,
                "evaluation_scope" -> (if (mode == "smoke") "training_validation_only" else "fixed_test"),
                "cv_total_seconds" -> cvSeconds, "final_fit_seconds" -> fitSeconds,
                "class_weighting" -> "N/(2*N_class), recomputed inside each training fold",
                "model_reload_verified" -> true,
                "classifier_parameters" -> fitted.stages.last.extractParamMap().toSeq.map(p => p.param.name -> p.value.toString).toMap)
              writeJson(spark, modelDir + "/metrics.json", record)
              results += record
              println(s"$modelName seed=$seed fraction=$f: F0.5=${scores("customer_f0_5")} fit=${fitSeconds}s")
            } finally predictions.unpersist()
          } finally folded.unpersist()
        }
      }
      writeJson(spark, runDir + "/results.json", results.toSeq)
      val summary = spark.read.json(spark.createDataset(results.map(r => mapper.writeValueAsString(r)).toSeq)(org.apache.spark.sql.Encoders.STRING))
      summary.select("model", "seed", "requested_fraction", "training_rows", "evaluation_rows", "evaluation_scope",
        "customer_f0_5", "customer_precision", "customer_recall", "accuracy", "area_under_pr",
        "tp", "fp", "tn", "fn", "cv_total_seconds", "final_fit_seconds")
        .coalesce(1).write.mode("errorifexists").option("header", "true").csv(runDir + "/summary.csv")
      train.unpersist()
      println("COMPLETE: " + runDir)
    } finally spark.stop()
  }
}
