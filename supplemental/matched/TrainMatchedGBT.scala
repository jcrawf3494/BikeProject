import java.nio.charset.StandardCharsets
import java.time.Instant
import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{Imputer, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import scala.collection.JavaConverters._

/** Isolated P1 fitting worker. Python coordinates the shared training-only search.
  * Test features are opened only by an explicitly frozen final-fit job.
  */
object TrainMatchedGBT {
  private val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
  private def write(spark: SparkSession, path: String, value: Any): Unit = {
    val p = new Path(path); val fs = p.getFileSystem(spark.sparkContext.hadoopConfiguration)
    val stream = fs.create(p, false)
    try stream.write(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(value).getBytes(StandardCharsets.UTF_8))
    finally stream.close()
  }
  private def exists(spark: SparkSession, path: String): Boolean = {
    val p = new Path(path); p.getFileSystem(spark.sparkContext.hadoopConfiguration).exists(p)
  }
  private def strings(node: JsonNode): Seq[String] = node.elements().asScala.map(_.asText()).toSeq
  private def parameters(node: JsonNode): Map[String, String] =
    node.fields().asScala.map(e => e.getKey -> e.getValue.asText()).toMap
  private def project(frame: DataFrame, numeric: Seq[String], categorical: Seq[String]): DataFrame =
    frame.select((Seq(col("id"), col("label").cast("double").as("label")) ++
      numeric.map(c => col(c).cast("double").as(c)) ++
      categorical.map(c => coalesce(col(c).cast("string"), lit("__MISSING__")).as(c))): _*)
  private def prep(numeric: Seq[String], categorical: Seq[String]): Pipeline = {
    val stages = scala.collection.mutable.ArrayBuffer.empty[PipelineStage]
    stages += new SQLTransformer().setStatement("SELECT *, LOG1P(duration_sec) AS duration_sec__log FROM __THIS__")
    stages += new Imputer().setStrategy("median")
      .setInputCols(numeric.map(c => if (c == "duration_sec") "duration_sec__log" else c).toArray)
      .setOutputCols(numeric.map(_ + "__imputed").toArray)
    stages += new StringIndexer().setInputCols(categorical.toArray)
      .setOutputCols(categorical.map(_ + "__index").toArray)
      .setHandleInvalid("keep").setStringOrderType("alphabetAsc")
    stages += new VectorAssembler().setInputCols((numeric.map(_ + "__imputed") ++
      categorical.map(_ + "__index")).toArray).setOutputCol("features").setHandleInvalid("error")
    new Pipeline().setStages(stages.toArray)
  }
  def main(args: Array[String]): Unit = {
    require(args.length == 2 && args(0) == "--config", "Use --config private-job.json")
    val spark = SparkSession.builder().appName("Matched full-data GBT worker").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try {
      val configPath = new Path(args(1))
      val stream = configPath.getFileSystem(spark.sparkContext.hadoopConfiguration).open(configPath)
      val config = try mapper.readTree(stream) finally stream.close()
      val numeric = strings(config.get("numeric_features")); val categorical = strings(config.get("categorical_features"))
      require((numeric ++ categorical).distinct.size == 39)
      require(!(numeric ++ categorical).exists(Set("id", "label", "subscription_type", "zip_code", "local_zip", "trip_date", "start_ts")))
      val root = config.get("output_dir").asText(); val data = config.get("data_dir").asText()
      val technical = config.path("technical_verification").asBoolean(false)
      val stage = config.get("stage").asText(); val seed = config.get("seed").asLong()
      require(Set("cv", "stack_validation", "calibration", "final")(stage))
      require(stage != "final" || config.path("settings_frozen").asBoolean(false), "Freeze settings before final scoring")
      spark.sparkContext.setCheckpointDir(root + "/checkpoints")
      val raw = project(spark.read.parquet(data + "/train.parquet"), numeric, categorical)
        .persist(StorageLevel.MEMORY_AND_DISK)
      val inputRows = raw.count()
      require(technical || inputRows == 536406)
      require(raw.select("id").distinct().count() == inputRows)
      require(raw.filter(col("label").isNull || !col("label").isin(0.0,1.0)).limit(1).count() == 0)
      require(raw.filter(col("duration_sec") < 0).limit(1).count() == 0)
      val roles = spark.read.parquet(config.get("roles_file").asText())
      require(roles.count() == inputRows && roles.select("id").distinct().count() == inputRows)
      require(raw.select("id").except(roles.select("id")).count() == 0)
      val joined = raw.join(roles.select("id","phase","cv_fold"), Seq("id"))
      val folds = if (stage == "cv") Seq(0,1,2) else Seq(-1)
      val candidates = config.get("candidates").elements().asScala.toSeq
      folds.foreach { fold =>
        val fitting = stage match {
          case "cv" => joined.filter(col("phase") === "cv" && col("cv_fold") =!= fold)
          case "stack_validation" => joined.filter(col("phase") === "cv")
          case "calibration" => joined.filter(col("phase") =!= "calibration")
          case "final" => raw
        }
        val scoring = stage match {
          case "cv" => joined.filter(col("phase") === "cv" && col("cv_fold") === fold)
          case "stack_validation" => joined.filter(col("phase") === "stack_validation")
          case "calibration" => joined.filter(col("phase") === "calibration")
          case "final" => project(spark.read.parquet(data + "/test.parquet"), numeric, categorical)
        }
        val fittingRows = fitting.count(); val scoringRows = scoring.count()
        require(fitting.select("id").join(scoring.select("id"),Seq("id")).limit(1).count() == 0)
        if (stage == "final" && !technical) require(fittingRows == 536406 && scoringRows == 133553)
        val pending = candidates.zipWithIndex.filter { case (_,index) =>
          !exists(spark, root + s"/candidate-$index/fold-$fold/DONE.json")
        }
        if (pending.nonEmpty) {
          val fittingEvidence = root + s"/fold-$fold/fitting_ids.parquet"
          if (!exists(spark,fittingEvidence)) fitting.select("id","label").write.mode("errorifexists").parquet(fittingEvidence)
          val preprocessing = prep(numeric,categorical).fit(fitting)
          val counts = fitting.groupBy("label").count().collect().map(r => r.getDouble(0) -> r.getLong(1)).toMap
          require(counts.keySet == Set(0.0,1.0))
          val weights = Map("Customer" -> (fittingRows / (2.0 * counts(1.0))),
                            "Subscriber" -> (fittingRows / (2.0 * counts(0.0))))
          val vectors = preprocessing.transform(fitting).select("id","label","features")
            .withColumn("class_weight",when(col("label") === 1.0,weights("Customer")).otherwise(weights("Subscriber")))
            .persist(StorageLevel.MEMORY_AND_DISK)
          val evaluation = preprocessing.transform(scoring).select("id","label","features")
            .persist(StorageLevel.MEMORY_AND_DISK)
          vectors.count(); evaluation.count()
          try pending.foreach { case (candidate,index) =>
            val directory = root + s"/candidate-$index/fold-$fold"
            // Failed attempts are preserved rather than overwritten.
            val attempt = directory + "/attempt-" + Instant.now().toString.replaceAll("[:.]","-")
            val gbt = new GBTClassifier().setSeed(seed).setWeightCol("class_weight")
              .setMaxDepth(candidate.get("maxDepth").asInt()).setMaxIter(candidate.get("maxIter").asInt())
              .setStepSize(candidate.get("stepSize").asDouble())
              .setMinInstancesPerNode(candidate.get("minInstancesPerNode").asInt())
              .setMaxBins(128).setSubsamplingRate(1.0).setFeatureSubsetStrategy("all")
              .setCheckpointInterval(10).setMaxMemoryInMB(512)
            println(s"START P1 stage=$stage fold=$fold candidate=$index fitting=$fittingRows scoring=$scoringRows")
            val started = System.nanoTime(); val fitted = gbt.fit(vectors)
            val seconds = (System.nanoTime()-started)/1e9
            val threshold = config.path("threshold").asDouble(0.5)
            require(threshold > 0 && threshold < 1)
            fitted.setThresholds(Array(1-threshold,threshold))
            val predictions = fitted.transform(evaluation).select(col("id"),col("label"),col("prediction"),
              vector_to_array(col("probability")).getItem(1).as("customer_probability"))
            predictions.write.mode("errorifexists").parquet(attempt + "/predictions.parquet")
            var reloadVerified = false
            if (stage == "final") {
              val pipeline = new Pipeline().setStages(preprocessing.stages ++ Array[PipelineStage](fitted)).fit(fitting)
              pipeline.write.save(attempt + "/model")
              val restored = PipelineModel.load(attempt + "/model")
              val probe = scoring.orderBy("id").limit(25).drop("label")
              val before = pipeline.transform(probe).select("id","prediction","probability")
              val after = restored.transform(probe).select("id","prediction","probability")
              require(before.except(after).count() == 0 && after.except(before).count() == 0)
              reloadVerified = true
            }
            val record = Map("stage" -> stage,"fold" -> fold,"candidate" -> index,"seed" -> seed,
              "training_rows" -> fittingRows,"evaluation_rows" -> scoringRows,"fit_seconds" -> seconds,
              "class_weights" -> weights,"parameters" -> parameters(candidate),
              "predictions" -> (attempt + "/predictions.parquet"),"model_reload_verified" -> reloadVerified,
              "fitting_ids" -> fittingEvidence,
              "threshold" -> threshold,"attempt_directory" -> attempt,
              "test_used_for_selection" -> false,"technical_verification" -> technical)
            write(spark, directory + "/DONE.json",record)
            println(s"COMPLETE P1 stage=$stage fold=$fold candidate=$index fit=${seconds}s")
          } finally { vectors.unpersist(); evaluation.unpersist() }
        }
      }
      raw.unpersist()
      println("MATCHED_GBT_JOB_COMPLETE")
    } finally spark.stop()
  }
}
