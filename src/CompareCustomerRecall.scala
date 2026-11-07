import java.time.Instant
import java.util.UUID
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{GBTClassificationModel, RandomForestClassificationModel, LogisticRegressionModel}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

/** Selects a recall-constrained operating point using saved training-validation predictions.
  * Reuses frozen fitted models and preserves every earlier model and result.
  */
object CompareCustomerRecall {
  private val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
  private def metric(data: DataFrame, threshold: Double): Map[String, Any] = {
    val p = col("customer_probability") > threshold
    val r = data.agg(
      sum(when(col("label") === 1.0 && p, 1L).otherwise(0L)),
      sum(when(col("label") === 0.0 && p, 1L).otherwise(0L)),
      sum(when(col("label") === 0.0 && !p, 1L).otherwise(0L)),
      sum(when(col("label") === 1.0 && !p, 1L).otherwise(0L))).head()
    val tp = r.getLong(0); val fp = r.getLong(1); val tn = r.getLong(2); val fn = r.getLong(3)
    def divide(a: Double, b: Double): Double = if (b == 0.0) 0.0 else a / b
    Map("threshold" -> threshold, "customer_precision" -> divide(tp, tp + fp),
      "customer_recall" -> divide(tp, tp + fn), "customer_f0_5" -> divide(1.25 * tp, 1.25 * tp + fp + 0.25 * fn),
      "accuracy" -> divide(tp + tn, tp + fp + tn + fn), "evaluation_rows" -> (tp + fp + tn + fn),
      "tp" -> tp, "fp" -> fp, "tn" -> tn, "fn" -> fn)
  }

  private def readPredictions(spark: SparkSession, path: String): DataFrame = {
    val data = spark.read.parquet(path).persist(StorageLevel.MEMORY_AND_DISK)
    require(data.count() > 0 && data.select("id").distinct().count() == data.count(), "Empty predictions or duplicate IDs")
    require(data.filter(col("id").isNull || col("label").isNull || !col("label").isin(0.0, 1.0) ||
      col("customer_probability").isNull || isnan(col("customer_probability")) ||
      col("customer_probability") < 0.0 || col("customer_probability") > 1.0).limit(1).count() == 0,
      "Invalid IDs, labels, or probabilities")
    require(data.select("label").distinct().count() == 2, "Both classes are required")
    data
  }

  private def changeThreshold(model: PipelineModel, threshold: Double): PipelineModel = {
    require(threshold > 0.0 && threshold < 1.0, "Cutoff must be between zero and one")
    val changed = model.copy(ParamMap.empty)
    changed.stages.last match {
      case m: GBTClassificationModel => m.setThresholds(Array(1.0 - threshold, threshold))
      case m: RandomForestClassificationModel => m.setThresholds(Array(1.0 - threshold, threshold))
      case m: LogisticRegressionModel => m.setThreshold(threshold)
      case _ => throw new IllegalArgumentException("Unsupported classifier")
    }
    changed
  }

  private def recordPath(record: Row): String = record.getAs[String]("model") + "/seed-" +
    record.getAs[Long]("seed") + "/fraction-" + math.round(record.getAs[Double]("requested_fraction") * 10000)

  def main(args: Array[String]): Unit = {
    if (args.contains("--help")) {
      println("CompareCustomerRecall --validation-run results/tune-... --final-run results/train-... " +
        "--data-dir data --output-dir results --minimum-recall 0.60 --model P1 --validation-seed 42")
      return
    }
    require(args.length % 2 == 0, "Use --name value pairs")
    val options = args.grouped(2).map(a => a(0) -> a(1)).toMap
    require(options.size * 2 == args.length && options.keySet.subsetOf(Set("--validation-run", "--final-run",
      "--data-dir", "--output-dir", "--minimum-recall", "--model", "--validation-seed")), "Duplicate or unknown option")
    val validationRun = options.getOrElse("--validation-run", throw new IllegalArgumentException("Provide --validation-run")).stripSuffix("/")
    val finalRun = options.getOrElse("--final-run", throw new IllegalArgumentException("Provide --final-run")).stripSuffix("/")
    val dataDir = options.getOrElse("--data-dir", "data").stripSuffix("/")
    val minimumRecall = options.getOrElse("--minimum-recall", "0.60").toDouble
    require(minimumRecall > 0.0 && minimumRecall < 1.0, "Minimum recall must be between zero and one")
    val modelName = options.getOrElse("--model", "P1").toUpperCase
    require(Set("P1", "P2", "P3")(modelName), "Select P1, P2, or P3")
    val validationSeed = options.getOrElse("--validation-seed", "42").toLong
    val output = options.getOrElse("--output-dir", "results").stripSuffix("/") + "/recall-" +
      Instant.now().toString.replaceAll("[:.]", "-") + "-" + UUID.randomUUID().toString.take(8)
    val spark = SparkSession.builder().appName("Customer Recall Operating Point").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", 8)
    try {
      require(spark.version.startsWith("3.5."), "Use Spark 3.5.x")
      def manifest(run: String): Row = spark.read.option("multiLine", "true").json(run + "/manifest.json").head()
      require(manifest(validationRun).getAs[String]("evaluation") == "validation", "Cutoffs must be selected from training validation, never project test results")
      val records = spark.read.option("multiLine", "true").json(validationRun + "/results.json")
        .filter(col("model") === modelName && col("seed") === validationSeed).collect()
      require(records.length == 1 && records.head.getAs[String]("evaluation_scope") == "training_validation_only",
        "Expected exactly one completed training-validation record")
      val record = records.head
      val predictions = readPredictions(spark, validationRun + "/" + recordPath(record) + "/predictions.parquet")
      try {
        val binary = new BinaryClassificationMetrics(predictions.select("customer_probability", "label").rdd
          .map(r => (r.getDouble(0), r.getDouble(1))), 0)
        val best = try {
          binary.fMeasureByThreshold(0.5).join(binary.recallByThreshold)
            .filter { case (threshold, (_, recall)) => threshold > 0.0 && recall >= minimumRecall }
            .map { case (threshold, (f05, _)) => (threshold, f05) }
            .fold((0.5, Double.NegativeInfinity)) { (a, b) =>
              if (a._2 > b._2 || (a._2 == b._2 && a._1 > b._1)) a else b
            }
        } finally binary.unpersist()
        require(best._2 > Double.NegativeInfinity, "Recall target is unattainable with a positive probability cutoff")
        val lower = predictions.filter(col("customer_probability") < best._1).agg(max("customer_probability")).head()
        val threshold = if (lower.isNullAt(0)) best._1 / 2.0 else (best._1 + lower.getDouble(0)) / 2.0
        require(threshold > 0.0 && threshold < 1.0, "Selected cutoff cannot be represented by the native classifier")
        val selected = metric(predictions, threshold)
        require(selected("customer_recall").asInstanceOf[Double] + 1e-12 >= minimumRecall, "Selected cutoff misses the validation recall target")
        require(math.abs(selected("customer_f0_5").asInstanceOf[Double] - best._2) < 1e-10, "Cutoff curve and confusion counts disagree")
        val oldThreshold = record.getAs[Double]("decision_threshold")
        val comparison = Seq(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, oldThreshold, threshold).distinct.sorted
          .map(t => metric(predictions, t) ++ Map("selected" -> (t == threshold), "evaluation_scope" -> "training_validation_exploration"))
        // Persist the policy before reading any final-test predictions or test-score records.
        TrainBikeShareModels.writeJson(spark, output + "/selected_recall_policy.json", Map(
          "model" -> modelName, "validation_seed" -> validationSeed, "minimum_validation_recall" -> minimumRecall,
          "selected_threshold" -> threshold, "selection_objective" -> "Highest validation Customer F0.5 subject to minimum recall",
          "selection_source" -> new Path(validationRun).getName, "test_used_for_selection" -> false,
          "validation_selection_metrics" -> selected,
          "interpretation" -> "Validation metrics describe cutoff selection; recall on another dataset is not guaranteed"))
        TrainBikeShareModels.writeJson(spark, output + "/validation_cutoff_comparison.json", comparison)
        import spark.implicits._
        val table = spark.read.json(spark.createDataset(comparison.map(mapper.writeValueAsString)))
        table.select("threshold", "selected", "customer_precision", "customer_recall", "customer_f0_5").show(20, false)
        println(s"Frozen validation-selected cutoff=$threshold; requested validation recall=$minimumRecall")

        require(manifest(finalRun).getAs[String]("evaluation") == "test", "Final run must contain completed fixed-test results")
        val finalRecords = spark.read.option("multiLine", "true").json(finalRun + "/results.json")
          .filter(col("model") === modelName).orderBy("seed").collect()
        require(finalRecords.nonEmpty, "No final results for the selected model")
        val enriched = new Path(dataDir + "/enriched/test.parquet")
        val activeData = if (enriched.getFileSystem(spark.sparkContext.hadoopConfiguration).exists(enriched)) dataDir + "/enriched" else dataDir
        val test = spark.read.parquet(activeData + "/test.parquet")
        require(predictions.select("id").join(test.select("id"), Seq("id")).limit(1).count() == 0,
          "Validation selection IDs overlap the project test set")
        val probe = test.orderBy("id").limit(25).drop("label")
        val results = finalRecords.map { finalRecord =>
          require(finalRecord.getAs[String]("evaluation_scope") == "fixed_test", "Expected fixed-test evaluation")
          require(finalRecord.getAs[String]("feature_profile") == record.getAs[String]("feature_profile") &&
            finalRecord.getAs[String]("duration_transform") == record.getAs[String]("duration_transform"), "Feature preprocessing differs")
          def representation(row: Row, field: String, fallback: String): String =
            if (row.schema.fieldNames.contains(field)) row.getAs[String](field) else fallback
          require(representation(record, "tree_encoding", "onehot") == representation(finalRecord, "tree_encoding", "onehot") &&
            representation(record, "feature_variant", "standard") == representation(finalRecord, "feature_variant", "standard"),
            "Category encoding or generated features differ")
          val a = record.getAs[Row]("classifier_parameters"); val b = finalRecord.getAs[Row]("classifier_parameters")
          Seq("maxDepth", "maxIter", "numTrees", "maxBins", "minInstancesPerNode", "regParam", "elasticNetParam",
            "featureSubsetStrategy", "stepSize", "subsamplingRate").filter(a.schema.fieldNames.contains).foreach { name =>
            require(a.getAs[String](name) == b.getAs[String](name), s"Frozen hyperparameters differ: $name")
          }
          val relative = recordPath(finalRecord)
          val saved = readPredictions(spark, finalRun + "/" + relative + "/predictions.parquet")
          try {
            require(saved.select("id", "label").except(test.select("id", "label")).limit(1).count() == 0 &&
              test.select("id", "label").except(saved.select("id", "label")).limit(1).count() == 0, "Final prediction IDs or labels differ from project test")
            val original = PipelineModel.load(finalRun + "/" + relative + "/model")
            val changed = changeThreshold(original, threshold)
            changed.write.save(output + "/" + relative + "/model")
            val reloaded = PipelineModel.load(output + "/" + relative + "/model")
            val check = reloaded.transform(probe).select(col("id"), col("prediction"),
              vector_to_array(col("probability")).getItem(1).as("reloaded_probability"))
              .join(saved.select("id", "customer_probability"), Seq("id"))
            require(check.count() == 25 && check.filter(abs(col("reloaded_probability") - col("customer_probability")) > 1e-12 ||
              col("prediction") =!= when(col("customer_probability") > threshold, 1.0).otherwise(0.0)).count() == 0,
              "Reloaded model probabilities or cutoff predictions disagree")
            saved.withColumn("prediction", when(col("customer_probability") > threshold, 1.0).otherwise(0.0))
              .write.mode("errorifexists").parquet(output + "/" + relative + "/predictions.parquet")
            val result = metric(saved, threshold) ++ Map("model" -> modelName, "seed" -> finalRecord.getAs[Long]("seed"),
              "evaluation_scope" -> "fixed_test_recall_operating_point", "source_run" -> new Path(finalRun).getName,
              "minimum_validation_recall" -> minimumRecall, "model_reload_verified" -> true,
              "original_threshold" -> finalRecord.getAs[Double]("decision_threshold"),
              "original_metrics" -> metric(saved, finalRecord.getAs[Double]("decision_threshold")))
            TrainBikeShareModels.writeJson(spark, output + "/" + relative + "/metrics.json", result)
            println(s"$modelName seed=${finalRecord.getAs[Long]("seed")}: recall=${result("customer_recall")} precision=${result("customer_precision")} F0.5=${result("customer_f0_5")}")
            result
          } finally saved.unpersist()
        }
        TrainBikeShareModels.writeJson(spark, output + "/results.json", results.toSeq)
        val summary = spark.read.json(spark.createDataset(results.toSeq.map(mapper.writeValueAsString)))
        summary.select("model", "seed", "threshold", "customer_precision", "customer_recall", "customer_f0_5", "accuracy")
          .coalesce(1).write.mode("errorifexists").option("header", "true").csv(output + "/summary.csv")
        println("COMPLETE: " + output)
      } finally predictions.unpersist()
    } finally spark.stop()
  }
}
