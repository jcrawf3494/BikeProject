import java.time.Instant
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.hadoop.fs.Path

/** Adds only the proposal's station-cluster and daily-weather features.
  * The original saved training/test trip membership is preserved exactly.
  */
object PrepareBikeShareFeatures {
  val weatherNumeric: Seq[String] = Seq(
    "max_temperature_f", "mean_temperature_f", "min_temperature_f",
    "max_dew_point_f", "mean_dew_point_f", "min_dew_point_f",
    "max_humidity", "mean_humidity", "min_humidity",
    "max_sea_level_pressure_inches", "mean_sea_level_pressure_inches", "min_sea_level_pressure_inches",
    "max_visibility_miles", "mean_visibility_miles", "min_visibility_miles",
    "max_wind_speed_mph", "mean_wind_speed_mph", "max_gust_speed_mph",
    "precipitation_inches", "cloud_cover", "wind_dir_degrees")
  val weatherDerived: Seq[String] = Seq("precipitation_trace", "weather_fog", "weather_rain", "weather_thunder")
  val addedNumeric: Seq[String] = weatherNumeric ++ weatherDerived ++ Seq("is_business_day", "is_inter_cluster")
  val addedCategorical: Seq[String] = Seq("start_cluster", "end_cluster", "weather_events")

  def build(spark: SparkSession, inputDir: String, rawDir: String, outputDir: String): Unit = {
    import spark.implicits._
    val destination = new Path(outputDir)
    require(!destination.getFileSystem(spark.sparkContext.hadoopConfiguration).exists(destination),
      "Enriched output already exists; select a new output directory to preserve previous data")
    val stations = spark.read.option("header", "true").csv(rawDir + "/station.csv")
      .select(col("id").cast("int").as("station_id"), col("lat").cast("double"),
        col("long").cast("double"), col("city"))
    require(stations.count() == 70 && stations.select("station_id").distinct().count() == 70,
      "Expected 70 distinct stations")
    require(stations.filter(col("station_id").isNull || col("lat").isNull || col("long").isNull || col("city").isNull).count() == 0,
      "Invalid station metadata")
    // Station locations are public, static metadata; no trip labels are used for clustering.
    val vectors = new VectorAssembler().setInputCols(Array("lat", "long")).setOutputCol("station_coordinates").transform(stations)
    val km = new KMeans().setK(5).setSeed(42L).setMaxIter(100).setTol(1e-6)
      .setFeaturesCol("station_coordinates").setPredictionCol("station_cluster").fit(vectors.orderBy("station_id"))
    val clusters = km.transform(vectors).select("station_id", "city", "station_cluster")
    require(clusters.select("station_cluster").distinct().count() == 5, "K-means did not produce five populated clusters")

    val rawWeather0 = spark.read.option("header", "true").csv(rawDir + "/weather.csv")
    val rawWeather = rawWeather0.toDF(rawWeather0.columns.map(_.toLowerCase): _*)
    val cityZip = Seq("94107" -> "San Francisco", "95113" -> "San Jose", "94041" -> "Mountain View",
      "94301" -> "Palo Alto", "94063" -> "Redwood City").toDF("weather_zip", "weather_city")
    require(rawWeather.count() == 3665, "Unexpected weather row count; verify the dataset version")
    require(rawWeather.select(trim(col("zip_code")).as("weather_zip")).distinct()
      .join(cityZip, Seq("weather_zip"), "left_anti").count() == 0, "Unmapped weather ZIP code")
    val traceRows = rawWeather.filter(upper(trim(col("precipitation_inches"))) === "T").count()
    val parsedDate = coalesce(to_date(col("date"), "M/d/yyyy"), to_date(col("date"), "yyyy-MM-dd"))
    var weather = rawWeather.withColumn("trip_date", date_format(parsedDate, "yyyy-MM-dd"))
      .withColumn("weather_zip", trim(col("zip_code")))
      .join(broadcast(cityZip), Seq("weather_zip"), "left")
      .withColumn("precipitation_trace", when(upper(trim(col("precipitation_inches"))) === "T", 1.0).otherwise(0.0))
      .withColumn("weather_events", when(col("events").isNull || length(trim(col("events"))) === 0, "none")
        .otherwise(regexp_replace(lower(trim(col("events"))), "\\s*-\\s*", "-")))
      .withColumn("weather_fog", when(col("weather_events").contains("fog"), 1.0).otherwise(0.0))
      .withColumn("weather_rain", when(col("weather_events").contains("rain"), 1.0).otherwise(0.0))
      .withColumn("weather_thunder", when(col("weather_events").contains("thunder"), 1.0).otherwise(0.0))
    weatherNumeric.foreach { name =>
      val text = trim(col(name))
      val isTrace = if (name == "precipitation_inches") upper(text) === "T" else lit(false)
      require(weather.filter(length(text) > 0 && !isTrace && text.cast("double").isNull).count() == 0,
        s"Unexpected nonnumeric weather value in $name")
      weather = weather.withColumn(name,
        if (name == "precipitation_inches") when(upper(text) === "T", 0.0).otherwise(text.cast("double"))
        else text.cast("double"))
    }
    require(weather.filter(col("trip_date").isNull || col("weather_city").isNull).count() == 0, "Invalid weather date/city")
    require(weather.groupBy("weather_city", "trip_date").count().filter(col("count") =!= 1).count() == 0,
      "Duplicate daily weather keys would duplicate trips")
    weather = weather.select((Seq(col("weather_city").as("start_city"), col("trip_date"), lit(1).as("weather_matched")) ++
      (weatherNumeric ++ weatherDerived).map(col) :+ col("weather_events")): _*)

    val reports = scala.collection.mutable.ArrayBuffer.empty[Map[String, Any]]
    Seq("train", "test").foreach { split =>
      val original = spark.read.parquet(inputDir + "/" + split + ".parquet")
      val count = original.count()
      require(original.select("id").distinct().count() == count, s"Duplicate original $split IDs")
      val stationMatch = original.select("start_station_id", "start_city", "start_lat", "start_long").distinct()
        .join(stations, col("start_station_id") === col("station_id"), "left")
      require(stationMatch.filter(col("station_id").isNull || col("start_city") =!= col("city") ||
        abs(col("start_lat") - col("lat")) > 1e-8 || abs(col("start_long") - col("long")) > 1e-8).count() == 0,
        "Station CSV disagrees with the supplied trip features")
      val starts = clusters.select(col("station_id").as("start_station_id"), col("station_cluster").as("start_cluster"))
      val ends = clusters.select(col("station_id").as("end_station_id"), col("station_cluster").as("end_cluster"))
      val enriched = original.join(broadcast(starts), Seq("start_station_id"), "left")
        .join(broadcast(ends), Seq("end_station_id"), "left")
        .join(broadcast(weather), Seq("start_city", "trip_date"), "left")
        .withColumn("is_business_day", when(col("is_weekend") === 0 && col("is_holiday") === 0, 1.0).otherwise(0.0))
        .withColumn("is_inter_cluster", when(col("start_cluster") =!= col("end_cluster"), 1.0).otherwise(0.0)).cache()
      require(enriched.count() == count && enriched.select("id").distinct().count() == count, "A join changed trip membership")
      require(enriched.select("id", "label").except(original.select("id", "label")).count() == 0 &&
        original.select("id", "label").except(enriched.select("id", "label")).count() == 0, "Trip IDs or labels changed")
      require(enriched.filter(col("start_cluster").isNull || col("end_cluster").isNull).count() == 0, "Unmatched station cluster")
      val missingWeather = enriched.filter(col("weather_matched").isNull).count()
      require(missingWeather == 0, s"$missingWeather trips have no matching city/date weather")
      val nullCounts = enriched.agg(addedNumeric.map(c => sum(when(col(c).isNull, 1L).otherwise(0L)).as(c)).head,
        addedNumeric.tail.map(c => sum(when(col(c).isNull, 1L).otherwise(0L)).as(c)): _*).head()
      val missing = addedNumeric.zipWithIndex.map { case (c, i) => c -> nullCounts.getLong(i) }.toMap
      // Do not fill numeric values here: training-fold imputation belongs in the classifier pipeline.
      enriched.drop("weather_matched").write.mode("errorifexists").parquet(outputDir + "/" + split + ".parquet")
      reports += Map("split" -> split, "rows" -> count, "weather_unmatched" -> missingWeather,
        "numeric_nulls_before_training_imputation" -> missing, "ids_and_labels_preserved" -> true)
      enriched.unpersist()
      println(s"Enriched $split: $count rows; zero unmatched station/weather joins")
    }
    km.write.save(outputDir + "/station_kmeans_model")
    clusters.coalesce(1).write.mode("errorifexists").option("header", "true").csv(outputDir + "/station_clusters.csv")
    val train = spark.read.parquet(outputDir + "/train.parquet")
    val test = spark.read.parquet(outputDir + "/test.parquet")
    require(train.select("id").join(test.select("id"), Seq("id")).count() == 0, "Enriched splits overlap")
    TrainBikeShareModels.writeJson(spark, outputDir + "/feature_audit.json", Map(
      "created_utc" -> Instant.now().toString, "station_count" -> 70, "cluster_count" -> 5, "cluster_seed" -> 42,
      "weather_rows" -> 3665, "trace_rain_rows" -> traceRows, "numeric_features_added" -> addedNumeric,
      "categorical_features_added" -> addedCategorical, "split_audits" -> reports.toSeq,
      "clustering_inputs" -> "Static station latitude and longitude only; no trip labels",
      "imputation" -> "Deferred to each model fitting fold; no test-derived replacement values"))
    println("FEATURE PREPARATION COMPLETE: " + outputDir)
  }

  def main(args: Array[String]): Unit = {
    require(args.length % 2 == 0, "Use --data-dir path --raw-dir path --output-dir path")
    val opts = args.grouped(2).map(a => a(0) -> a(1)).toMap
    require(opts.keySet.subsetOf(Set("--data-dir", "--raw-dir", "--output-dir")), "Unknown feature-preparation option")
    val data = opts.getOrElse("--data-dir", "data")
    val spark = SparkSession.builder().appName("Bike Share Feature Preparation").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", 4)
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    try build(spark, data, opts.getOrElse("--raw-dir", data + "/raw"), opts.getOrElse("--output-dir", data + "/enriched"))
    finally spark.stop()
  }
}
