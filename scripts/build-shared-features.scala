//
// Builds the shared feature table and the fixed train/test split that every
// model in the project trains and is scored on.
//
//   trip.csv + station.csv  ->  features.parquet  ->  train / test parquet
//
//
// Run from the project root:
//   spark-shell --driver-memory 4g -i src/02_build_features.scala

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val ROOT = sys.env.getOrElse("BIKESHARE_RAW", "data/raw/archive")
val OUT = sys.env.getOrElse("BIKESHARE_BASE_OUTPUT", "data/base-new")
val SEED = 42L

// ---------------------------------------------------------------- ingestion
// Schemas are declared explicitly rather than inferred: inferSchema makes an
// extra pass over 76 MB and still guesses the date columns wrong.
val tripSchema = StructType(Seq(
  StructField("id",                 IntegerType),
  StructField("duration",           IntegerType),
  StructField("start_date",         StringType),
  StructField("start_station_name", StringType),
  StructField("start_station_id",   IntegerType),
  StructField("end_date",           StringType),
  StructField("end_station_name",   StringType),
  StructField("end_station_id",     IntegerType),
  StructField("bike_id",            IntegerType),
  StructField("subscription_type",  StringType),
  StructField("zip_code",           StringType)))

val stationSchema = StructType(Seq(
  StructField("id",                IntegerType),
  StructField("name",              StringType),
  StructField("lat",               DoubleType),
  StructField("long",              DoubleType),
  StructField("dock_count",        IntegerType),
  StructField("city",              StringType),
  StructField("installation_date", StringType)))

val trips = spark.read.option("header", "true").schema(tripSchema).csv(s"$ROOT/trip.csv")
val stations = spark.read.option("header", "true").schema(stationSchema).csv(s"$ROOT/station.csv")

println(s"trips loaded   : ${trips.count()}%,d".replace("%,d", ""))
println(s"stations loaded: ${stations.count()}")

// ------------------------------------------------------------ parse the dates
// Dates are text like "8/29/2013 14:13" - single-digit month and day, 24h clock.
spark.conf.set("spark.sql.session.timeZone", "UTC")
val TS = "M/d/yyyy H:mm"
val withTime = trips
  .withColumn("start_ts", to_timestamp($"start_date", TS))
  .withColumn("end_ts",   to_timestamp($"end_date",   TS))

// Fail loudly rather than silently dropping rows if the format ever changes.
val unparsed = withTime.filter($"start_ts".isNull || $"end_ts".isNull).count()
require(unparsed == 0, s"$unparsed rows failed timestamp parsing - check the format")
println(s"timestamps parsed, 0 failures")

// --------------------------------------------------------------- the join
// Join each trip to its START station to pick up city and coordinates.
// Verified: every start_station_id in trip.csv exists in station.csv.
val startSt = stations.select(
  $"id".as("s_id"), $"city".as("start_city"),
  $"lat".as("start_lat"), $"long".as("start_long"),
  $"dock_count".as("start_dock_count"))

val joined = withTime.join(startSt, $"start_station_id" === $"s_id", "left").drop("s_id")

val orphans = joined.filter($"start_city".isNull).count()
require(orphans == 0, s"$orphans trips did not match a station")
println("join to station: 0 orphans")

// ------------------------------------------------------------- the features
// US federal holidays inside the data window (Aug 2013 - Aug 2015).
val holidays = Set(
  "2013-09-02", "2013-10-14", "2013-11-11", "2013-11-28", "2013-12-25",
  "2014-01-01", "2014-01-20", "2014-02-17", "2014-05-26", "2014-07-04",
  "2014-09-01", "2014-10-13", "2014-11-11", "2014-11-27", "2014-12-25",
  "2015-01-01", "2015-01-19", "2015-02-16", "2015-05-25", "2015-07-03")
val isHoliday = udf((d: String) => if (d != null && holidays.contains(d)) 1 else 0)

val features = joined
  .withColumn("trip_date",      date_format($"start_ts", "yyyy-MM-dd"))
  .withColumn("duration_sec",   $"duration")
  .withColumn("duration_min",   round($"duration" / 60.0, 2))
  .withColumn("start_hour",     hour($"start_ts"))
  .withColumn("day_of_week",    dayofweek($"start_ts"))          // 1 = Sunday
  .withColumn("month",          month($"start_ts"))
  .withColumn("is_weekend",     when($"day_of_week".isin(1, 7), 1).otherwise(0))
  .withColumn("is_holiday",     isHoliday($"trip_date"))
  .withColumn("is_round_trip",  when($"start_station_id" === $"end_station_id", 1).otherwise(0))
  // label: Customer (the minority class we care about) is positive
  .withColumn("label",          when($"subscription_type" === "Customer", 1.0).otherwise(0.0))
  .select(
    $"id", $"label", $"subscription_type",
    $"duration_sec", $"duration_min",
    $"start_hour", $"day_of_week", $"month", $"is_weekend", $"is_holiday",
    $"is_round_trip",
    $"start_station_id", $"end_station_id",
    $"start_city", $"start_lat", $"start_long", $"start_dock_count",
    $"trip_date", $"start_ts")

//   (start_city, trip_date) -> weather columns, then re-run the split below.

features.cache()
val total = features.count()
println(s"feature rows: $total")

println("\n=== class balance ===")
features.groupBy("subscription_type", "label").count()
  .withColumn("pct", round($"count" / total * 100, 2))
  .orderBy(desc("count")).show(false)

println("=== date range ===")
features.select(min($"start_ts").as("first_trip"), max($"start_ts").as("last_trip")).show(false)

println("=== trips per city ===")
features.groupBy("start_city").count().orderBy(desc("count")).show(false)

// ------------------------------------------------- fixed stratified split
// sampleBy approximately preserves the class ratio; actual counts must be verified.
// Everyone on the team must use THESE files, or the model comparison is invalid.
val fractions = Map(1.0 -> 0.8, 0.0 -> 0.8)
val train = features.stat.sampleBy("label", fractions, SEED)
val test  = features.join(train.select($"id".as("t_id")), $"id" === $"t_id", "left_anti")

train.cache(); test.cache()
val nTrain = train.count()
val nTest  = test.count()
println(f"\ntrain rows: $nTrain%d (${nTrain * 100.0 / total}%.2f%%)")
println(f"test  rows: $nTest%d (${nTest * 100.0 / total}%.2f%%)")
require(nTrain + nTest == total, "split lost or duplicated rows")

println("=== positive-class rate in each split (should match) ===")
Seq(("all", features), ("train", train), ("test", test)).foreach { case (nm, df) =>
  val n = df.count()
  val pos = df.filter($"label" === 1.0).count()
  println(f"  $nm%-6s n=$n%-8d customer=$pos%-7d rate=${pos * 100.0 / n}%.3f%%")
}

// --------------------------------------------------------------- write out
features.write.mode("errorifexists").parquet(s"$OUT/features.parquet")
train.write.mode("errorifexists").parquet(s"$OUT/train.parquet")
test.write.mode("errorifexists").parquet(s"$OUT/test.parquet")
println(s"\nwrote $OUT/features.parquet, train.parquet, test.parquet (seed = $SEED)")
println("=== BUILD COMPLETE ===")
