ThisBuild / scalaVersion := "2.12.18"
name := "bikeshare-models"
version := "1.0.0"
Compile / unmanagedSourceDirectories += baseDirectory.value / "src"
Compile / mainClass := Some("TrainBikeShareModels")
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.5.9" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.9" % "provided"
)
