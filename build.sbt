name := "recommender-spark"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.1.0"
//libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka" % "0.10.0"

// mongodb for scala 2.12 https://oss.sonatype.org/content/repositories/releases/org/mongodb/casbah_2.12/
libraryDependencies += "org.mongodb" %% "casbah" % "3.1.1"
libraryDependencies += "org.jblas" % "jblas" % "1.2.4"