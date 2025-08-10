package com.intel.hibench.sparkbench.ml

import com.intel.hibench.sparkbench.common.IOCommon

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.random.RandomRDDs

object BlazeSVDPPDataGenerator {
  def generateDistributedRowMatrix(
                                    sc: SparkContext,
                                    m: Long,
                                    n: Int,
                                    numPartitions: Int,
                                    seed: Long = System.currentTimeMillis()): RDD[Vector] = {
    val data: RDD[Vector] = RandomRDDs.normalVectorRDD(sc, m, n, numPartitions, seed)
    data
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("BlazeSVDPPDataGenerator")
    val sc = new SparkContext(conf)

    var outputPath = ""
    var numExamples: Int = 200000
    var numFeatures: Int = 20
    val parallel = sc.getConf.getInt("spark.default.parallelism", sc.defaultParallelism)
    val numPartitions = IOCommon.getProperty("hibench.default.shuffle.parallelism")
      .getOrElse((parallel / 2).toString).toInt

    if (args.length == 3) {
      outputPath = args(0)
      numExamples = args(1).toInt
      numFeatures = args(2).toInt
      println(s"Output Path: $outputPath")
      println(s"Num of Examples: $numExamples")
      println(s"Num of Features: $numFeatures")
    } else {
      System.err.println(
        s"Usage: $SVDDataGenerator <OUTPUT_PATH> <NUM_EXAMPLES> <NUM_FEATURES>"
      )
      System.exit(1)
    }

    val data = generateDistributedRowMatrix(sc, numExamples, numFeatures, numPartitions)

    data.zipWithIndex()
      .flatMap {
        case (feature, id) => feature.toArray.zipWithIndex.map {
          case (value, featureId) => s"${id}::${featureId}::${value}"
        }
      }
      .coalesce(1)
      .saveAsTextFile(outputPath)

    sc.stop()
  }
}
