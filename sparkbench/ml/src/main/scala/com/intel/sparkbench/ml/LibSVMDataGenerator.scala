package com.intel.hibench.sparkbench.ml

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.intel.hibench.sparkbench.common.IOCommon
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random

object LibSVMDataGenerator {
  /**
   * Generate an RDD containing test data for SVM.
   *
   * @param sc SparkContext to use for creating the RDD.
   * @param nExamples Number of examples that will be contained in the RDD.
   * @param nFeatures Number of features to generate for each example.
   * @param nParts Number of partitions of the generated RDD. Default value is 2.
   */
  def generateSVMRDD(
                      sc: SparkContext,
                      nExamples: Int,
                      nFeatures: Int,
                      nParts: Int = 2): RDD[LabeledPoint] = {
    val globalRnd = new Random(42)
    val trueWeights = Array.fill[Double](nFeatures)(globalRnd.nextGaussian())
    val data: RDD[LabeledPoint] = sc.parallelize(0 until nExamples,nParts).map { idx =>
      val rnd = new Random(42 + idx)

      val x = Array.fill[Double](nFeatures) {
        rnd.nextDouble() * 2.0 - 1.0
      }
      val yD = blas.ddot(trueWeights.length, x, 1, trueWeights, 1) + rnd.nextGaussian() * 0.1
      val y = if (yD < 0) 0.0 else 1.0
      LabeledPoint(y, Vectors.dense(x))
    }
    data
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LibSVMDataGenerator")
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
        s"Usage: $LibSVMDataGenerator <OUTPUT_PATH> <NUM_EXAMPLES> <NUM_FEATURES>"
      )
      System.exit(1)
    }

    val data = generateSVMRDD(sc, numExamples, numFeatures, numPartitions)

    MLUtils.saveAsLibSVMFile(data, outputPath)

    sc.stop()
  }
}
