import time

from pyspark import SparkContext
from util import UtilityMethods
from util.Configurator import Configurator


def main():
    config = Configurator("config.ini")
    sc = SparkContext()

    # flush output files.
    outputFile = config.getOutputPath() + "/centroids"
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    fs = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    fs.delete(Path(outputFile))


    initialTimeInstant = round(time.time() * 1000)
    inputPoints = sc.textFile(config.getInputPath()).map(UtilityMethods.parseLine).cache()

    # randomly select first centroids.
    sampledCentroids = inputPoints.takeSample(False, config.getK(), 1)

    # update the centroids
    brCentroids = sc.broadcast(sampledCentroids)
    oldCumulativeError = float("inf")
    iteration = 0

    while iteration < config.getMaxIterations():
        newCentroids = inputPoints.map(lambda point: UtilityMethods.getClosestCentroid(point, brCentroids.value))\
                            .reduceByKey(lambda x, y: UtilityMethods.getPointsSum(x, y))\
                            .map(lambda accumulator: UtilityMethods.computeCentroid(accumulator))\
                            .collect()

        brCentroids = sc.broadcast(newCentroids)
        cumulativeError = inputPoints.map(lambda point: UtilityMethods.computeMinSquaredDistance(point, brCentroids.value)).sum()

        if verifyStopCondition(cumulativeError, oldCumulativeError, iteration, config.getThreshold()):
            sc.parallelize(newCentroids).map(UtilityMethods.toString).saveAsTextFile(config.getOutputPath() + "/centroids")
            sc.stop()
            finalTimeInstant = round(time.time() * 1000)
            executionTimeInterval = finalTimeInstant - initialTimeInstant
            executionTimeIntervalInSeconds = executionTimeInterval/1000
            print("*************** Execution time: " + str(executionTimeIntervalInSeconds) + " ***************")
            return

        iteration += 1
        oldCumulativeError = cumulativeError

    print("*************** The maximum number of allowed iterations has been reached. Algorithm is stopped. ***************")
    finalTimeInstant = round(time.time() * 1000)
    executionTimeInterval = finalTimeInstant - initialTimeInstant
    executionTimeIntervalInSeconds = executionTimeInterval / 1000
    print("*************** Execution time: " + str(executionTimeIntervalInSeconds) + " ***************")
    sc.stop()


def verifyStopCondition(cumulativeError, oldCumulativeError, iteration, threshold):
    print("*************** Iteration: " + str(iteration + 1) + " ***************")
    print("*************** Old cumulative error: " + str(oldCumulativeError) + " ***************")
    print("*************** Cumulative error: " + str(cumulativeError) + " ***************")

    percentageError = ((oldCumulativeError - cumulativeError)/oldCumulativeError)*100
    print("*************** Current error percentage: " + str(percentageError) + "% ***************")

    if percentageError <= threshold:
        print("*************** The stop criterion has been verified. Actual error percentage: " + str(percentageError) + "% ***************\n")
        return True

    return False



if __name__ == "__main__":
    main()