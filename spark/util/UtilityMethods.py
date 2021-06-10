import numpy as np


def parseLine(line):
    return np.array(line.split(','), dtype=np.float64)


def getClosestCentroid(point, centroids):
    squaredDistance = np.sum(((np.array(centroids) - point) ** 2), axis=1)
    targetMeanIndex = np.where(squaredDistance == squaredDistance.min())[0][0]

    return tuple(centroids[targetMeanIndex]), (point, 1)


def getPointsSum(pointsBinder1, pointsBinder2):
    # Each binder tuple has the format (sum_of_points, number_of_points)
    return pointsBinder1[0] + pointsBinder2[0], pointsBinder1[1] + pointsBinder2[1]


def computeCentroid(pointsBinder):
    # (sum_of_points, number_of_points)
    newCentroid = pointsBinder[1][0]/pointsBinder[1][1]

    return newCentroid


def computeMinSquaredDistance(point, centroids):
    squaredDistance = np.sum(((np.array(centroids) - point) ** 2), axis=1)
    #evaluate the squared distances w.r.t each centoid and return the minimun
    return squaredDistance.min()


def toString(point):
    return np.array2string(point, separator=',')[1:-1].replace(' ', '')