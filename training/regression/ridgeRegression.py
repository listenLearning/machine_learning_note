#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def ridgeRegress(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx * eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).transpose()
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def ridgePlot(ridgeWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
