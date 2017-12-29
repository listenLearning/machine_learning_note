#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def ridgeRegress(xMat, yMat, lam=0.2):
    '''
    实现了给定lambda下的岭回归求解,如果没有指定lambda,则默认为0.2
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    '''
    # 构造X^TX矩阵
    xTx = xMat.T * xMat
    # 使用lambda乘以单位矩阵(可以调用eye来生成)
    denom = xTx * eye(shape(xMat)[1]) * lam
    # 判断X^TX的行列式是否为零,如果为零,会导致计算逆矩阵的时候出现错误,直接返回
    if linalg.det(denom) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    # 如果矩阵非奇异就计算回归系数并返回
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''
    为了方便使用岭回归和缩减技术,需要对特征做标准化处理
    具体做法就是所有特征都减去各自的均值并除以方差
    :param xArr:
    :param yArr:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).transpose()
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    # 初始化lambda数目
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    # 在numTestPts个不同的lambda下调用ridgeRegress函数
    for i in range(numTestPts):
        # 需要注意的是,这里的lambda应以指数级变化,
        # 这样可以看出lambda在取非常小的值时和取非常大的值时分别对结果造成的影响
        ws = ridgeRegress(xMat, yMat, exp(i - 10))
        # 将所有的回归系数输出到一个矩阵
        wMat[i, :] = ws.T
    return wMat


def ridgePlot(ridgeWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
