#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    计算最佳拟合直线
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).transpose()
    # 获取样本数
    m = shape(xMat)[0]
    # 创建对角权重矩阵
    # 权重矩阵是一个方阵,阶数等于样本点个数,
    # 即:该矩阵为每个样本点初始化了一个权重
    weights = mat(eye((m)))
    # 遍历所有数据集,计算每个样本点对应的权重
    for j in range(m):
        # 计算样本点与带预测点距离
        diffMat = testPoint - xMat[j, :]
        # 随着距离的递增,权重将以指数级衰减,可以使用K控制衰减的速度
        weights[j, j] = exp(diffMat * diffMat.transpose() / (-2.0 * k ** 2))
    # 计算X^TX
    xTx = xMat.T * (weights * xMat)
    # 判断X^TX的行列式是否为零,如果为零,会导致计算逆矩阵的时候出现错误,直接返回
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    # 获取当前可以估计出的w最优解
    ws = xTx.I * (xMat.T * (weights * yMat))
    # 返回对回归系数ws的一个估计
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    给定x空间中的任意一点,计算对应的预测值yHat
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotLwlr(xArr, yArr, yHat):
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure(111)
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()
