#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def rssError(yArr, yHatArr):
    '''
    获取平方误差值
    :param yArr:
    :param yHatArr:
    :return:
    '''
    return ((yArr - yHatArr) ** 2).sum()


def loadDataSet(fileName):
    '''
    数据导入函数
    :param fileName:
    :return:
    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''
    计算最佳拟合直线
    :param xArr:
    :param yArr:
    :return: 回归系数
    '''
    # 读入x和y并将它们保存到矩阵中
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 计算X^TX
    xTx = xMat.transpose() * xMat
    # 判断X^TX的行列式是否为零,如果为零,会导致计算逆矩阵的时候出现错误,直接返回
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    # 获取当前可以估计出的w最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws


def plotRegres(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    fig = plt.figure(111)
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
