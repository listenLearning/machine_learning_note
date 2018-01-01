#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from training.action.regression import regression


def regularize(xMat):
    inMat = xMat.copy()
    # 沿矩阵纵轴计算矩阵元素的平均值
    inMeans = mean(inMat, 0)
    # 沿矩阵纵轴计算矩阵元素的方差
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    逐步线性回归算法,本算法与lasso算法相近但计算更加简单
    :param xArr: 矩阵数据
    :param yArr: 预测变量
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    # 初始化一个以0填充的矩阵，用于保存回归系数
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    # 首先迭代numIt次
    for i in range(numIt):
        # 打印w, 用于分析算法执行的过程和效果
        print(ws.T)
        # 初始化lowestError
        lowestError = inf
        # 在所有特征上运行两次for循环
        for j in range(n):
            # 分别计算增加或减少该特征对误差的影响
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                # 调用rssError函数获取平方误差
                rssE = regression.rssError(yMat.A, yTest.A)
                # 比较新获取的平方误差与初始误差值，取最小误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
