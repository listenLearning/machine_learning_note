#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类,所有在阈值一边的数据会分到类别-1,而在另外一边的数据分到类别+1
    :param dataMatrix:数据集
    :param dimen:特征列
    :param threshVal:特征比较值
    :param threshIneq:
    :return:
    '''
    # 初始化返回组,将其全部元素填充为1
    retArray = ones((shape(dataMatrix)[0], 1))
    # 根据threshIneq符号比较dataMatrix与threshVal大小,以便修改retArray的值
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    遍历得到stumpClassify函数所有可能的输入值,并找到数据集上最佳的单层决策树
    这里的最佳时基于数据的权重向量D来定义的
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    # 将输入数组转换成矩阵,并获取样本数与特征数
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # numSteps用于在所有可能的特征值上进行遍历
    numSteps = 10.0
    # 初始化用于存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    # 初始化最小错误率为正无穷大,用于之后寻找最小错误率
    minError = inf
    # 遍历数据集上的所有特征
    for i in range(n):
        # 计算数据集的最大值与最小值来了解需要多大的步长
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 根据每个步长进行遍历
        for j in range(-1, int(numSteps) + 1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                # 计算特征列要比较的值
                threshVal = (rangeMin + float(j) * stepSize)
                # 调用stumpClassify函数,返回分类预测结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化错误向量
                errArr = mat(ones((m, 1)))
                # 将errArr中真实值与预测值相等的部分值修改为0
                errArr[predictedVals == labelMat] = 0
                # 错误向量errArr和权重向量的相应元素相乘并求和
                # weightedError是AdaBoost和分类器交互的地方
                weightedError = D.T * errArr
                print('split:dim %d, thresh %.2f, thresh ineqal: %s,the '
                      'weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                # 将当前错误率与已有的最小错误率进行对比，如果当前值比较小
                # 就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回最小错误率的单层决策树，最小错误率与类别估计值
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    基于单层决策树的AdaBoost训练过程
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return:
    '''
    # 创建一个新的列表用与存储输出的单层决策树组
    weakClassArr = []
    # 获取样本数
    m = shape(dataArr)[0]
    # 初始化一个权重向量D，D包含了每个数据点的权重,并且这些权重都赋予了相等的值
    D = mat(ones((m, 1)) / m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 调用buildStump函数建立一个单层决策函数,输入权重向量D
        # 返回利用D得到的具有最小错误率的单层决策树,同时返回最小的错误率以及估计的类别向量
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        # 计算权重值,\alpha=\dfrac{1}{2}\ln\left(\frac{1-\epsilon}{\epsilon}\right)
        # alpha会告诉总分类器本次单层决策树输出结果的权重
        # max(error, 1e-16)用于去额宝在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 将alpha值更新到单层决策树列表中,并且将该字典追加到weakClassArr列表中
        # bestStump字典包含了分类所需要的所有的信息
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: ', classEst.T)
        # 计算下一次迭代中的新权重向量D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 保持一个运行时的类别估计值
        aggClassEst += alpha * classEst
        print('aggClassEst: ', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error: ', errorRate, '\n')
        # 如果训练错误率为0,就提前结束循环
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    '''
    AdaBoost分类函数
    :param datToClass: 一个或多个待分类样例
    :param classifierArr: 多个弱分类器
    :return:
    '''
    dataMatrix = mat(datToClass)
    m, n = shape(dataMatrix)
    # 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    # 遍历classifierArr中的所有弱分类器
    for i in range(len(classifierArr)):
        # 调用stumpClassify对每个分类器得到一个类别估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 输出的类别估计值乘上该单层决策树的alpha权重然后累加到aggClassEst
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print('aggClassEst: ', aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Postive Rare')
    plt.ylabel('True Postive Rare')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is: ', ySum * xStep)
