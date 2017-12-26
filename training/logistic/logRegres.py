#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def loadDataSet():
    '''
    加载文本文件并逐行读取,每行前两个值分别是x1和x2,第三个值是数据对应的类别标签
    为了方便计算,将x0的值设置为默认的1.0
    :return:
    '''
    dataMat = []
    labelMat = []
    fr = open('../../data/logistic/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    '''
    sigmoid函数
    :param inX:
    :return:
    '''
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    Logistic回归梯度上升优化算法
    :param dataMatIn: 2维Numpy数组,每列分别代表每个不同的特征,每行则代表每个训练样本
    :param classLabels: 类别标签
    :return:
    '''
    dataMatrix = mat(dataMatIn)
    # 将分类标签转换为numpy矩阵,并转置
    labelMat = mat(classLabels).T
    # m 样本数
    # n 特征数
    m, n = shape(dataMatrix)
    # 目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 回归系数,此处创建长度和特征数相同的矩阵
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        # 计算真是类别与预测类别的差值
        errors = (labelMat - h)
        # 按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.T * errors
    return array(weights)


def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    '''
    m, n = shape(dataMatrix)
    # 目标移动的步长
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 计算真是类别与预测类别的差值,此时的误差值为向量
        error = classLabels[i] - h
        # 按照差值的方向调整回归系数
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=225):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次迭代的时候调整步长,可以缓解数据波动或者高频波动
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取样本更新回归系数,可以有效减小周期性波动
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    '''
    以回归系数和特征向量作为输入来计算对应的sigmoid值,
    如果sigmoid值大于0.5,函数返回1,否则返回0
    :param inX:
    :param weights:
    :return:
    '''
    prob = sigmoid(sum(inX * weights))
    if prob < 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    '''
    打开测试集和训练集,并对数据进行格式化处理
    :return:
    '''
    # 导入数据集,数据的最后一列是类别标签
    # 类别标签分别是:'未能存活'，仍存活'
    frTrain = open('../../data/logistic/horseColicTraining.txt')
    frTest = open('../../data/logistic/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        # 判断当前是否为类别标签
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 调用stocGradAscent1函数计算回归系数向量
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 导入测试集
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    # 计算分类错误率
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is :%f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is :%f' % (numTests, errorSum / float(numTests)))


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
