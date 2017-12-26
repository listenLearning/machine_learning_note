#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np
import operator
from numpy import *
from matplotlib import pyplot as plt
import os


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    # # inX:用于分类的输入向量
    # # dataSet:输入的训练样本集
    # # labels:标签向量
    # # k:用于选择最近邻居的数目
    # 计算一直类别数据集中的点与当前点之间的距离
    # 欧式距离公式: d=\sqrt{(xA_0-xB_0)^2+(xA_1-xB_1)^2}
    dataSetSize = dataSet.shape[0]
    # tile 拷贝,dataSetSize-重复几行,1-列方向上重复几次
    diff = tile(inX, (dataSetSize, 1))
    # 计算inX与dataSet点与点之间的距离=>(xA_0-xB_0)
    diffMat = diff - dataSet
    # 将得到的两点距离开方
    sqDiffMat = power(diffMat, 2)
    # 将开方之后的距离值相加得到同一行的距离值
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号,得到最终距离
    distances = sqrt(sqDistances)
    # 按顺序排列distances，返回相关的索引下标
    # 按照距离递增次序排列
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 选取与当前点距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount['A'] = 0+1,添加元素到字典
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 确定前k个点所在类别的出现频率
    # sorted函数用来排序，sorted(iterable[, cmp[, key[, reverse]]])
    # 其中key的参数为一个函数或者lambda函数(key=lambda item: item[1])。所以itemgetter可以用来当key的参数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回前k个点出现频率最高的类别作为当前点的预测分类
    return sortedClassCount[0][0]


# 将文本记录转化为numpy的解析程序
def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 获取文件长度
    numberOfLines = len(fr.readlines())
    # 根据文件长度组装以0填充的(numberOfLines,3)的矩阵
    returnMat = zeros((numberOfLines, 3))
    # print(returnMat.shape)
    classLabelVector = []
    # 重新读取文件,因为readlines会一次将offset移动到最后一位
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 删除字符串中开头结尾的空白符(包括'\n', '\r',  '\t',  ' ')
        line = line.strip()
        # 分割字符串,获取元素列表
        listFromLine = line.split('\t')
        # 修改指定下标的数组值
        returnMat[index, :] = listFromLine[0:3]
        # 获取listFromLine列表最后一个值,并追加到classLabelVector列表
        # 注意,此处必须明确告诉编译器元素列表中存储的元素是整数类型的,不然编译器会当作字符串处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    # newValue = (oldValue-min)/(max-min)
    # 归一化数值:将取值范围处理为0到1或者1到-1之间
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest(filename):
    hoRatio = 0.10
    # 调用file2matrix和autoNorm函数从文件中读取数据,并将其转换为归一化特征值
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取归一化特征值的个数
    m = normMat.shape[0]
    # 获取测试集数据
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 预测分类
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with :%d,the real answer is :%d"
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is :%f" % (errorCount / float(numTestVecs)))


# 约会网站预测函数
def classifyPerson(filename):
    # 分类器列表
    resultList = ['not at all', 'is small doses', 'in large doses']
    # 输入某人的信息以便获取预测值
    percentTats = float(input("percentage of time spent playing video game?"))
    ffMiles = float(input("frequent flier mils earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix(filename)
    print(datingLabels)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    # 归一化
    auto = (inArr - minVals) / ranges
    classifierResult = classify0(auto, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])


# 将图像转换为向量
def img2vector(filename):
    # 创建1*1024的numpy数组
    returnVect = zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 循环读取文件的前32行
    for i in range(32):
        lineStr = fr.readline().strip()
        # 将每行的头32个字符值存储到numpy数组中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回数组
    return returnVect


# 手写数字识别系统的测试代码
def handwritingClassTest():
    # 创建分类列表
    hwLabels = []
    # 列出给定目录的文件名列表
    trainingFileList = os.listdir('../../data/digits/trainingDigits')
    # 获取文件名列表的长度
    m = len(trainingFileList)
    # 创建(1934, 1024)以零填充的矩阵
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        # 1.获取指定的文件名
        fileNameStr = trainingFileList[i]
        # 2.解析获取文件名前缀,eg:0_0
        fileStr = fileNameStr.split('.')[0]
        # 3.解析获取分类数字,eg:0
        classNumStr = int(fileStr.split('_')[0])
        # 4.将分类数字添加到分类列表中
        hwLabels.append(classNumStr)
        # 将图像转换为向量,并替换矩阵指定位置的值
        trainingMat[i, :] = img2vector('../../data/digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('../../data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../../data/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print('the classifier came back with: %d , the real answer is: %d'
        #       % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print('\n the total number of errors is : %d' % errorCount)
    print('\n the total error rate is %f' % (errorCount / float(mTest)))


if __name__ == "__main__":
    handwritingClassTest()
