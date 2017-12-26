#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

import numpy as np
import math
import operator
from matplotlib import pyplot as plt
import pickle
from os import listdir


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 为所有可能分类创建字典
    # 1.创建一个数据字典,
    labelCounts = {}
    for featVec in dataSet:
        # 2.键值是最有一列的数值
        currentLabel = featVec[-1]
        # 3.如果当前键值不存在,则扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 记录了当前了类别出现的次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底求对数
    for key in labelCounts:
        # 使用所有类别标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 此处采用python自带的math模块的log函数而不是numpy的log函数，
        # 是因为numpy的log函数第二个参数不是base而是out array
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    # 为了不修改原始数据集,创建一个新的列表对象
    retDataSet = []
    # 遍历数据集中的每个元素,发现符合要求的值,就将其添加到这个新创建的列表中
    for featVec in dataSet:
        # 将符合特诊的数据抽取出来
        if featVec[axis] == value:
            # 按照某个特征划分数据集时,需要将所有符合要求的元素抽取出来
            reduceFeatVec = featVec[:axis]
            # extend: 包含两个列表所有元素的新列表
            reduceFeatVec.extend(featVec[axis + 1:])
            # append：原始列表得到一个新的元素,新元素也是一个列表
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    '''
    在函数中调用的数据需要满足一定的要求:
        1.数据必须是一种由列表元素组成的列表,而且所有的列表元素都要具有相同的数据长度
        2.数据的最后一列或者每个示例的最后一个元素是当前示例的类别标签
    :param dataSet:
    :return:
    '''
    # 洞顶当前数据集包含多少特征属性
    numFeatures = len(dataSet[0]) - 1
    # 计算整个数据集的原始香农熵,用于与划分完之后的数据集计算的熵值进行比较
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历数据集中的所有特征
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        # 1.使用列表推导来创建新的列表,将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中
        featList = [example[i] for example in dataSet]
        # 2.使用set函数清除重复的元素
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        # 1.遍历当前特种中的所有唯一属性值
        for value in uniqueVals:
            # 2.对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 3.计算数据集的新熵值,并对所有唯一特征值得到的熵求和
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益
        # 比较所有特征中的信息增益,返回最好特征划分的索引值
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    # 创建键值为classList中唯一值的数据字典
    classCount = {}
    for vote in classList:
        # 存储classList中每个类标签出现的频率
        # 初始化当前分类出现的次数
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # 利用operator操作键值排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的分类名称
    return sortedClassCount


def createTree(dataSet, labels):
    """
    创建树的函数代码
    :param dataSet: 数据集
    :param labels: 标签列表(包含了数据集中所有特征的标签)
    :return:
    """
    # 创建了包含数据集的所有类标签的列表变量
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    # 所有类标签完全相同,则直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    # 使用完所有特征,仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最好的数据集划分方式，得到最佳方式索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 根据最佳方式索引得到相关分类标签
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 得到列表包含的所有属性值
    # labels列表是可变对象,在python函数中作为参数时是穿址引用,能够被全局修改
    del (labels[bestFeat])
    # 得到最优列,使用该列表的分支做分类
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        # 复制得到剩余的类标签,并将其存储在新列表变量subLables中
        subLabels = labels[:]
        # 在每个数据集划分上递归调用函数createTree,得到的返回值将被插入到字典变量myTree中
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    # 函数终止,返回myTree
    return myTree


def classify(inputTree, featLables, testVec):
    '''
    使用决策树的分类函数
    :param inputTree: 决策树模型
    :param featLables: feature标签对应的名称
    :param testVec: 测试输入的数据
    :return classLabel: 分类的结果值,需要映射label才能知道名称
    '''
    # 获取输入分类树的顶点key值
    firstStr = inputTree.keys()[0]
    # 通过key值得到下级节点的相关内容
    secondDict = inputTree[firstStr]
    # 根据根节点的key值获取根节点在分类标签中的索引值
    featIndex = featLables.index(firstStr)
    # 循环遍历下级节点，获得整棵树的所有节点的key值
    for key in secondDict.keys():
        # 判断测试数据的顶点key值是否与当前key相等
        if testVec[featIndex] == key:
            # 结束标识：判断下级节点的类型是否为dict
            if type(secondDict[key]).__name__ == 'dict':
                # 递归调用classify函数,直到下级节点不是dict为止，返回分类的结果值，结束递归
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


if __name__ == "__main__":
    # 信息的定义: l(x_i)=-\log_2p(x_i)
    # p(x_i)：选择该分类的概率

    # 信息期望值：
    # H=-\sum_{i=1}^np(x_i)\log_2p(x_i)
    # n：分类的数目
    # 熵：指的是体系的混乱程度,在不同的学科中也有引申出的更为具体的定义,是各领域十分重要的参量
    # 信息熵(香农熵):是一种信息的度量方式,表示信息的混乱程度,也就是说:信息越有序,信息熵越低
    # 信息增益:在划分数据集前后信息发生的变化称为信息增益,
    # 即划分数据集之后得到的子数据集的信息熵和之前没划分数据集之前的数据集的信息熵的差值
    # dataSet, labels = createDataSet()
    # # print(calcShannonEnt(dataSet))
    # print(splitDataSet(dataSet, 0, 0))
    fr = open('../../data/decisionTree/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
