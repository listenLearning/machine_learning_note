#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *


class treeNode():
    def __init__(self, feat, val, right, left):
        '''
        创建树节点
        :param feat:
        :param val:
        :param right:
        :param left:
        '''
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0];
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;
    bestIndex = 0;
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue


def loadDataSet(fileName):
    '''
    数据导入函数
    :param fileName:
    :return:
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    在给定特征和特征值的情况下,
    本函数将通过数据过滤的方式将数据集合切分得到两个子集并返回
    :param dataSet: 数据集合
    :param feature: 待切分的特征
    :param value: 该特征的某个特征值
    :return:
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    创建树
    :param dataSet: 数据集
    :param leafType: 建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 包含树构建所需其它参数的元组
    :return:
    '''
    # 调用chooseBestSplit函数,尝试将数据集分成两个部分,
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果满足停止条件,chooseBestSplit函数将返回None和某类模型的值
    # 如果构建的是回归树,该模型是一个常数
    # 如果是模型树,其模型是一个线性方程
    if feat == None: return val
    # 如果不满足停止条件,chooseBestSplit函数将创建一个新的数据字典并将数据集分成两份
    # 在这两份数据集上分别继续递归调用createTree函数
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
