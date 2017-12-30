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
    '''
    生成叶节点,当chooseBestSplit函数确定不再对数据进行切分时,
    调用本函数来得到叶节点的模型,在回归树中,该模型就是目标变量的均值
    :param dataSet:
    :return:
    '''
    return mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    在给定的数据上计算目标变量的平方误差
    :param dataSet:
    :return:
    '''
    # 此处返回的时总方差,所以要用均方差乘以数据集中样本的个数
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    用最佳方式切分数据集和生成相应的叶节点,本函数是回归树构建的核心函数
    :param dataSet: 数据集
    :param leafType: 对创建叶节点的函数的引用
    :param errType: 对总方差计算函数的引用
    :param ops: 用户定义的参数构成的元组
    :return:
    '''
    # 为ops设置tolS和tolN两个值,用于控制函数的停止时机
    # tolS时容许的误差下降值,tolN时切分的最少样本数
    tolS = ops[0];
    tolN = ops[1]
    # 对当前所有母表变量建立一个集合,函数会统计不同剩余特征的数目
    # 如果数目为1,就不需要再切分而直接返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 计算当前数据集的大小和误差
    m, n = shape(dataSet)
    # 误差将用于与心切分误差进行对比,来检查新切分能否降低误差
    S = errType(dataSet)
    bestS = inf;
    bestIndex = 0;
    bestValue = 0
    # 在所有可能的特征及其可能值上遍历,找到最佳的切分方式
    # 最佳切分也就是使得切分后能达到最低误差的切分
    for featIndex in range(n - 1):
        setDat = set(dataSet[:, featIndex].T.tolist()[0])
        # 下面的一行表示的是将某一列全部的数据转换为行，然后设置为list形式
        for splitVal in setDat:
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分数据集后效果提升不够大,那么就不应进行切分操作而直接创建叶节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 比较两个切分后的子集大小,如果某个子集的大小小于用户定义的参数tolN,那么也不应切分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 如果这些提前终止条件都不满足,那么就返回切分特征和特征值
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
        fltLine = [float(x) for x in curLine]
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
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
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
