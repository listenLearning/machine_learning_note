#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *


def loadDataSet():
    '''
    创建了一个用于测试的简单数据集
    :return:
    '''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    '''
    构建集合C1,C1是大小为1的所有候选项集的集合
    Apriori算法首先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求
    满足最低要求的项集构成集合L1.而L1中的元素相互组合构成C2,C2再进一步过滤变为L2
    :param dataSet:
    :return:
    '''
    # 初始化一个空列表C1
    C1 = []
    # 遍历数据集中的所有数据,由loadDataSet可知,遍历后的数据为[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]
    for transaction in dataSet:
        # 遍历所有数据集中的数据,获取样本值
        for item in transaction:
            # 判断样本数据是否再C1中存在,如果不存在,就添加到C1
            if not [item] in C1:
                C1.append([item])
    # 排序
    C1.sort()
    # 遍历C1中的数据并放置到不可变集合frozenset中
    # 这里使用frozenset而不是set是因为之后必须要将这些集合作为字典值使用,frozenset可以实现这一点,而set却做不到
    return [frozenset(x) for x in C1]


def scanD(D, Ck, minSupport):
    '''
    本函数用于从C1生成L1,函数会返回一个包含支持度值的字典以备后用
    :param D: 数据集
    :param Ck: 候选项集列表
    :param minSupport: 感兴趣项集的最小支持度
    :return:
    '''
    # 初始化一个空字典ssCnt
    ssCnt = {}
    # 遍历数据集中的所有交易记录以及C1中的所有候选项集
    for tid in D:
        for can in Ck:
            # 如果c1中的集合是记录的一部分,那么增加字典中对应的计数值
            if can.issubset(tid):
                # 如果C1中的集合不存在于字典ssCnt,那么初始化到字典ssCnt,使其等于1
                # 如果存在,计数值+1
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 数据集D的数量,用于之后计算所有项集的支持度
    numItems = float(len(D))
    # 初始化一个包含满足最小支持度要求的集合
    retList = []
    supportData = {}
    # 遍历字典中的每个元素
    for key in ssCnt:
        # 计算所有项集的支持度
        support = ssCnt[key] / numItems
        # 如果支持度满足最小支持度要求,则将字典元素添加到retList
        if support >= minSupport:
            retList.insert(0, key)
        # 存储所有的候选项(key)和对应的支持度(support)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = [set(x) for x in dataSet]
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
