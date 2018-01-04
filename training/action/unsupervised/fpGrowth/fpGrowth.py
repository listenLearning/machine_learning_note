#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

import operator


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        # 用于存放节点名字
        self.name = nameValue
        # 计数值
        self.count = numOccur
        # 用于连接相似的元素项
        self.nodeLink = None
        # 指向当前节点的父变量,通常情况下是从上往下迭代访问节点,
        # 因此不需要这个变量,但当需要根据给定叶子节点上溯整棵树,就需要指向父节点的指针
        self.parent = parentNode
        # 存放节点的子节点
        self.children = {}

    def inc(self, numOccur):
        '''
        对count变量增加给定值
        :param numOccur:
        :return:
        '''
        self.count += numOccur

    def disp(self, ind=1):
        '''
        将树以文本形式显示
        :param ind:
        :return:
        '''
        print(' ' * ind, self.name, '   ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    '''

    :param dataSet: 数据集
    :param minSup: 最小支持度
    :return:
    '''
    # 初始化一个头指针表
    headerTable = {}
    # 遍历数据集中的所有数据，将每个元素出现的次数记录到头指针表中
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 遍历头指针表,清除小于最小支持度的数据
    # 字典在遍历时不能进行修改,所以此处转换成List
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    # 将字典放入集合中,避免字典中有重复的数据
    freqItemSet = set(headerTable.keys())
    # 如果所有项都不频繁,那么就不需要下一步操作
    if len(freqItemSet) == 0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderItems = [v[0] for v in sorted(localD.items(),
                                               key=operator.itemgetter(1), reverse=True)]
            updateTree(orderItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
