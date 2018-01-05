#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

import operator
import json
from time import sleep
import twitter
import re


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
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    '''
    创建FP树以及头指针表
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
    # 扩展headerTable,将headerTable格式化为dict{key:[count,None]}
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 初始化父节点,即创建一个空集,用于装载频繁项集
    retTree = treeNode('Null Set', 1, None)
    # 遍历数据集中的所有数据,取得所有元素及其对应的出现次数
    for tranSet, count in dataSet.items():
        # 初始化一个空字典
        localD = {}
        # 循环遍历所有的元素项
        for item in tranSet:
            # 如果当前元素出现在集合freqItemSet中,就将当前元素及其出现的次数加载到localD
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # 如果字典localD的长度大于0,即当前字典不是一个空字典
        if len(localD) > 0:
            # 1.将localD中的元素进行排序
            # 2.将排序后的localD循环遍历并取出其中的元素项集放置到列表orderItems中
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # 填充数,通过有序的orderItems的第一位,进行顺序填充第一层的子节点
            updateTree(orderedItems, retTree, headerTable, count)
    # 返回树以及头指针表
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    '''
    更新FP-tree
    :param items: 排序后的列表
    :param inTree: fp树
    :param headerTable: 头指针表
    :param count: 元素出现的次数
    :return:
    '''
    # 测试事务中的第一个元素是否作为子节点存在,
    if items[0] in inTree.children:
        # 如果存在的话,则更新该元素项的计数
        inTree.children[items[0]].inc(count)
    else:
        # 如果不存在,则创建一个treeNode并将其作为一个子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 检查当前头指针表的子节点是否为None
        if headerTable[items[0]][1] == None:
            # 如果是,就将上面的元素设置为当前头指针表节点的子节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 如果不是,就调用updateHeader,将头指针表更新以指向新的节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 迭代调用本函数,每次调用时都会去掉列表中的第一个元素,直到当前没有元素为止
        # items[1::]通过切片取出除头元素的所有元素
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    '''
    更新头指针,建立相同元素间的关系
    从头指针的nodeLink开始,一直沿着nodeLink直到到达链表末尾,
    如果链表很长,可能会遇到迭代调用的次数限制
    :param nodeToTest: 满足最小支持度的字典
    :param targetNode: Tree对象的子节点
    :return:
    '''
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


def ascendTree(leafNode, prefixPath):
    '''
    上溯FP-tree
    :param leafNode:
    :param prefixPath:
    :return:
    '''
    # 检查当前节点的父节点是否为空
    if leafNode.parent != None:
        # 如果不为空,就将当前节点的名字添加到prefixPath中
        prefixPath.append(leafNode.name)
        # 递归调用ascendTree函数直到当前节点没有上一级父节点
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    '''
    发现以给定元素项结尾的所有路径的函数
    :param basePat:
    :param treeNode:
    :return:
    '''
    # 创建一个空字典,用于放置条件模式基
    condPats = {}
    # 遍历链表,直到达到链尾
    while treeNode != None:
        # 创建一个空列表,用于放置前缀路径
        prefixPath = []
        # 调用ascendTree函数上溯fp树
        ascendTree(treeNode, prefixPath)
        # 如果前缀路径长度大于1
        # 此处主要是为了避免前缀路径中只有单独一个元素,添加了空节点
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''
    递归查找频繁项集
    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    '''
    # 对头指针表中的元素按照其出现频率进行排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    # 循环遍历最频繁项集的key,从小到大的递归寻找对应的频繁项集
    for basePat in bigL:
        # preFix为newFreqSet上一次的存储记录,一旦没有myHead,就不会更新
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # 将每一个频繁项添加到频繁项集列表freqItemList中
        freqItemList.append(newFreqSet)
        # 调用findPrefixPath函数来创建条件基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 将条件基传输给createTree函数,创建新的fp树以及头指针列表
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for:', newFreqSet)
            myCondTree.disp(1)
            # 递归调用mineTree函数,直到fp树中没有元素项
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# 无法连接twitter,所以不做运行
# def getLotsOfTweets(searchStr):
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY,
#                       consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     resultsPages = []
#     for i in range(1, 15):
#         print('fetching page %d' % i)
#         searchResult = api.GetSearch(searchStr, per_page=100, page=i)
#         resultsPages.append(searchResult)
#         sleep(6)
#     return resultsPages
#
#
# def textParse(bigString):
#     urlsRemoved = re.sub('(https[s]?:[/][/]|www.)'
#                          '([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
#     listOfTokens = re.split(r'\W*', urlsRemoved)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
#
# def mineTweets(tweetArr, minSup=5):
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text))
#     initStr = createInitSet(parsedList)
#     myFPtree, myHeaderTable = createTree(initStr, minSup)
#     myFreqList = []
#     mineTree(myFPtree, myHeaderTable, minSup, set([]), myFreqList)
#     return myFreqList
