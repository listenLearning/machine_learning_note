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
    '''
    计算候选项集ck
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return:
    '''
    # 初始化一个空列表
    retList = []
    # 计算Lk中的元素数目
    lenLk = len(Lk)
    # 循环遍历lk中的所有元素
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 获取索引下标为i的集合的前k-2个元素
            L1 = list(Lk[i])[: k - 2]
            # print('i%d:k:%d,k-2:%d '%(i,k,k-2),' Lk: ',Lk,' Lk[i]: ',Lk[i],' L1: ',L1)
            # 获取索引下标为i的集合的前k-2个元素
            L2 = list(Lk[j])[: k - 2]
            # print('j%d:k:%d,k-2:%d ' % (j, k, k - 2), ' Lk: ', Lk, ' Lk[j]: ', Lk[j], ' L2: ', L2)
            # 将L1L2排序,方便比较两个列表是否相等
            L1.sort()
            L2.sort()
            # 如果L1与L2的前K-2个元素都相等
            if L1 == L2:
                # 将这两个集合合并成一个大小为k的集合
                # 集合的合并操作在python中对应操作符|
                retList.append(Lk[i] | Lk[j])
    # 返回元素两两合并的数据集
    return retList


def apriori(dataSet, minSupport=0.5):
    '''
    生成候选项集的列表
    :param dataSet: 数据集
    :param minSupport: 支持度
    :return:
    '''
    # 调用createC1创建C1
    C1 = createC1(dataSet)
    # 读入数据集将其转化为D(集合列表)
    D = [set(x) for x in dataSet]
    # 调用scanD函数来创建一个包含满足最小支持度要求的集合L1,
    # 并将L1放入到列表L中,L会包含L1,L2,L3...
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    # 通过while循环创建包含更大项集的更大列表,直到下一个大的项集为空
    while (len(L[k - 2]) > 0):
        # 调用aprioriGen计算候选项集Ck
        Ck = aprioriGen(L[k - 2], k)
        # 使用scanD函数基于Ck来创建LK,Ck是一个候选项集列表
        # scanD会遍历Ck,丢掉不满足最小支持度要求的项集
        Lk, supK = scanD(D, Ck, minSupport)
        # 更新候选项和对应的支持度
        supportData.update(supK)
        # 将Lk列表添加到L
        L.append(Lk)
        # 增加k的值,重复遍历
        k += 1
    # 当LK为空时,程序返回L并退出
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    '''
    生成一个包含可信度的规则列表,后面可以基于可信度对他们进行排序
    :param L: 频繁项集列表
    :param supportData: 包含频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return:
    '''
    # 初始化一个空的规则列表
    bigRuleList = []
    # 遍历L中的每一个频繁项集
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 对每一个频繁项集创建只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                # 如果当前项集的元素数目超过2,调用rulesFromConSeq函数对它做进一步合并
                rulesFromConSeq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 如果当前项集的元素数目只有2,调用calcConf来计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    # 返回规则列表
    return bigRuleList


def calcConf(freqSet, H, supportData, bigRuleList, minConf=0.7):
    '''
    计算规则的可信度以及找到满足最小可信度要求的规则
    :param freqSet:
    :param H:
    :param supportData:
    :param bigRuleList:
    :param minConf:
    :return:
    '''
    # 初始化一个空列表prunedH
    prunedH = []
    # 遍历列表H中的所有项集
    for conseq in H:
        # 计算可信度值,依照规则support(P|H)/support(P)
        # supportData[freqSet - conseq],因为supportData中的key是frozenset,所以相减求差值
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        # 判断可信度值是否满足最小可信度值
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 对列表bigRuleList进行填充,bigRuleList是前面通过检查的bigRuleList
            bigRuleList.append((freqSet - conseq, conseq, conf))
            # 将通过检查的规则填充到规则列表中
            prunedH.append(conseq)
    # 返回满足最小可信度要求的规则列表
    return prunedH


def rulesFromConSeq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    '''
    生成候选规则集合
    :param freqSet:
    :param H:
    :param supportData:
    :param bigRuleList:
    :param minConf:
    :return:
    '''
    # 计算H中的频繁集大小m
    m = len(H[0])
    # 查看该频繁项集是否大到可以移除大小为M的子集,如果可以,则将其移除
    if (len(freqSet) > (m + 1)):
        # 调用aprioriGen来生成H中的元素的无重复组合,并将结果存储在Hmp1
        # Hmp1也是下次迭代的H列表,Hmp1中包含所有可能的规则
        Hmp1 = aprioriGen(H, m + 1)
        # 调用calcConf函数来测试它们的可信度已确定规则是否满足要求
        Hmp1 = calcConf(freqSet, Hmp1, supportData, bigRuleList, minConf)
        # 如果不止一条规则满足要求,那么使用Hmp1迭代调用函数rulesFromConSeq来判断是否可以进一步组合这些规则
        if (len(Hmp1) > 1):
            rulesFromConSeq(freqSet, Hmp1, supportData, bigRuleList, minConf)
