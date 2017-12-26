#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
import re
import feedparser
import operator


def loadDataSet():
    '''
    创建一些实验样本
    :return:
    '''
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 0 正常言论 1 侮辱性文字
    # postingList:进行词条切分后的文档集合
    # classVec:类别标签的集合
    return postingList, classVec


def createVocabList(dataSet):
    '''
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet:
    :return:
    '''
    # 创建一个空集合
    vocabSet = set([])
    for document in dataSet:
        # 将每篇文档返回的新词集合添加到该集合中
        # 操作符|用于求两个集合的并集,这也是一个按位或(or)操作符
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    检查输入的文档在当前词汇表中是否出现
    :param vocabList: 词汇表
    :param inputSet: 文档
    :return returnVec: 文档向量,向量的每一元素为1或0,分别表示词汇表中的单词在输入文档中是否出现
    '''
    # 创建一个和词汇表等长度的元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历文档中的所有单词
    for word in inputSet:
        # 如果出现了词汇表的单词,则将输出的文档向量中的对应值设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:", word, "  is not in my Vocabulary!")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 文档矩阵
    :param trainCategory: 文档类别标签所构成的向量
    :return:
    '''
    # 获取文档总数
    numTrainDocs = len(trainMatrix)
    # 获取文档向量列表总数
    numWords = len(trainMatrix[0])
    # 求有侮辱性单词的文档占所有文档个数的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化程序中的分子变量和分母变量
    # 此时的分子变量是一个元素个数等于词汇大小的Numpy数组
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历出训练集中的所有文档,一旦某个词语在某一文档中出现,
    # 则该词对应的个数(p1Num或者p0Num)加1,该文档的总词数也相应加1
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 每个元素除以该类别中的总词数
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p1Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(Vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    朴素贝叶斯分类函数
    :param Vec2Classify: 分类向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return: 分类标签
    '''
    # 1.使用numpy数组计算两个向量对应元素相乘的结果
    # 2.将词汇表中所有词的对应值相加
    # 3.将该值加到类别的对数概率上
    # 为了防止多个小数相乘四舍五入变成0，所以在此优化取pClass1对数
    p1 = sum(Vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(Vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 比较类别的概率返回大概率对应的类别标签
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    '''
    朴素贝叶斯词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word:%s is not in my vocabulary" % word)
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    '''
    文件解析
    :param bigString:
    :return:
    '''
    # 接收一个大字符串并将其解析为字符串列表
    listOfTokens = re.split(r'\W*', bigString)
    # 去掉少于两个字符的字符串,并将所有字符串转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    对贝叶斯垃圾邮件分类器进行自动化处理
    :return:
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 读取spam与ham文件夹下的文本文件，并将其解析为词列表
        wordList = textParse(open('../../data/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../../data/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 构建测试集与训练集,两个集合中的邮件随机抽取
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))


def calcMostFreq(vocabList, fullText):
    '''
    该函数遍历词汇表中的每个词并统计他在文本中出现的次数,
    然后根据出现次数从高到低对词典及排序,并返回排序最高的30个单词
    :param vocabList:
    :param fullText:
    :return:
    '''
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    '''
    使用了两个RSS源作为参数
    RSS源需要在函数外导入,是因为RSS源会随着时间改变,
    如果想通过代码来比较程序执行的差异,就应该使用相同的输入,
    :param feed1:
    :param feed0:
    :return:
    '''
    docList = []
    classList = []
    fullText = []
    # 找出RSS源中长度最小的源,防止遍历的时候数组下标越界
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        # 将RSS源解析成为字符串列表
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 去除列表中的重复词
    vocabList = createVocabList(docList)
    # 获取词组列表中出现次数最多的前30个词，并移除掉这些词
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    # 获取训练数据和测试数据
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    # 训练集和测试集转化为向量的形式
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:
        # 调用朴素贝叶斯词袋模型,将字符串转化为向量并一次追加到训练集列表
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 调用贝叶斯分类函数,判断分类后的值与分类标签中的值是否相等
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    '''
    使用RSS源作为输入,然后训练并测试朴素贝叶斯分类器,返回使用的概率
    :param ny:
    :param sf:
    :return:
    '''
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -0.6:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -0.6:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF'.center(50, '*'))
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY'.center(50, '#'))
    for item in sortedNY:
        print(item[0])
