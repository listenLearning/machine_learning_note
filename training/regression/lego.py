#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from time import sleep
import json
import urllib3
from numpy import *
from training.regression import ridgeRegression as rr
from training.regression import regression as re


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    '''
    调用Google购物API并保证数据抽取的正确性
    :param retX:
    :param retY:
    :param setNum:
    :param yr:
    :param numPce:
    :param origPrc:
    :return:
    '''
    # 防止短时间内有过多的API调用,休眠10秒
    sleep(10)
    http = urllib3.PoolManager()
    # 拼接查询的URL字符串,添加API的key和待查询的套装信息
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = http.urlopen(method='GET', url=searchURL)
    # 调用json.loads()方法实现打开和解析操作,获取字典
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                # 过滤掉不完整的套装
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    '''
    交叉验证测试岭回归
    :param xArr:xarr与yarr存有数据集中的x和y值的list对象,默认具有相同的长度
    :param yArr:
    :param numVal:算法中交叉验证的次数,如果没有指定,则默认是10
    :return:
    '''
    # 计算数据点的个数
    m = len(yArr)
    # 生成一个list
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        # 创建训练集和测试集的容器
        trainX = []
        trainY = []
        testX = []
        testY = []
        # 调用random.shuffle()函数对其中的元素进行混洗
        # 实现训练集或测试集数据点的随机选取
        random.shuffle(indexList)
        for j in range(m):
            # 将数据集分割成训练集与测试集,并将二者放入对应的容器中
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 创建一个心的矩阵wMat来保存岭回归中的所有回归系数
        # ridgeTest函数使用30个不同的λ值创建了30组不同的回归系数
        wMat = rr.ridgeTest(trainX, trainY)
        # 使用30组回归系数,来循环测试回归效果
        for k in range(30):
            # 数据标准化
            # 岭回归需要使用标准化后的数据,
            # 因此测试数据也需要使用测试集相同的参数来执行标准化
            matTextX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            # 调用rssError计算误差,并将结果保存在errorMat
            # 在完成所有交叉验证后,errorMat保存了ridgeTest里每个λ对应的多个误差值
            errorMat[i, k] = re.rssError((yEst.T.A, array(testY)))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minmean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = mean(xMat, 0)
    # 对数据还原并将最终结果展示
    # 由于岭回归使用了数据标准化,而standRegres则没有,因此需要将数据还原处理
    unReg = bestWeights / varX
    print('the best model from Ridge Regression is:\n', unReg)
    print('with constant term: ', -1 * sum(multiply(meanX, unReg)) + mean(yMat))
