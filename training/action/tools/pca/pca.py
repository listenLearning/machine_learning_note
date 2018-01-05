#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *
from matplotlib import pyplot as plt


def loadDataSet(fileName, delim='\t'):
    '''
    加载文本文件,解析成矩阵
    :param fileName:
    :param delim:
    :return:
    '''
    # 读取文本文件
    fr = open(fileName)
    # 按delim分隔符分割每一行文件
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # 遍历字符列表,将字符列表的中美元素转换成浮点类型
    datArr = [[float(x) for x in line] for line in stringArr]
    # 返回矩阵
    return mat(datArr)


def pca(datMat, topNfeat=9999999):
    '''
    PCA算法
    pca就是说在某一方向的投影长度最长的k个特征就筛选它来代表整个数据集
    :param datMat: 数据集
    :param topNfeat: 应用的N个特征
    :return:
    '''
    # 沿着纵轴计算矩阵的平均值
    meanVals = mean(datMat, axis=0)
    # 减去原始数据集中的平均值
    meanRemoved = datMat - meanVals
    # 计算meanRemoved的协方差矩阵X^T*X,rowvar表示一行代表一个样本
    # N维的数据集需要计算n!/((n-2)!*2)个协方差
    # 特征数m,样本数n, 特征间两两的相关系数求出来就是n*n,即(n,n)
    covMat = cov(meanRemoved, rowvar=0)
    # print('shape:\n', shape(meanRemoved), '\n')
    # print('covMat:\n', covMat, '\n')
    # 计算covMat特征值和特征向量
    # 参照公式Av=λv,λ是A的特征值,非零向量v称为A对应于特征值λ的特征向量
    # 特征向量是某一方向   特征值是在这一方向的长度
    eigVals, eigVects = linalg.eig(mat(covMat))
    # 调用argsort函数对特征值进行从小到大排序
    # 调用argsort,返回数组从从小到大的索引值
    eigValInd = argsort(eigVals)
    # # 根据特征值排序结果的逆序就可以得到topNfeat个最大的特征向量
    # [::-1]表示翻转,即逆序,[:-(topNfeat + 1):-1]表示从索引为0取到-(topNfeat + 1)的值,步长为-1,即反转
    # -(topNfeat + 1)此处为负数,即从末尾开始,去到当前的数,末尾从-1开始计算
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 上面的特征向量将构成对数据进行转换的矩阵,
    # meanRemoved利用N个特征将原始数据转换到新空间中
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 原始数据被重构后返回用于调试,同时降维之后的数据集也被返回了
    return lowDDataMat, reconMat


def replaceNanWithMean(fileName):
    dataMat = loadDataSet(fileName, ' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat


def drawingWithMatplot(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0])
    ax.scatter(reconMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
