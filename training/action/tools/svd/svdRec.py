#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *
from numpy import linalg as la


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData2_1():
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 5, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def ecludSim(inA, inB):
    '''
    通过欧式距离计算相似度,并归一化
    :param inA:
    :param inB:
    :return:
    '''
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    '''
    计算皮尔逊相关系数
    :param inA:
    :param inB:
    :return:
    '''
    # 检查是否存在3个或更多的点,如果不存在,该函数返回1.0,因为词时两个向量完全相关
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    '''
    计算余弦相似度
    :param inA:
    :param inB:
    :return:
    '''
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMass, item):
    '''
    计算在给定相似度计算方法的条件下,用户对物品的估计评分值
    :param dataMat:数据矩阵
    :param user:用户编号
    :param simMass:相似度计算方法
    :param item:物品编号
    :return:
    '''
    # 提取数据集中的物品数目
    # 此处行对应用户,列对应物品,
    m, n = shape(dataMat)
    # 初始化用于计算估计评分值的变量
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历数据中的每一个物品
    # 本循环大体上是对用户评分过的物品进行遍历,并将它和其它物品进行比较
    for j in range(n):
        # 提取出用户评分值
        userRating = dataMat[user, j]
        # 如果当前评分值为0，就意味着用户没有对该物品进行评分,直接跳过该物品
        if userRating == 0: continue
        # 计算两个物品中已经被评分的元素
        # logical_and 计算x1和x2元素的真值,注意:如果是两个参数见真值发生冲突,函数以第一个参数的真值为准值
        # overLap给出的是两个物品当中已经被评分的那个元素的索引ID
        overLap = nonzero(logical_and(dataMat[:, item].A > 0,
                                      dataMat[:, j].A > 0))[0]
        # 检测如果连着没有任何重复元素,则相似度为0且终止本次循环
        if len(overLap) == 0:
            similarity = 0
        else:
            # 如果存在重合的物品,则基于这些重合物品计算相似度
            similarity = simMass(dataMat[overLap, item],
                                 dataMat[overLap, j])
        print('the %d and %d similarity is :%f' % (item, j, similarity))
        # 相似度不断累加
        simTotal += similarity
        # 每次计算还需要考虑相似度和当前用户评分的乘积
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 通过除以所有的评分总和,对上述相似度评分的乘积进行归一化
        # 这就可以使得最后的评分值在0到5之间,而这些评分制则用于对预测值进行排序
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMass=cosSim, estMethod=standEst):
    '''
    获取最优的N个推荐结果
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param N: N值
    :param simMass: 相似度计算方法
    :param estMethod: 估计方法
    :return:
    '''
    # 对给定的用户建立一个未评分的物品列表
    unrateItems = nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品,那么就退出函数,否则在所有的未评分物品上进行循环
    if len(unrateItems) == 0: return 'you rated everything'
    itemScoress = []
    for item in unrateItems:
        # 对每个未评分的物品,通过调用estMethod来产生该物品的预测得分
        estimatedScore = estMethod(dataMat, user, simMass, item)
        # 将该物品的编号和估计得分值放在元素列表itemScoress中
        itemScoress.append((item, estimatedScore))
    # 按照估计得分,对该列表进行排序并返回.该列表是从大到小逆序排列的,因此其第一个值就是最大值
    return sorted(itemScoress, key=lambda jj: jj[1], reverse=True)[:N]


def svdEst(dataMat, user, simMass, item):
    '''
    对给定用户给定物品构建一个评分估计值
    :param dataMat:
    :param user:
    :param simMass:
    :param item:
    :return:
    '''
    # 提取数据集中的物品数目
    n = shape(dataMat)[1]
    simTotla = 0.0
    ratSimTotal = 0.0
    # 对数进行SVD分解
    U, Sigma, VT = la.svd(dataMat)
    # SVD分解之后,我们只利用包含了90%能量值的奇异值,这些奇异值会以Numpy数组的形式得以保存
    # 如果要及进行矩阵运算,那么就必须要用这些奇异值构建出一个对角矩阵
    # eye()生成对角矩阵,对角线值为1
    Sigma2 = mat(eye(4) * Sigma[:4])
    # 利用U矩阵将物品转换到低维空间中
    # Sigma2.I对Sigma2进行归一化
    xformedItems = dataMat.T * U[:, :4] * Sigma2.I
    # 对于给定用户,for循环在用户对行的所有元素上进行遍历
    for j in range(n):
        # 提取出用户评分值
        userRating = dataMat[user, j]
        # 如果当前评分值为0，就意味着用户没有对该物品进行评分,直接跳过该物品
        if userRating == 0 or j == item: continue
        # 调用相似度计算方法计算出相似度
        similarity = simMeas(xformedItems[item, :].T,
                             xformedItems[j, :].T)
        print('the %d and %d similarity is :%f' % (item, j, similarity))
        # 相似度不断累加
        simTotal += similarity
        # 每次计算还需要考虑相似度和当前用户评分的乘积
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 通过除以所有的评分总和,对上述相似度评分的乘积进行归一化
        # 这就可以使得最后的评分值在0到5之间,而这些评分制则用于对预测值进行排序
        return ratSimTotal / simTotal


def printMat(inMat, thresh=0.8):
    '''
    打印矩阵
    :param inMat:
    :param thresh:
    :return:
    '''
    # 遍历所有的矩阵元素,当元素大于阈值时打印1,否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,end=' ')
            else:
                print(0,end=' ')
        print(' ')


def analysisSigma(sigma, loop):
    '''
    分析当前Sigma值,获取Sigma平方和大于90%的维度数
    :param sigma:
    :param loop:
    :return:
    '''
    Sigma2 = sigma ** 2
    sumSig = sum(Sigma2) * 0.9
    ite = 0
    for i in range(loop):
        if sumSig <= sum(Sigma2[:i]):
            print('sumSig', '-->', sumSig, '   ', 'sum(Sigma2[:%d])' % i, '-->', sum(Sigma2[:i]))
            ite = i
            break
    return ite


def imgCompress(fileName, numSV=3, thresh=0.8):
    '''
    实现图像的压缩,允许基于任意给定的奇异值数目来重构图像
    :param fileName:
    :param numSV:
    :param thresh:
    :return:
    '''
    # 构建一个空列表
    my1 = []
    # 打开文本文件，并从文件中以字符方式读入字符
    for line in open(fileName).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow)
    myMat = mat(my1)
    n = shape(myMat)[1]
    print("****original matrix****")
    printMat(myMat, thresh)
    # 对原始图像进行SVD分解并重构图像
    # 通过Sigma 重新构成SigRecom来实现
    U, Sigma, VT = la.svd(myMat)
    # 调用analysisSigma函数,获取Sigma平方和大于90%的维度数
    numSV = analysisSigma(Sigma, n)
    # SVD分解之后,我们只利用包含了90%能量值的奇异值,这些奇异值会以Numpy数组的形式得以保存
    # 如果要及进行矩阵运算,那么就必须要用这些奇异值构建出一个对角矩阵
    # eye()生成对角矩阵,对角线值为1
    SigRecon = mat(eye(numSV) * Sigma[:numSV])
    # SigRecon = mat(zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k, k] = Sigma[k]
    # 利用U矩阵将物品转换到低维空间中
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("reconstructed matrix using %d singular values" % numSV)
    printMat(reconMat, thresh)
