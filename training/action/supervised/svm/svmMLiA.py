#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *


def loadDataSet(filename):
    '''
    打开文件并对其进行逐行解析,从而得到每行的类标签和整个数据矩阵
    :param filename:
    :return:
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append((float(lineArr[0]), float(lineArr[1])))
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''
    在某个区间范围内随机选择一个整数
    :param i: alpha下标
    :param m: 所有alpha的数目,只要函数值不等于与输入值i,函数就会进行随机选择
    :return:
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''
    用于调整大于H或者小于L的alpha值
    :param aj:
    :param H:
    :param L:
    :return:

    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版smo算法
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大循环次数,当没有任何alpha发生改变时会将整个集合的一次遍历过程集成一次迭代
    :return:
    '''
    # 将输入参数转换成numpy矩阵,可以简化很多数学处理操作
    # 由于转职了类别标签,因此得到就是一个列向量而不是列表
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    b = 0
    # 获取样本数m和特征数n
    m, n = shape(dataMatrix)
    # 构建alpha列矩阵,矩阵中的元素都初始化为0
    alphas = mat(zeros((m, 1)))
    # 创建iter变量,存储在没有任何alpha改变的情况下遍历数据集的次数
    # 当该变量达到输入值maxIter时,函数结束运行并退出
    iter = 0
    while (iter < maxIter):
        # 初始化alphaPairsChanged,该变量用于记录alpha是否已进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # 计算预测类别 y = w^Tx_i+b 并且 w= \sum_{1-n} a_n\times label_n \times x_n
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 预测结果与真是结果对比,计算误差
            Ei = fXi - float(labelMat[i])
            # 不管是正间隔还是副间隔都会被测试,同时也要检查alpha值,以保证其不能等于0或C
            # 由于后面alpha值小于0或大于C时将被调整为0或C,所以一旦他们等于这两个值的话,就表示他们已经在‘边界’上了,不能再进行优化(减小或增大)了
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 调用selectJrand函数随机选择第二个alpha值
                j = selectJrand(i, m)
                # 计算预测类别与误差
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算L,H,用于将alpha[j]调整到0到C之间,如果l和h相等,就不做任何改变,直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # eta是alpha[j]的最优修改量,如果eta为0,即需要退出for循环的当前迭代过程
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                      - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 计算出一个新的alpha[j],并调用clipAlpha函数调整大于H或者小于l的alpha值
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否有轻微改变,如果是,退出for循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 然后alpha[i]和alpha[j]同样进行改变,虽然改变的大小一样,但是改变的方向正好相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter :%d, i:%d, pairs changed %d ' % (iter, i, alphaPairsChanged))
        # 检查alphaPairsChanged是否做了更新,如果再更新，则iter重置为0,继续遍历
        # 直到更新结束,alphaPairsChanged没变化,则退出遍历
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number:%d ' % iter)
    return b, alphas


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        '''
        实现成员变量的填充
        :param dataMatIn:
        :param classLabels:
        :param C:
        :param toler:
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 添加了一个m*2的矩阵成员变量,该变量的第一列给出的是eCache是否有效的标志位,第二列给出的是实际的E值
        self.eCache = mat(zeros((self.m, 2)))


def calcEk(oS, k):
    '''
    计算E值并返回
    :param oS:
    :param k:
    :return:
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''
    选择合适第二个alpha或者说内循环的alpha值以保证在每次优化中采用最大步长
    :param i: 下标值
    :param oS: optStruct对象
    :param Ei: 初始的alpha值
    :return:
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 将输入值Ei设置为有效的
    oS.eCache[i] = Ei
    # 构建一个非0列表
    # nonzero返回一个列表,该列表中包含以输入列表为目录的列表值,这里的值是非零的
    # nonzero返回的是非零E值岁对应的alpha值,而不是E值本身
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # 在所有的值上进行循环并选择最大的那个值
        for k in validEcacheList:
            if k == i: continue
            # 求Ek的误差值,预测值-真实值
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 选择最大的步长
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次循环,就随机选择一个alpha值
        j = selectJrand(i, oS.m)
        # 求Ek的误差值
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    计算误差值并存入缓存当中,在对alpha值进行优化之后会用到该值
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    '''
    寻找巨册边界的优化例程
    :param i:
    :param oS:
    :return:
    '''
    # 计算误差值
    Ei = calcEk(oS, i)
    # 不管是正间隔还是副间隔都会被测试,同时也要检查alpha值,以保证其不能等于0或C
    # 由于后面alpha值小于0或大于C时将被调整为0或C,所以一旦他们等于这两个值的话,就表示他们已经在‘边界’上了,不能再进行优化(减小或增大)了
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差Ej对应的随机数进行优化
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[j])
        if L == H: print('L==H');return 0
        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
              oS.X[j, :] * oS.X[j, :].T
        if eta >= 0: print('eta>=0');return 0
        # 计算出一个新的alphas[j]值并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough');
            return 0
        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)
        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smop(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    smo外循环代码
    :param dataMatIn:数据集
    :param classLabels:类别标签
    :param C: 不同优化问题的权重
    :param toler:容错率
    :param maxIter: 一次迭代定义为一次循环过程,而不管该循环具体做了什么事
    :param kTup:包含核函数信息的元组
    :return:
    '''
    import time
    time1 = time.time()
    # 构建一个数据结构来容纳所有数据
    oS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler)
    # 对控制函数退出的变量进行初始化
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 当迭代次数超过指定的最大值或者遍历整个集合都未对任意alpha对进行修改时，退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历任意可能的alpha
            for i in range(oS.m):
                # 调用innerL函数来选择第二个alpha,并在可能时对其进行优化处理
                # 如果有任意一对alpha值发生改变，返回1
                alphaPairsChanged += innerL(i, oS)
            print('fullSet,iter:%d, i:%d, pairs changed: %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历所有的非边界alpha值,也就是不在边界0或C上的值
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound,iter:%d, i:%d, pairs changed:%d' % (iter, i, alphaPairsChanged))
            iter += 1
        # 对for循环在非边界循环和完整遍历之间进行切换,并打印次数
        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' % iter)
        print('times:', time.time() - time1)
    # 返回常数b和alpha值
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    '''
    基于alpha值计算回归系数w
    :param alphas:拉格朗日乘子
    :param dataArr:数据集
    :param classLabels:标签类别
    :return:
    '''
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        # multiply(a,b)就是个乘法,如果a,b是两个数组,那么对应元素相乘
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
