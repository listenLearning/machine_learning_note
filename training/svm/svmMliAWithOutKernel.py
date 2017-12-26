#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
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
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag


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


def calcEk(oS, k):
    '''
    计算E值并返回
    :param oS:
    :param k:
    :return:
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def updateEkK(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


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


def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJK(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


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


def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoPK(dataMatIn, classLabels, C, toler, maxIter):  # full Platt SMO
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


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
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
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
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H"); return 0
        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0: print("eta>=0"); return 0
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
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def kernelTrans(X, A, kTup):
    '''
    核转换函数
    :param X:
    :param A:
    :param kTup:包含核函数信息的元组,第一个参数是描述所用核函数类型的一个字符串,其它两个则都是核函数可能需要的可选参数
    :return:
    '''
    # 构建列向量
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    # 检查元组以确定核函数的类型
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        # 遍历以对矩阵中每个元素计算搞死函数的值
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        # 将计算过程应用到整个向量
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        # 遇到不识别的元组,程序抛出异常,保证程序的正常运行
        raise NameError('Houston We Have a Problem -- Taht Kernel is not recognized')
    return K
