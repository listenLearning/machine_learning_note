#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"
from training.action.supervised.CART import regTrees
from numpy import *

if __name__ == '__main__':
    # testMat = mat(eye(4))
    # print(testMat)
    # mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # print(mat1)
    # myDat = regTrees.loadDataSet('../../../../data/cart/ex00.txt')
    # myDat1 = regTrees.loadDataSet('../../../../data/cart/ex0.txt')
    # myDat2 = regTrees.loadDataSet('../.././../data/cart/ex2.txt')
    # print(regTrees.createTree(mat(myDat)))
    # print(regTrees.createTree(mat(myDat1)))
    # myTree = regTrees.createTree(dataSet=mat(myDat2), ops=(0, 1))
    # myDatTest = regTrees.loadDataSet('../../../../data/cart/ex2test.txt')
    # myMat2Test = mat(myDatTest)
    # print(regTrees.prune(myTree, myMat2Test))
    # myMat2 = mat(regTrees.loadDataSet('../../../../data/cart/exp2.txt'))
    # print(regTrees.createTree(myMat2, leafType=regTrees.modelLeaf, errType=regTrees.modelErr, ops=(1, 10)))
    trainMat = mat(regTrees.loadDataSet('../../../../data/cart/bikeSpeedVsIq_train.txt'))
    testMat = mat(regTrees.loadDataSet('../../../../data/cart/bikeSpeedVsIq_test.txt'))
    myTree = regTrees.createTree(trainMat, ops=(1, 20))
    yHat = regTrees.createForeCast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1], '\n')
    myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1, 20))
    yHat = regTrees.createForeCast(myTree, testMat[:, 0], regTrees.modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1], '\n')
