#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
from training.AdaBoost import adaboost, boost
from numpy import *

if __name__ == "__main__":
    dataArr, classArr = adaboost.loadSimpData()
    D = mat(ones((5, 1)) / 5)
    # print(boost.buildStump(dataArr, classArr, D))
    # print(boost.adaBoostTrainDS(dataArr, classArr, 9))
    dataArr, classArr = adaboost.loadDataSet('../../data/adaBoost/horseColicTraining2.txt')
    classifierArray = boost.adaBoostTrainDS(dataArr, classArr, 10)
    testArr, testLabelArr = adaboost.loadDataSet('../../data/adaBoost/horseColicTest2.txt')
    prediction10 = boost.adaClassify(testArr, classifierArray)
    # print('prediction10: ',prediction10)
    errArr = mat(ones((67, 1)))
    print(errArr[prediction10 != mat(testLabelArr).T].sum())
