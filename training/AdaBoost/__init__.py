#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
from training.AdaBoost import adaboost, boost
from numpy import *

if __name__ == "__main__":
    dataArr, classArr = adaboost.loadSimpData()
    D = mat(ones((5, 1)) / 5)
    # print(boost.buildStump(dataArr, classArr, D))
    print(boost.adaBoostTrainDS(dataArr, classArr, 9))
