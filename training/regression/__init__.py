#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.regression import regression as re
from training.regression import locallyWeightedLinearRegression as lo
from numpy import *


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


if __name__ == "__main__":
    # xArr, yArr = re.loadDataSet('../../data/regression/ex0.txt')
    # ws = re.standRegres(xArr, yArr)
    # # print('ws: ', ws)
    # # re.plotRegres(xArr, yArr, ws)
    # yHat = lo.lwlrTest(array(xArr), array(xArr), array(yArr), 0.003)
    # # print(yHat)
    # lo.plotLwlr(array(xArr), array(yArr), array(yHat))
    abX, abY = re.loadDataSet('../../data/regression/abalone.txt')
    yHat01 = lo.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lo.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lo.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('0.1: ',rssError(abY[0:99], yHat01.transpose()))
    print('1: ', rssError(abY[0:99], yHat1.transpose()))
    print('10: ', rssError(abY[0:99], yHat10.transpose()))
