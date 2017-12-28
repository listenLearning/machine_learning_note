#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.regression import regression as re
from training.regression import locallyWeightedLinearRegression as lo
from numpy import *

if __name__ == "__main__":
    xArr, yArr = re.loadDataSet('../../data/regression/ex0.txt')
    ws = re.standRegres(xArr, yArr)
    # print('ws: ', ws)
    # re.plotRegres(xArr, yArr, ws)
    yHat = lo.lwlrTest(array(xArr), array(xArr), array(yArr), 0.003)
    # print(yHat)
    lo.plotLwlr(array(xArr), array(yArr), array(yHat))
