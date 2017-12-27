#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from numpy import *


def loadSimpData():
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels
