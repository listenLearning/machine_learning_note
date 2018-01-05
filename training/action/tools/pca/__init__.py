#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.action.tools.pca import pca
from numpy import *

if __name__ == "__main__":
    path = '../../../../data/PCA/testSet.txt'
    # dataMat = pca.loadDataSet(path)
    # lowDMat, reconMat = pca.pca(dataMat, 2)
    # # print(shape(lowDMat))
    # pca.drawingWithMatplot(dataMat, reconMat)
    path = '../../../../data/PCA/secom.data'
    pca.replaceNanWithMean(path)
