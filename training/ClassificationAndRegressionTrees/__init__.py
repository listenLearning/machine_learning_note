#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
from training.ClassificationAndRegressionTrees import regTrees
from numpy import *

if __name__ == "__main__":
    testMat = mat(eye(4))
    # print(testMat)
    mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    print(mat1)
