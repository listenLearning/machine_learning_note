#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.action.supervised.svm import svmMLiA, svmMliADigitalRecognition

if __name__ == "__main__":
    # dataArr, labelArr = svmMLiA.loadDataSet('../../../../data/svm/testSet.txt')
    # b, alphas = svmMLiA.smop(array(dataArr), array(labelArr), 0.6, 0.001, 40)
    # # print(b, '\n')
    # # print(alphas[alphas > 0], '\n')
    # w = svmMLiA.calcWs(alphas, dataArr, labelArr)
    # print('\n', 'ws:', w)
    # svmMLiAWithKernel.testRbf()
    svmMliADigitalRecognition.testDigits()
