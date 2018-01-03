#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"
from training.action.unsupervised.Apriori import Apriori

if __name__ == '__main__':
    dataSet = Apriori.loadDataSet()
    # print('dataSet: \n', dataSet, '\n')
    # C1 = Apriori.createC1(dataSet)
    # print('C1: \n', C1, '\n')
    # D = [set(x) for x in dataSet]
    # print('D: \n', D, '\n')
    # L1, suppData0 = Apriori.scanD(D, C1, 0.5)
    # print('L1: \n', L1, '\n')
    # print('suppData0: \n', suppData0, '\n')
    # L, supportData = Apriori.apriori(dataSet)
    # print(Apriori.aprioriGen(L[0],2))
    # rules = Apriori.generateRules(L, supportData, minConf=0.7)
    # print('rules: ',rules,'\n')
    mushDatSet = [line.split() for line in open('../../../../data/Apriori/mushroom.dat').readlines()]
    # print(mushDatSet)
    L, supportData = Apriori.apriori(mushDatSet, minSupport=0.3)
    # for item in L[1]:
    #     if item.intersection('2'): print(item)

    for item in L[3]:
        if item.intersection('2'): print(item)
