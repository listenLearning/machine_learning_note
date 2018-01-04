#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from training.action.unsupervised.fpGrowth import fpGrowth
from numpy import *

if __name__ == '__main__':
    path = '../../../../data/fpGrowth/kosarak.dat'
    # rootNode = fpGrowth.treeNode('pyrmid', 9, None)
    # rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)
    # rootNode.disp()
    # rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)
    # rootNode.disp()
    simpDat = fpGrowth.loadSimpDat()
    # print(simpDat)
    initSet = fpGrowth.createInitSet(simpDat)
    # print(initSet)
    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
    myFPtree.disp()
