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
    initSet = fpGrowth.createInitSet(simpDat)
    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
    # myFPtree.disp()
    # print(fpGrowth.findPrefixPath('x',myHeaderTab['x'][1]))
    freqItem = []
    # fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItem)
    fpGrowth.sort(myHeaderTab)
