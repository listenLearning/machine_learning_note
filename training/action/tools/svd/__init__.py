#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from numpy import *
from numpy import linalg as lin
from training.action.tools.svd import svdRec

if __name__ == '__main__':
    # U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
    # print('U:\t', U, '\n')
    # print('Sigma:\t', Sigma, '\n')
    # print('VT:\t', VT, '\n')
    myData = mat(svdRec.loadExData())
    # U, Sigma, VT = lin.svd(Data)
    # # print(Sigma)
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])
    # print(svdRec.ecludSim(myData[:, 0], myData[:, 4]))
    # print(svdRec.ecludSim(myData[:, 0], myData[:, 0]))
    # print(svdRec.cosSim(myData[:, 0], myData[:, 4]))
    # print(svdRec.cosSim(myData[:, 0], myData[:, 0]))
    # print(svdRec.pearsSim(myData[:, 0], myData[:, 4]))
    # print(svdRec.pearsSim(myData[:, 0], myData[:, 0]))
    # myData[0,1]=myData[0,0]=myData[1,0]=myData[2,0]=4
    # myData[3,3]=2
    # # print(myData)
    # # print(svdRec.recommend(myData,2))
    # # print(svdRec.recommend(myData, 2,simMass=svdRec.ecludSim))
    # print(svdRec.recommend(myData, 2,simMass=svdRec.pearsSim))
    # U, Sigma, VT = lin.svd(mat(svdRec.loadExData2_1()))
    # Sigma2 = Sigma ** 2
    # print(sum(Sigma2)*0.9)
    # print(sum(Sigma2[:5]))
    # print(eye(5))
    # myMat = mat(svdRec.loadExData2_1())
    # print(svdRec.recommend(myMat, 1, estMethod=svdRec.svdEst))
    svdRec.imgCompress('../../../../data/svd/0_5.txt')
