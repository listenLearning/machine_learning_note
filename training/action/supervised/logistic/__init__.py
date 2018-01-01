#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.action.supervised.logistic import logRegres

if __name__ == "__main__":
    # Sigmoid函数 \sigma(w^Tx)=\frac{1}{1+e^-z}
    # f(x,y)梯度 \nabla f(x,y)  = \binom{\frac {\partial f(x,y)}{\partial x}}{\frac{\partial f(x,y)}{\partial y} }
    # 梯度上升算法的迭代公式 w:=w+{\alpha}^{\nabla}_wf(w)
    # dataMat, labelMat = logRegres.loadDataSet()
    # weight = logRegres.gradAscent(array(dataMat), labelMat)
    # print(weight)
    # logRegres.plotBestFit(array(weight))
    logRegres.multiTest()
