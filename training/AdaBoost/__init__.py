#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
'''
AdaBoost
优点: 泛化错误率低,易编码,可以应用在大部分分类器上,无参数调整
缺点: 对离群点敏感
使用数据类型: 数值型和标称型数据
AdaBoost的一般流程
1.收集数据: 可以使用任意方法
2.准备数据: 依赖于所使用的弱分类器类型,这里使用的时单层决策树,这种分类器可以处理任何数据类型.
    当然也可以使用任意分类器作为弱分类器.作为弱分类器,简单分类器的效果更好
3.分析数据: 可以使用任意方法
4.训练算法: AdaBoost的大部分时间都用在训练上,分类器将多次在同一数据集上训练弱分类器
5.测试算法: 计算分类器的错误率
6.使用算法: 同SVM一样,AdaBoost预测两个类别中的一个.如果想把它应用到多个类别的场合,
    那么就要像多分类SVM中的做法一样多AdaBoost进行修改
'''
if __name__ == "__main__":
    print("")