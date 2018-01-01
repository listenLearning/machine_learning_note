#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # 统计函数 用于从数组中给定的元素中查找最小，最大，百分标准差和方差等
    # numpy.amin() 和 numpy.amax() 从给定数组中的元素沿指定轴返回最小值和最大值
    a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
    print(a, '\n')
    # axis 1:横轴 0:纵轴 i:按维处理
    print(np.amin(a, axis=1), '\n')
    print(np.amax(a, 0), '\n')

    # numpy.ptp()函数返回沿轴的值的范围(最大值 - 最小值)
    print(np.ptp(a), '\n')
    print(np.ptp(a, axis=1), '\n')

    # numpy.percentile(a, q, axis) 百分位数是统计中使用的度量，表示小于这个值得观察值占某个百分比
    # # a 输入数组
    # # q 要计算的百分位数，在 0 ~ 100 之间
    # # axis 沿着它计算百分位数的轴
    a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])
    print(a, '\n')
    print(np.percentile(a, 50), '\n')
    print(np.percentile(a, 50, axis=1), '\n')

    # numpy.median() 中值,将数据样本的上半部分与下半部分分开的值
    print(np.median(a), '\n')
    print(np.median(a, axis=0), '\n')

    # numpy.mean() 算术平均值是沿轴的元素的总和除以元素的数量。
    # numpy.mean()函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算
    a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    print(a, '\n')
    print(np.mean(a, axis=1), '\n')

    # numpy.average() 加权平均值是由每个分量乘以反映其重要性的因子得到的平均值
    # numpy.average()函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值
    # 该函数可以接受一个轴参数。 如果没有指定轴，则数组会被展开。
    # 考虑数组[1,2,3,4]和相应的权重[4,3,2,1]，通过将相应元素的乘积相加，并将和除以权重的和，来计算加权平均值
    # 加权平均值 = (1*4+2*3+3*2+4*1)/(4+3+2+1)
    a = np.array([1, 2, 3, 4])
    print(a, '\n')
    # 不指定权重时相当于 mean 函数
    print(np.average(a), '\n')
    wts = np.array([4, 3, 2, 1])
    print(np.average(a, weights=wts), '\n')
    # 如果 returned 参数设为 true，则返回权重的和
    print(np.average(a, weights=wts, returned=True))
    a = np.arange(6).reshape(3, 2)
    wt = np.array([3, 5])
    print(np.average(a, axis=1, weights=wt, returned=True), '\n')

    # 标准差 标准差是与均值的偏差的平方的平均值的平方根
    # std = sqrt(mean((x - x.mean())**2))
    print(np.std([1, 2, 3, 4]), '\n')

    # 方差 偏差的平方的平均值，即mean((x - x.mean())** 2)。
    # 换句话说，标准差是方差的平方根
    print(np.var([1, 2, 3, 4]), '\n')
