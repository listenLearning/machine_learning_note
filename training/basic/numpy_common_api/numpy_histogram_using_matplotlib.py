#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # NumPy 有一个numpy.histogram()函数，它是数据的频率分布的图形表示。
    # 水平尺寸相等的矩形对应于类间隔，称为bin，变量height对应于频率
    # numpy.histogram()函数将输入数组和bin作为两个参数。 bin数组中的连续元素用作每个bin的边界
    a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    hist, bins = np.histogram(a, bins=[0, 20, 40, 60, 80, 100])
    print(hist, '\n')
    print(bins, '\n')

    # plt() Matplotlib 可以将直方图的数字表示转换为图形。
    # pyplot子模块的plt()函数将包含数据和bin数组的数组作为参数，并转换为直方图
    plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
    plt.title('histogram')
    plt.show()
