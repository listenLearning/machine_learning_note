#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np
import numpy.matlib as mat

if __name__ == "__main__":
    # NumPy 包包含一个 Matrix库numpy.matlib。此模块的函数返回矩阵而不是返回ndarray对象
    # numpy.matlib.empty(shape, dtype, order) 函数返回一个新的矩阵，而不初始化元素
    # # shape 定义新矩阵形状的整数或整数元组
    # # Dtype 可选，输出的数据类型
    # # order C 或者 F
    print(mat.empty((2, 2)), '\n')

    # numpy.matlib.zeros() 返回以零填充的矩阵
    print(mat.zeros((2, 2)), '\n')

    # numpy.matlib.ones() 返回以一填充的矩阵
    print(mat.ones((2, 2)), '\n')

    # numpy.matlib.eye(n, M,k, dtype) 返回一个矩阵，对角线元素为 1，其他位置为零
    # # n 返回矩阵的行数
    # # M 返回矩阵的列数，默认为n
    # # k 对角线的索引
    # # dtype 输出的数据类型
    print(mat.eye(n=3, M=3, k=0, dtype=np.float), '\n')

    # numpy.matlib.identity()函数返回给定大小的单位矩阵。单位矩阵是主对角线元素都为 1 的方阵
    print(mat.identity(5, dtype=np.int), '\n')

    # numpy.matlib.rand()函数返回给定大小的填充随机值的矩阵
    print(mat.rand(3, 3), '\n')
    # 注意，矩阵总是二维的，而ndarray是一个 n 维数组。 两个对象都是可互换的

    i = np.matrix('1,2;3,4')
    j = np.asarray(i)
    k = np.asmatrix(j)
    print(i, '\n')
    print(j, '\n')
    print(k, '\n')
