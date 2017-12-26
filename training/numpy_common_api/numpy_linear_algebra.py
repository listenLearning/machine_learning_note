#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # NumPy 包包含numpy.linalg模块，提供线性代数所需的所有功能
    # # dot 返回两个数组的点积。 对于二维向量，其等效于矩阵乘法。 对于一维数组，它是向量的内积。
    # # 对于 N 维数组，它是a的最后一个轴上的和与b的倒数第二个轴的乘积
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[11, 12], [13, 14]])
    print(np.dot(a, b), '\n')
    # 点积计算为 [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]

    # # vdot 返回两个向量的点积。 如果第一个参数是复数，
    # # 那么它的共轭复数会用于计算。 如果参数id是多维数组，它会被展开
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[11, 12], [13, 14]])
    print(np.vdot(a, b), '\n')

    # # inner 返回一维数组的向量内积。 对于更高的维度，它返回最后一个轴上的和的乘积
    print(np.inner(np.array([1, 2, 3]), np.array([0, 1, 0])), '\n')
    print(np.inner(a, b), '\n')
    # 多维矩阵内积计算如下
    # 1 * 11 + 2 * 12, 1 * 13 + 2 * 14
    # 3 * 11 + 4 * 12, 3 * 13 + 4 * 14

    # # numpy.matmul()函数返回两个数组的矩阵乘积。 虽然它返回二维数组的正常乘积，
    # # 但如果任一参数的维数大于2，则将其视为存在于最后两个索引的矩阵的栈，并进行相应广播。
    # # 另一方面，如果任一参数是一维数组，则通过在其维度上附加 1 来将其提升为矩阵，并在乘法之后被去除
    # 对于二维数组，它就是矩阵乘法
    a = [[1, 0], [0, 1]]
    b = [[4, 1], [2, 2]]
    print(np.matmul(a, b), '\n')
    # 二维和一维运算
    a = [[1, 0], [0, 1]]
    b = [1, 2]
    print(np.matmul(a, b), '\n')
    print(np.matmul(a, a), '\n')
    # 维度大于二的数组
    a = np.arange(8).reshape(2, 2, 2)
    b = np.arange(4).reshape(2, 2)
    print(np.matmul(a, b), '\n')

    # # numpy.linalg.det()行列式在线性代数中是非常有用的值。 它从方阵的对角元素计算。
    # # 对于 2×2 矩阵，它是左上和右下元素的乘积与其他两个的乘积的差。
    # # 换句话说，对于矩阵[[a，b]，[c，d]]，行列式计算为ad-bc。较大的方阵被认为是 2×2 矩阵的组合。
    # # numpy.linalg.det()函数计算输入矩阵的行列式
    a = np.array([[1, 2], [3, 4]])
    print(np.linalg.det(a), '\n')
    b = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    print(np.linalg.det(b), '\n')
    print(6 * (-2 * 7 - 5 * 8) - 1 * (4 * 7 - 5 * 2) + 1 * (4 * 8 - -2 * 2), '\n')

    # # numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解
    # 考虑以下线性方程
    '''
    x + y + z = 6
    2y + 5z = -4
    2x + 5y - z = 27
    '''
    # 使用矩阵表示为
    # [[1,1,1],[0,2,5],[2,5,-1]]*[x,y,z] = [6,-4,27]
    # 如果矩阵成为A、X和B，方程变为
    # AX = B 或 X = A^(-1)B

    # # numpy.linalg.inv() 函数来计算矩阵的逆。 矩阵的逆是这样的，如果它乘以原始矩阵，则得到单位矩阵
    x = np.array([[1, 2], [3, 4]])
    y = np.linalg.inv(x)
    print(x, '\n')
    print(y, '\n')
    print(np.dot(x, y), '\n')

    a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
    ainv = np.linalg.inv(a)
    b = np.array([[6], [-4], [27]])
    # 计算：A^(-1)B
    x = np.linalg.solve(a, b)
    print(x, '\n')
    # 这就是线性方向 x = 5, y = 3, z = -2 的解
