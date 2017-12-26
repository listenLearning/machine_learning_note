#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # numpy.nditer：一个有效的多维迭代器对象，可以用于在数组上进行迭代。
    # 数组的每个元素可使用 Python 的标准Iterator接口来访问

    # 示例
    a = np.arange(0, 60, 5)
    a = a.reshape(3, 4)
    print(a, "\n")
    for x in np.nditer(a):
        print(x)
    print("")

    # 迭代的顺序匹配数组的内容布局，而不考虑特定的排序。 这可以通过迭代上述数组的转置来看到
    # 示例
    # 转置
    b = a.T
    print(b, "\n")
    for x in np.nditer(b):
        print(b)
    print("")

    # 迭代顺序
    # 如果相同元素使用 F 风格顺序存储，则迭代器选择以更有效的方式对数组进行迭代
    c = b.copy(order='C')
    print(c, "\n")
    for x in np.nditer(c):
        print(x)
    print("")

    d = b.copy(order="F")
    print(d, "\n")
    for x in np.nditer(d):
        print(x)
    print("")

    # 通过显式提醒，来强制nditer对象使用某种顺序
    for x in np.nditer(a, order="C"):
        print(x)
    print("")

    # 修改数组的值
    # nditer对象有另一个可选参数op_flags。
    # 其默认值为只读，但可以设置为读写或只写模式。 这将允许使用此迭代器修改数组元素
    # 示例
    for x in np.nditer(a, op_flags=['readwrite']):
        x[...] = x * 2
    print(a, "\n")

    # 外部循环
    # nditer类的构造器拥有flags参数，它可以接受下列值
    # # c_index：可以跟踪 C 顺序的索引
    # # f_index：可以跟踪 Fortran 顺序的索引
    # # multi-index：每次迭代可以跟踪一种索引类型
    # # external_loop：给出的值是具有多个值的一维数组，而不是零维数组
    # 示例
    # 迭代器遍历对应于每列的一维数组
    a = np.arange(0, 60, 5)
    a = a.reshape(3, 4)
    for x in np.nditer(a, flags=['external_loop'], order='F'):
        print(x)
    print("")

    # 广播迭代
    # 如果两个数组是可广播的，nditer组合对象能够同时迭代它们。
    # 假设数组a具有维度 3X4，并且存在维度为 1X4 的另一个数组b，则使用以下类型的迭代器(数组b被广播到a的大小)。
    # 示例
    b = np.array([1, 2, 3, 4], dtype=int)
    for x, y in np.nditer([a, b]):
        print("%d:%d" % (x, y))
    print("")
