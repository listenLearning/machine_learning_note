#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # 在执行函数时，其中一些返回输入数组的副本，而另一些返回视图。
    # 当内容物理存储在另一个位置时，称为副本。另一方面，如果提供了相同内存内容的不同视图，我们将其称为视图

    # 查看当前是是视图还是副本，可以通过查看id获得

    # 无复制
    # 简单的赋值不会创建数组对象的副本。相反，它使用原始数组的相同id()来访问它。id()返回 Python 对象的通用标识符，类似于 C 中的指针。
    # 此外，一个数组的任何变化都反映在另一个数组上。 例如，一个数组的形状改变也会改变另一个数组的形状
    a = np.arange(6)
    print(a, '\n')
    # 调用 id() 函数
    print(id(a), '\n')
    # a 赋值给 b
    b = a
    print(b, id(b), '\n')
    # 修改 b 的形状
    b.shape = (3, 2)
    print(b, '\n')
    print(a, '\n')

    # 视图或浅复制
    # NumPy 拥有ndarray.view()方法，它是一个新的数组对象，并可查看原始数组的相同数据。
    # 与前一种情况不同，新数组的维数更改不会更改原始数据的维数
    a = np.arange(6).reshape(3, 2)
    print(a, '\n')
    # 创建a的视图
    b = a.view()
    print(b, '\n')
    print(id(a), id(b), '\n')
    b.shape = (2, 3)
    print(b, '\n')
    print(a, '\n')

    # 数组的切片也会创建视图
    a = np.array([[10, 10], [2, 3], [4, 5]])
    print(a, '\n')
    s = a[:, :2]
    print(s, '\n')

    # 深复制
    # ndarray.copy()函数创建一个深层副本。 它是数组及其数据的完整副本，不与原始数组共享
    a = np.array([[10, 10], [2, 3], [4, 5]])
    print(a, '\n')
    # 创建 a 的深层副本
    b = a.copy()
    print(b, '\n')
    print(id(a), id(b), '\n')
    b.shape = (2, 3)
    print(b, '\n')
    print(a, '\n')
