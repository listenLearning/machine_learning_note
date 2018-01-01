#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # numpy.asarray：类似numpy.array，但参数比较少,构造如下：
    # numpy.asarray(a, dtype = None, order = None)
    # # a：任意形式的输入参数，比如列表、列表的元组、元组、元组的元组、元组的列表
    # # dtype：通常，输入数据的类型会应用到返回的ndarray
    # # order：'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组
    # 示例
    # 1.将列表转换为 ndarray
    x = [1, 2, 3]
    a = np.asarray(x)
    print(a, "\n")
    # 2.设置了 dtype
    b = np.asarray(x, dtype=np.float)
    print(b, "\n")
    # 3.来自元组的 ndarray
    y = (1, 2, 3)
    c = np.asarray(y)
    print(c, "\n")
    # 4.来自元组列表的 ndarray
    z = [(1, 2, 3), (4, 5, 6)]
    d = np.asarray(z)
    print(d, "\n")

    # numpy.frombuffer 将缓冲区解释为一维数组。 暴露缓冲区接口的任何对象都用作参数来返回ndarray
    # numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
    # # buffer 任何暴露缓冲区接口的对象
    # # dtype 返回数组的数据类型，默认为float
    # # count 需要读取的数据数量，默认为-1，读取所有数据
    # # offset 需要读取的起始位置，默认为0
    # 示例
    s = 'Hello World'
    e = np.frombuffer(s, dtype="S1")
    print(e, "\n")

    # numpy.fromiter 从任何可迭代对象构建一个ndarray对象，返回一个新的一维数组,构造如下
    # numpy.fromiter(iterable, dtype, count = -1)
    # # iterable 任何可迭代对象
    # # dtype 返回数组的数据类型
    # # count 需要读取的数据数量，默认为-1，读取所有数据
    # 示例
    # 使用 range 函数创建列表对象
    list = range(5)
    # 从列表中获得迭代器
    it = iter(list)
    f = np.fromiter(it, dtype=np.float)
    print(f, "\n")
