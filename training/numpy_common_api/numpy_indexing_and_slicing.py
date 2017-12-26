#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # ndarray对象的内容可以通过索引或切片来访问和修改，就像 Python 的内置容器对象一样
    # ndarray对象中的元素遵循基于零的索引。 有三种可用的索引方法类型： 字段访问，基本切片和高级索引
    # 基本切片是 Python 中基本切片概念到 n 维的扩展。
    # 通过将start，stop和step参数提供给内置的slice函数来构造一个 Python slice对象。
    # 此slice对象被传递给数组来提取数组的一部分
    # 示例
    # 分别用起始，终止和步长值2，7和2定义切片对象
    a = np.arange(10)
    s1 = slice(2, 7, 2)
    print(a[s1], "\n")

    # 由冒号分隔的切片参数(start:stop:step)直接提供给ndarray对象
    b = a[2:7:2]
    print(b, "\n")

    # 只输入一个参数，则将返回与索引对应的单个项目
    c = a[5]
    print(c, "\n")

    # 使用a:，则从该索引向后的所有项目将被提取
    d = a[2:]
    print(d, "\n")

    # 使用两个参数(以:分隔)，则对两个索引(不包括停止索引)之间的元素以默认步骤进行切片
    e = a[2:5]
    print(e, "\n")

    # 多维ndarray
    a1 = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    print(a1, "\n")
    # 对始于索引的元素进行切片
    print(a1[1:], "\n")

    # 切片还可以包括省略号(...)，来使选择元组的长度与数组的维度相同。
    # 如果在行位置使用省略号，它将返回包含行中元素的ndarray
    # # 返回第二列元素的数组
    print(a1[..., 1], "\n")
    # # 从第二行切片所有元素
    print(a1[1, ...], "\n")
    # # 从第二列向后切片所有元素
    print(a1[..., 1:], "\n")
