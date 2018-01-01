#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
import numpy as np

if __name__ == "__main__":
    # ndarray.shape 这一数组属性返回一个包含数组维度的元组，它也可以用于调整数组大小
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a, "\n")
    print(a.shape, "\n")
    # 调整数组大小
    a.shape = (3, 2)
    print(a, "\n")
    # 通过reshape函数来调整数组大小
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(b, "\n")
    b.reshape(3, 2)
    print(b, "\n")

    # ndarray.ndim 这一数组属性返回数组的维数
    # 等间隔数字的数组
    c = np.arange(24)
    print(c, "\n")
    print(c.ndim, "\n")
    # 调整大小
    d = c.reshape(2, 4, 3)
    print(d, "\n")

    # numpy.itemsize 这一数组属性返回数组中每个元素的字节单位长度
    # 数组的 dtype 为 int8(一个字节)
    e = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    print(e.itemsize, "\n")
    # 数组的 dtype 为 float32(四个字节)
    f = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(f.itemsize, "\n")

    # numpy.flags 对象拥有以下属性。这个函数返回了它们的当前值
    # 属性及描述
    # # C_CONTIGUOUS (C) 数组位于单一的、C 风格的连续区段内
    # # F_CONTIGUOUS (F) 数组位于单一的、Fortran 风格的连续区段内
    # # OWNDATA (O) 数组的内存从其它对象处借用
    # # WRITEABLE (W) 数据区域可写入。 将它设置为flase会锁定数据，使其只读
    # # ALIGNED (A) 数据和任何元素会为硬件适当对齐
    # # UPDATEIFCOPY (U) 这个数组是另一数组的副本。当这个数组释放时，源数组会由这个数组中的元素更新
    # 展示当前的标志
    g = np.array([1, 2, 3, 4, 5])
    print(g.flags, "\n")
