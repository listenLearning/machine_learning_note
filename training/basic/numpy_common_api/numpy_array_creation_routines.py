#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

# 更多资料请参考:http://blog.csdn.net/pipisorry/article/details/39406501
if __name__ == "__main__":
    # 新的ndarray对象可以通过任何下列数组创建例程或使用低级ndarray构造函数构造

    # numpy.empty 创建指定形状和dtype的未初始化数组,使用以下构造函数
    # numpy.empty(shape, dtype = float, order = 'C')
    # # Shape 空数组的形状，整数或整数元组
    # # Dtype 所需的输出数组类型，可选
    # # Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组
    # eg:
    a = np.empty([3, 2], dtype=np.int)
    print(a, "\n")

    # numpy.zeros 返回特定大小，以 0 填充的新数组
    # numpy.zeros(shape, dtype = float, order = 'C')
    # # Shape 空数组的形状，整数或整数元组
    # # Dtype 所需的输出数组类型，可选
    # # Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组
    b1 = np.zeros(5)
    print(b1, "\n")
    b2 = np.zeros((5,), dtype=int)
    print(b2, "\n")
    # 自定义类型
    b3 = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
    print(b3, "\n")

    # numpy.ones 返回特定大小，以 1 填充的新数组
    # numpy.ones(shape, dtype = None, order = 'C'),参数解释如numpy.zeros
    # 含有 5 个 1 的数组，默认类型为 float
    c1 = np.ones(5)
    print(c1, "\n")
    c2 = np.ones([2, 3], dtype=int)
    print(c2, "\n")
