#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"
import numpy as np

if __name__ == "__main__":
    # bitwise_and:对输入数组中的整数的二进制表示的相应位执行位与运算
    print('13 和 17 的二进制形式：')
    a, b = 13, 17
    print(bin(a), bin(b), '\n')
    print('13 和 17 的位与：')
    print(np.bitwise_and(a, b), '\n')

    # np.bitwise_or()函数对输入数组中的整数的二进制表示的相应位执行位或运算
    print('13 和 17 的位或：')
    print(np.bitwise_or(a, b), '\n')

    # invert 计算输入数组中整数的位非结果。 对于有符号整数，返回补码
    print('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
    print(np.invert(np.array([13], dtype=np.uint8)), '\n')
    print('13 的二进制表示：')
    print(np.binary_repr(242, width=8), '\n')
    # 注意，np.binary_repr()函数返回给定宽度中十进制数的二进制表示

    # numpy.left shift()函数将数组元素的二进制表示中的位向左移动到指定位置，右侧附加相等数量的 0
    print('将 10 左移两位：')
    print(np.left_shift(10, 2), '\n')
    print(np.binary_repr(10, width=8), '\n')
    print('40 的二进制表示：')
    print(np.binary_repr(40, width=8), '\n')

    # numpy.right_shift()函数将数组元素的二进制表示中的位向右移动到指定位置，左侧附加相等数量的 0
    print('将 40 右移两位：')
    print(np.right_shift(40, 2), '\n')
