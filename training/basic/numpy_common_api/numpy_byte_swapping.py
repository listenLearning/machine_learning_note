#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # numpy.ndarray.byteswap()函数在两个表示：大端和小端之间切换
    a = np.array([1, 256, 8755], dtype=np.int16)
    print(a, '\n')
    # 以十六进制表示内存中的数据
    print(map(hex, a), '\n')
    # byteswap() 函数通过传入 true 来原地交换
    print(a.byteswap(True), '\n')
    # 十六进制形式
    print(map(hex, a), '\n')
