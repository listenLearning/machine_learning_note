#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # numpy.arange返回ndarray对象，包含给定范围内的等间隔值,构造如下:
    # numpy.arange(start, stop, step, dtype)
    # # start 范围的起始值，默认为0
    # # stop 范围的终止值(不包含)
    # # step 两个值的间隔(步长)，默认为1
    # # dtype 返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型
    # 示例
    a = np.arange(5)
    print(a, "\n")
    # 设置数据类型
    b = np.arange(5, dtype=float)
    print(b, "\n")
    # 设置起始值终止值和步长
    c = np.arange(10, 20, 2)
    print(c, "\n")

    # numpy.linspace 类似于arange()函数。 在此函数中，指定了范围之间的均匀间隔数量，而不是步长,构造如下:
    # numpy.linspace(start, stop, num, endpoint, retstep, dtype)
    # # start 序列的起始值
    # # stop 序列的终止值，如果endpoint为true，该值包含于序列中
    # # num 要生成的等间隔样例数量，默认为50
    # # endpoint 序列中是否包含stop值，默认为ture
    # # retstep 如果为true，返回样例，以及连续数字之间的步长
    # # dtype 输出ndarray的数据类型
    # 示例
    d = np.linspace(10, 20, 5)
    print(d, "\n")
    # 将 endpoint 设为 false
    e = np.linspace(10, 20, 5, endpoint=False)
    print(e, "\n")
    # 输出 retstep 值
    f = np.linspace(10, 20, 5, retstep=True)
    print(f, "\n")

    # numpy.logspace：返回一个ndarray对象，其中包含在对数刻度上均匀分布的数字。 刻度的开始和结束端点是某个底数的幂，通常为 10
    # numpy.logscale(start, stop, num, endpoint, base, dtype)
    # # start 起始值是base ** start
    # # stop 终止值是base ** stop
    # # num 范围内的数值数量，默认为50
    # # endpoint 如果为true，终止值包含在输出数组当中
    # # base 对数空间的底数，默认为10
    # # dtype 输出数组的数据类型，如果没有提供，则取决于其它参数
    # 示例
    # 默认底数是 10
    g = np.logspace(1.0, 2.0, num=10)
    print(g, "\n")
    # 将对数空间的底数设置为 2
    h = np.logspace(1, 10, num=10, base=2)
    print(h, "\n")
