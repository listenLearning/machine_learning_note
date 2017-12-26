#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # NumPy 包含大量的各种数学运算功能,包括标准的三角函数，算术运算的函数，复数处理函数等

    # 三角函数：弧度制单位的给定角度返回三角函数比值
    a = np.array([0, 30, 45, 60, 90])
    # 通过× pi/180转化为弧度
    print(np.sin(a * np.pi / 180), '\n')
    # 数组中角度的余弦值
    print(np.cos(a * np.pi / 180), '\n')
    # 数组中角度的正切值
    print(np.tan(a * np.pi / 180), '\n')

    # arcsin，arccos，和arctan函数返回给定角度的sin，cos和tan的反三角函数。
    # 这些函数的结果可以通过numpy.degrees()函数通过将弧度制转换为角度制来验证
    sin = np.sin(a * np.pi / 180)
    # 计算角度的反正弦，返回值以弧度为单位
    inv = np.arcsin(sin)
    print(inv, '\n')
    # 通过转化为角度制来检查结果
    print(np.degrees(inv), '\n')
    cos = np.cos(a * np.pi / 180)
    print(cos, '\n')
    # 反余弦
    inv = np.arccos(cos)
    print(inv, '\n')
    # 角度制单位
    print(np.degrees(inv), '\n')
    tan = np.tan(a * np.pi / 180)
    print(tan, '\n')
    inv = np.arctan(tan)
    print(inv, '\n')
    # 角度制单位
    print(np.degrees(inv), '\n')

    # 舍入函数
    # numpy.around(a,decimals) 返回四舍五入到所需精度的值
    # # a 输入数组
    # # decimals 要舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
    a = np.array([1.0, 5.55, 123, 0.567, 25.532])
    print(a, '\n')
    # 舍入
    print(np.around(a), '\n')
    print(np.around(a, decimals=1), '\n')
    print(np.around(a, decimals=-1), '\n')

    # numpy.floor() 返回不大于输入参数的最大整数。 即标量x 的下限是最大的整数i ，使得i <= x。
    # 注意在Python中，向下取整总是从 0 舍入
    a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
    print(a, '\n')
    print(np.floor(a), '\n')

    # numpy.ceil() 返回输入值的上限，即，标量x的上限是最小的整数i ，使得i> = x
    print(np.ceil(a), '\n')
