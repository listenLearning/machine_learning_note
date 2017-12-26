#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # 算数运算 用于执行算术运算(如add()，subtract()，multiply()
    # 和divide())的输入数组必须具有相同的形状或符合数组广播规则

    a = np.arange(9, dtype=np.float_).reshape(3, 3)
    print(a, '\n')
    b = np.array([10, 10, 10])
    print(b, '\n')
    # 两个数组相加
    print(np.add(a, b), '\n')
    # 两个数组相减：
    print(np.subtract(a, b), '\n')
    # 两个数组相乘
    print(np.multiply(a, b), '\n')
    # 两个数组相除
    print(np.divide(a, b), '\n')

    # numpy.reciprocal() 返回参数逐元素的倒数。由于 Python 处理整数除法的方式，
    # 对于绝对值大于 1 的整数元素，结果始终为 0， 对于整数 0，则发出溢出警告
    a = np.array([0.25, 1.33, 1, 100])
    print(a, '\n')
    print(np.reciprocal(a), '\n')
    b = np.array([100], dtype=int)
    print(b, '\n')
    print(np.reciprocal(b), '\n')

    # numpy.power() 将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
    a = np.array([10, 100, 1000])
    print(a, '\n')
    print(np.power(a, 2), '\n')
    b = np.array([1, 2, 3])
    print(b, '\n')
    print(np.power(a, b), '\n')

    # numpy.mod() 返回输入数组中相应元素的除法余数。 函数numpy.remainder()也产生相同的结果
    a = np.array([10, 20, 30])
    b = np.array([3, 5, 7])
    print(a, '\n')
    print(b, '\n')
    print(np.mod(a, b), '\n')
    print(np.remainder(a, b), '\n')

    # 复数操作
    a = np.array([-5.6j, 0.2j, 11., 1 + 1j])
    print(a, '\n')
    # numpy.real() 返回复数类型参数的实部。
    print(np.real(a), '\n')

    # numpy.imag() 返回复数类型参数的虚部。
    print(np.imag(a), '\n')

    # numpy.conj() 返回通过改变虚部的符号而获得的共轭复数。
    print(np.conj(a), '\n')

    # numpy.angle() 返回复数参数的角度。 函数的参数是degree。
    # 如果为true，返回的角度以角度制来表示，否则为以弧度制来表示
    print(np.angle(a), '\n')
    print(np.angle(a, deg=True))
