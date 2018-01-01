#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # NumPy 支持比 Python 更多种类的数值类型
    # NumPy 数字类型是dtype(数据类型)对象的实例，每个对象具有唯一的特征

    # 数据类型及描述
    # bool_：存储为一个字节的布尔值(真或假)
    # int_：默认整数，相当于 C 的long，通常为int32或int64
    # intc：相当于 C 的int，通常为int32或int64
    # intp：用于索引的整数，相当于 C 的size_t，通常为int32或int64
    # int8：字节(-128 ~ 127)
    # int16：16 位整数(-32768 ~ 32767)
    # int32：32 位整数(-2147483648 ~ 2147483647)
    # int64：64 位整数(-9223372036854775808 ~ 9223372036854775807)
    # uint8：8 位无符号整数(0 ~ 255)
    # uint16：16 位无符号整数(0 ~ 65535)
    # uint32：32 位无符号整数(0 ~ 4294967295)
    # uint64：64 位无符号整数(0 ~ 18446744073709551615)
    # float_：float64的简写
    # float16半精度浮点：符号位，5 位指数，10 位尾数
    # float32单精度浮点：符号位，8 位指数，23 位尾数
    # float64双精度浮点：符号位，11 位指数，52 位尾数
    # complex_：complex128的简写
    # complex64：复数，由两个 32 位浮点表示(实部和虚部)
    # complex128：复数，由两个 64 位浮点表示(实部和虚部)

    # 数据类型对象 (dtype)
    # 数据类型对象描述了对应于数组的固定内存块的解释，取决于以下方面
    # 1.数据类型(整数、浮点或者 Python 对象)
    # 2.数据大小
    # 3.字节序(小端或大端)
    # 4.在结构化类型的情况下，字段的名称，每个字段的数据类型，和每个字段占用的内存块部分
    # 5.如果数据类型是子序列，它的形状和数据类型
    # 字节顺序取决于数据类型的前缀<,>。
    # <：意味着编码是小端(最小有效字节存储在最小地址中)。
    # >：意味着编码是大端(最大有效字节存储在最小地址中)

    # dtype可由以下语法构造
    # numpy.dtype(object, align, copy)
    # 参数为：
    # Object：被转换为数据类型的对象
    # Align：如果为true，则向字段添加间隔，使其类似 C 的结构体
    # Copy：生成dtype对象的新副本，如果为flase，结果是内建数据类型对象的引用

    # 示例1
    # 使用数组标量类型
    a = np.dtype(np.int32)
    print(a, "\n")

    # int8，int16，int32，int64 可替换为等价的字符串 'i1'，'i2'，'i4'，以及其他
    b = np.dtype("i4")
    print(b, "\n")

    # 使用端记号
    c = np.dtype('>i4')
    print(c, "\n")

    # 示例2
    # 首先创建结构化数据类型
    d = np.dtype([('age', np.int8)])
    print(d, "\n")
    # 将其应用于 ndarray 对象
    e = np.array([(10,), (20,), (30,)], dtype=d)
    print(e, "\n")
    print(e['age'], "\n")

    # 示例3
    student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
    print(student, "\n")
    a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
    print(a, "\n")

    # 每个内建类型都有一个唯一定义它的字符代码：
    # 'b'：布尔值，'i'：符号整数，'u'：无符号整数，'f'：浮点，'c'：复数浮点
    # 'm'：时间间隔，'M'：日期时间，'O'：Python 对象，'S', 'a'：字节串
    # 'U'：Unicode，'V'：原始数据(void)
