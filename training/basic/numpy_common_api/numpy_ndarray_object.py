#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "
import numpy as np

if __name__ == "__main__":
    # NumPy的主要对象是同种元素的多维数组。这是一个所有的元素都是一种类型、通过一个正整数元组索引的元素表格(通常是元素是数字)。
    # 在NumPy中维度(dimensions)叫做轴(axes)，轴的个数叫做秩(rank)。
    # NumPy 中定义的最重要的对象是称为 ndarray 的 N 维数组类型。它描述相同类型的元素集合,可以使用基于零的索引访问集合中的项目
    # ndarray中的每个元素在内存中使用相同大小的块。 ndarray中的每个元素是数据类型对象的对象(称为 dtype)
    # 基本的ndarray是使用 NumPy 中的数组函数创建的
    # def array(p_object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
    # 参数及描述
    # p_object：任何暴露数组接口方法的对象都会返回一个数组或任何(嵌套)序列
    # dtype：数组的所需数据类型，可选
    # copy：可选，默认为true，对象是否被复制
    # order：C(按行)、F(按列)或A(任意，默认)
    # subok：默认情况下，返回的数组被强制为基类数组。 如果为true，则返回子类
    # ndmin：指定返回数组的最小维数
    # 实例
    a = np.array([1, 2, 3])
    print(a, "\n")
    # 多于一个维度
    b = np.array([[1, 2], [3, 4]])
    print(b, "\n")
    # 最小维度
    c = np.array([1, 2, 3, 4, 5], ndmin=2)
    print(c, "\n")
    # dtype 参数
    d = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    print(d, "\n")
    # ndarray 对象由计算机内存中的一维连续区域组成，带有将每个元素映射到内存块中某个位置的索引方案。
    # 内存块以按行(C 风格)或按列(FORTRAN 或 MatLab 风格)的方式保存元素
