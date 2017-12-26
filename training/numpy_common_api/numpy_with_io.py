#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # ndarray对象可以保存到磁盘文件并从磁盘文件加载。 可用的 IO 功能有：
    # #load()和save()函数处理 numPy 二进制文件(带npy扩展名)
    # #loadtxt()和savetxt()函数处理正常的文本文件
    # NumPy 为ndarray对象引入了一个简单的文件格式。 这个npy文件在磁盘文件中，
    # 存储重建ndarray所需的数据、图形、dtype和其他信息，
    # 以便正确获取数组，即使该文件在具有不同架构的另一台机器上

    # numpy.save()文件将输入数组存储在具有npy扩展名的磁盘文件中
    a = np.array([1, 2, 3, 4, 5])
    np.save('outfile', a)

    # 使用load函数加载outfile.npy重建数组
    b = np.load('outfile.npy')
    print(b, '\n')

    # save()和load()函数接受一个附加的布尔参数allow_pickles。
    # Python中的pickle用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化

    # savetxt()以简单文本文件格式存储和获取数组数据，是通过savetxt()和loadtx()函数完成的
    a = np.array([1, 2, 3, 4, 5])
    np.savetxt('out.txt', a)
    b = np.loadtxt('out.txt')
    print(b, '\n')
    # savetxt()和loadtxt()数接受附加的可选参数，例如页首，页尾和分隔符
