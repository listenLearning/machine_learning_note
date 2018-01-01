#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # 字符串函数用于对dtype为numpy.string_或numpy.unicode_的数组执行向量化字符串操作,
    # 基于 Python 内置库中的标准字符串函数

    # numpy.char.add() 返回两个str或Unicode数组的逐个字符串连接
    # 连接两个字符串
    print(np.char.add(['hello'], ['world']), '\n')
    print(np.char.add(['hello', 'hi'], ['abc', 'xyz']), '\n')

    # numpy.char.multiply() 返回按元素多重连接后的字符串
    print(np.char.multiply('hello', 3), '\n')

    # numpy.char.center() 返回给定字符串的副本，其中元素位于特定字符串的中央
    print(np.char.center('hello', 20, fillchar='*'), '\n')

    # numpy.char.capitalize() 返回给定字符串的副本，其中只有第一个字符串大写
    print(np.char.capitalize('hello world'), '\n')

    # numpy.char.title() 返回字符串或 Unicode 的按元素标题转换版本
    print(np.char.title('hello how are you?'), '\n')

    # numpy.char.lower() 返回一个数组，其元素转换为小写
    print(np.char.lower(['HELLO', 'WORLD']), '\n')
    print(np.char.lower('HELLO'), '\n')

    # numpy.char.upper() 返回一个数组，其元素转换为大写
    print(np.char.upper(['hello', 'world']), '\n')
    print(np.char.upper('hello'), '\n')

    # numpy.char.split() 返回字符串中的单词列表，并使用分隔符来分割
    print(np.char.split('hello how are you?'), '\n')
    print(np.char.split('YiibaiPoint,Hyderabad,Telangana', sep=','), '\n')

    # numpy.char.splitlines() 返回元素中的行列表，以换行符分割
    print(np.char.splitlines('hello\nhow are you?'), '\n')
    print(np.char.splitlines('hello\rhow are you?'), '\n')

    # numpy.char.strip() 返回数组副本，其中元素移除了开头或者结尾处的特定字符
    print(np.char.strip('ashok arora', 'a'), '\n')
    print(np.char.strip(['arora', 'admin', 'java'], 'a'), '\n')

    # numpy.char.join() 返回一个字符串，它是序列中字符串的连接
    print(np.char.join(':', 'dmy'), '\n')
    print(np.char.join([':', '-'], ['dmy', 'ymd']), '\n')

    # numpy.char.replace() 返回字符串的副本，其中所有子字符串的出现位置都被新字符串取代
    print(np.char.replace('He is a good boy', 'is', 'was'), '\n')

    # numpy.char.decode() 按元素调用str.decode
    a = np.char.encode('hello', 'cp500')
    print(a, '\n')
    print(np.char.decode(a, 'cp500'), '\n')

    # numpy.char.encode() 按元素调用str.encode
    a = np.char.encode('hello', 'cp500')
    print(a, '\n')
