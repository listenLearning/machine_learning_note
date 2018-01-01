#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

import numpy as np

if __name__ == "__main__":
    # 排序、搜索和计数函数,排序函数实现不同的排序算法，
    # 每个排序算法的特征在于执行速度，最坏情况性能，所需的工作空间和算法的稳定性
    '''
        种类 	            速度 	最坏情况 	工作空间 	稳定性
    'quicksort'(快速排序) 	1 	    O(n^2) 	        0 	    否
    'mergesort'(归并排序) 	2 	    O(n*log(n)) 	~n/2 	是
    'heapsort'(堆排序) 	    3 	    O(n*log(n)) 	0 	    否
    '''
    # numpy.sort(a, axis, kind, order) 返回输入数组的排序副本
    # # a 要排序的数组
    # # axis 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序
    # # kind 默认为'quicksort'(快速排序)
    # # order 如果数组包含字段，则是要排序的字段
    a = np.array([[3, 7], [9, 1]])
    print(np.sort(a), '\n')
    print(np.sort(a, axis=0), '\n')
    # 在 sort 函数中排序字段
    dt = np.dtype([('name', 'S10'), ('age', int)])
    a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)
    print(a, '\n')
    print(np.sort(a, order='name'), '\n')

    # numpy.argsort()函数对输入数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。
    # 这个索引数组用于构造排序后的数组
    x = np.array([3, 1, 2])
    # 对 x 调用 argsort() 函数
    y = np.argsort(x)
    print(y, '\n')
    # 以排序后的顺序重构原数组
    print(x[y], '\n')
    # 使用循环重构原数组
    for i in y:
        print(x[i], )
    print("")

    # numpy.lexsort() 使用键序列执行间接排序。 键可以看作是电子表格中的一列。
    # 该函数返回一个索引数组，使用它可以获得排序数据。
    # 注意，最后一个键恰好是 sort 的主键
    nm = ('raju', 'anil', 'ravi', 'amar')
    dv = ('f.y.', 's.y.', 's.y.', 'f.y.')
    ind = np.lexsort((dv, nm))
    # 调用 lexsort() 函数
    print(ind, '\n')
    # 使用这个索引来获取排序后的数据
    print([nm[i] + ", " + dv[i] for i in ind], '\n')

    # NumPy 模块有一些用于在数组内搜索的函数。 提供了用于找到最大值，最小值以及满足给定条件的元素的函数
    # numpy.argmax() 和 numpy.argmin(),这两个函数分别沿给定轴返回最大和最小元素的索引
    a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])
    print(a, '\n')
    print(np.argmax(a), '\n')
    maxindex = np.argmax(a, axis=0)
    print(maxindex, '\n')

    # numpy.nonzero()函数返回输入数组中非零元素的索引
    a = np.array([[30, 40, 0], [0, 20, 10], [50, 0, 60]])
    print(a, '\n')
    print(np.nonzero(a), '\n')

    # numpy.where() 返回输入数组中满足给定条件的元素的索引
    x = np.arange(9.).reshape(3, 3)
    print(x, '\n')
    y = np.where(x > 3)
    print(y, '\n')
    print(x[y], '\n')

    # numpy.extract() 返回满足任何条件的元素
    x = np.arange(9.).reshape(3, 3)
    # 定义条件
    condition = np.mod(x, 2) == 0
    print(condition, '\n')
    # 使用条件提取元素
    print(np.extract(condition, x), '\n')
