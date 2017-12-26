#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

import numpy as np

if __name__ == "__main__":
    # 用于处理ndarray对象中的元素

    # numpy.reshape 不改变数据的条件下修改形状
    # numpy.reshape(arr, newshape, order')
    # # arr：要修改形状的数组
    # # newshape：整数或者整数数组，新的形状应当兼容原有形状
    # # order：'C'为 C 风格顺序，'F'为 F 风格顺序，'A'为保留原顺序
    # 示例
    a = np.arange(8)
    print(a, "\n")
    b = a.reshape(4, 2)
    print(b, "\n")

    # numpy.ndarray.flat 数组上的一维迭代器
    # 示例
    print(b.flat[5], "\n")

    # numpy.ndarray.flatten 返回折叠为一维的数组副本
    # ndarray.flatten(order)
    # # order：'C' — 按行，'F' — 按列，'A' — 原顺序，'k' — 元素在内存中的出现顺序。
    # 展开数组
    print('flatten', b.flatten(), "\n")
    # 以f风格顺序展开的数组
    print('flattenF', b.flatten(order='F'), "\n")

    # numpy.ravel 返回展开的一维数组，并且按需生成副本。返回的数组和输入数组拥有相同数据类型
    # numpy.ravel(a, order)
    # # order：'C' — 按行，'F' — 按列，'A' — 原顺序，'k' — 元素在内存中的出现顺序
    print('b.reval1', b.ravel(), "\n")
    # 以 F 风格顺序调用 ravel
    print("b.ravel2", b.ravel(order="F"), "\n")

    # 两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），
    # numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
    # 而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。

    # 翻转操作
    # numpy.transpose(arr, axes)：翻转给定数组的维度，返回一个视图
    # # arr：要转置的数组
    # # axes：整数的列表，对应维度，通常所有维度都会翻转。
    c = np.arange(12).reshape(3, 4)
    print("self:", c, "\n")
    # 转置数组
    print('transpose', np.transpose(c), "\n")

    # numpy.ndarray.T：类似于numpy.transpose
    print('T', c.T, "\n")

    # numpy.rollaxis(arr, axis, start)：向后滚动特定的轴，直到一个特定位置
    # # arr：输入数组
    # # axis：要向后滚动的轴，其它轴的相对位置不会改变
    # # start：默认为零，表示完整的滚动。会滚动到特定位置
    d = np.arange(8).reshape(2, 2, 2)
    print("self", d, "\n")
    # 将轴 2 滚动到轴 0(宽度到深度)
    print("rollaxis", np.rollaxis(d, 2), "\n")

    # 将轴 0 滚动到轴 1：(宽度到高度)
    print('rollaxis', np.rollaxis(d, 2, 1), "\n")

    # numpy.swapaxes(arr, axis1, axis2)交换数组的两个轴
    # numpy.swapaxes(arr, axis1, axis2)
    # # arr：要交换其轴的输入数组
    # # axis1：对应第一个轴的整数
    # # axis2：对应第二个轴的整数
    # 交换轴 0(深度方向)到轴 2(宽度方向)
    print(np.swapaxes(d, 2, 0))

    # 修改维度
    # broadcast 模仿广播机制。 它返回一个对象，该对象封装了将一个数组广播到另一个数组的结果
    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])
    # 对y广播x
    b = np.broadcast(x, y)
    # 拥有 iterator 属性，基于自身组件的迭代器元组
    r, c = b.iters
    print(next(r), next(c), "\n")
    # shape 返回广播对象的形状
    print(b.shape, "\n")
    # 手动使用broadcast将x与y相加
    b = np.broadcast(x, y, '\n')
    c = np.empty(b.shape, '\n')
    # c.flat = [u + v for (u, v) in b]
    # 获得和 NumPy 内建的广播支持相同的结果
    print(x + y, '\n')

    # numpy.broadcast_to(array, shape, subok)：将数组广播到新形状，它在原始数组上返回只读视图，通常不连续。
    # 如果新形状不符合 NumPy 的广播规则，该函数可能会抛出ValueError
    a = np.arange(4).reshape(1, 4)
    print(a, "\n")
    print(np.broadcast_to(a, (4, 4)), "\n")

    # numpy.expand_dims(arr, axis) 通过在指定位置插入新的轴来扩展数组形状
    # # arr 输入数组
    # # axis 新轴插入的位置
    x = np.array(([1, 2], [3, 4]))
    y = np.expand_dims(x, axis=0)
    print(x, '\n', x.shape, "\n")
    print(y, '\n', y.shape, "\n")
    # # 在位置 1 插入轴
    y = np.expand_dims(x, axis=1)
    print(y, '\n', y.shape, "\n")
    print(x.ndim, y.ndim, '\n')

    # numpy.squeeze(arr, axis)：从给定数组的形状中删除一维条目
    # # arr：输入数组
    # # axis：整数或整数元组，用于选择形状中单一维度条目的子集
    g = np.arange(9).reshape(1, 3, 3)
    print(g, '\n', g.shape, '\n')
    h = np.squeeze(g)
    print(h, '\n')
    print(g.shape, h.shape, '\n')

    # 数组的连接
    # numpy.concatenate((a1, a2, ...), axis)：数组的连接是指连接。 此函数用于沿指定轴连接相同形状的两个或多个数组
    # # a1, a2, ...：相同类型的数组序列
    # # axis：沿着它连接数组的轴，默认为 0
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print(a, 'n')
    print(b, 'n')
    # 沿轴 0 连接两个数组：
    print(np.concatenate((a, b)), '\n')
    # 沿轴 1 连接两个数组：
    print(np.concatenate((a, b), axis=1), '\n')

    # numpy.stack(arrays, axis)：此函数沿新轴连接数组序列。
    # # arrays：相同形状的数组序列
    # # axis：返回数组中的轴，输入数组沿着它来堆叠
    # 沿轴 0 堆叠两个数组
    print(np.stack((a, b), 0), "\n")
    # 沿轴 1 堆叠两个数组
    print(np.stack((a, b), 1), "\n")

    # numpy.hstack：numpy.stack函数的变体，通过堆叠来生成水平的单个数组
    c = np.hstack((a, b))
    print(c, '\n')

    # numpy.vstack：numpy.stack函数的变体，通过堆叠来生成竖直的单个数组
    c = np.vstack((a, b))
    print(c, "\n")

    # 数组分割
    # numpy.split(ary, indices_or_sections, axis)：该函数沿特定的轴将数组分割为子数组
    # # ary 被分割的输入数组
    # # indices_or_sections 可以是整数，表明要从输入数组创建的，等大小的子数组的数量。 如果此参数是一维数组，则其元素表明要创建新子数组的点
    # # axis 默认为 0
    a = np.arange(9)
    # 将数组分为三个大小相等的子数组
    b = np.split(a, 3)
    print(b, "\n")
    # 将数组在一维数组中表明的位置分割：
    b = np.split(a, [4, 7])
    print(b, "\n")

    # numpy.hsplit是split()函数的特例，其中轴为 1 表示水平分割，无论输入数组的维度是什么
    a = np.arange(16).reshape(4, 4)
    # 水平分割
    b = np.hsplit(a, 2)
    print(b, '\n')

    # numpy.vsplit是split()函数的特例，其中轴为 0 表示竖直分割，无论输入数组的维度是什么
    # 竖直分割
    b = np.vsplit(a, 2)
    print(b, "\n")
    # 添加/删除元素
    # numpy.resize(arr, shape) 此函数返回指定大小的新数组。 如果新大小大于原始大小，则包含原始数组中的元素的重复副本
    # # arr：要修改大小的输入数组
    # # shape：返回数组的新形状
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.resize(a, (3, 2))
    print(b, '\n')
    b = np.resize(a, (3, 3))
    print(b, '\n')

    # numpy.append(arr, values, axis)：在输入数组的末尾添加值。 附加操作不是原地的，而是分配新的数组。
    # 此外，输入数组的维度必须匹配否则将生成ValueError
    # # arr：输入数组
    # # values：要向arr添加的值，比如和arr形状相同(除了要添加的轴)
    # # axis：沿着它完成操作的轴。如果没有提供，两个参数都会被展开。
    a = np.array([[1, 2, 3], [4, 5, 6]])
    # 向数组添加元素
    print(np.append(a, [7, 8, 9]), '\n')
    # 沿轴 1 添加元素：
    print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1), '\n')

    # numpy.insert(arr, obj, values, axis):在给定索引之前，沿给定轴在输入数组中插入值。
    # 如果值的类型转换为要插入，则它与输入数组不同。
    # 插入没有原地的，函数会返回一个新数组。 此外，如果未提供轴，则输入数组会被展开
    # # arr：输入数组
    # # obj：在其之前插入值的索引
    # # values：要插入的值
    # # axis：沿着它插入的轴，如果未提供，则输入数组会被展开
    a = np.array([[1, 2], [3, 4], [5, 6]])
    # 未传递 Axis 参数。 在插入之前输入数组会被展开。
    print(np.insert(a, 3, [11, 12]), '\n')
    # 传递了 Axis 参数。 会广播值数组来配输入数组。
    # 沿轴 1 广播：
    print(np.insert(a, 1, 11, axis=1), '\n')

    # Numpy.delete(arr, obj, axis)：返回从输入数组中删除指定子数组的新数组。
    # 与insert()函数的情况一样，如果未提供轴参数，则输入数组将展开
    # # arr：输入数组
    # # obj：可以被切片，整数或者整数数组，表明要从输入数组删除的子数组
    # # axis：沿着它删除给定子数组的轴，如果未提供，则输入数组会被展开
    a = np.arange(12).reshape(3, 4)
    # 未传递 Axis 参数。 在删除之前输入数组会被展开。
    print(np.delete(a, 5), '\n')
    # 删除第二列：
    print(np.delete(a, 1, axis=1), '\n')
    # 包含从数组中删除的替代值的切片：
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(np.delete(a, np.s_[::2]))

    # numpy.unique(arr, return_index, return_inverse, return_counts)：返回输入数组中的去重元素数组。
    # 该函数能够返回一个元组，包含去重数组和相关索引的数组。 索引的性质取决于函数调用中返回参数的类型
    # # arr：输入数组，如果不是一维数组则会展开
    # # return_index：如果为true，返回输入数组中的元素下标
    # # return_inverse：如果为true，返回去重数组的下标，它可以用于重构输入数组
    # # return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
    a = np.array([5, 2, 6, 2, 7, 5, 6, 8, 2, 9])
    # 第一个数组的去重值
    u = np.unique(a)
    print(u, '\n')
    # 去重数组的索引数组
    u, indices = np.unique(a, return_index=True)
    print(indices, '\n')
    # 去重数组的下标
    u, indices = np.unique(a, return_inverse=True)
    print(u, '\n')
    print('下标为:', indices, '\n')
    # 使用下标重构原数组
    print(u[indices], '\n')
    # 返回去重元素的重复数量
    u, indices = np.unique(a, return_counts=True)
    print(u, indices, '\n')
