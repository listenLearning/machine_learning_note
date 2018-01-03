#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' ' * ind, self.name, '   ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)
