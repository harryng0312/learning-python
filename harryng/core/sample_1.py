#!/usr/bin/python

class MySupClass:
    __hidden_var = None

    def accessHiddenVar(self, x):
        self.__hidden_var = x


class MyClass(MySupClass):
    def __init__(self):
        super.__init__(0)
