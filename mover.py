# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:18:04 2019

@author: Cyrus
"""

import os
import shutil
import fnmatch

def gen_find(filepat,top):
    print("gen_finding")
    i = 0
    for path, dirlist, filelist in os.walk(top):
        i+=1
        print("outer")
        print(i)
        j = 0
        for name in fnmatch.filter(filelist,filepat):
            j+=1
            print("inner")
            print(j)
            yield os.path.join(path,name)

# Example use
def do():
    print("doing")
    src = './data/train/Vierecke' # input
    dst = './data/validation/Vierecke' # desired     location

    filesToMove = gen_find("*.png",src)
    for name in filesToMove:
        splitName = name.split(".png")[0].split('\\')[-1]
        print(splitName)
        numberAsString = splitName
        print(numberAsString)
        number = int(numberAsString)
        
        if number % 10 == 0:
            shutil.move(name, dst)

do()