# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:30:10 2023

@author: Yu Zhenyu

If you have a dream ...
"""
import numpy as np
import pandas as pd
import os
import glob
import PIL

file_num = 12
# path = r"D:\Data\2transferLearning\WangPei\Class-Aware\log\Class_Aware\Office31\\"
path = r"D:\Data\2transferLearning\WangPei\Class-Aware\log\Class_Aware\home\\"
path_out = r""

txts = glob.glob(path + "*.txt")
txts.sort()
txts2 = txts[-file_num:]

result = pd.DataFrame(columns=["dataset","best_acc"])
for t in txts2:
    # name = t[91:-14] # Office31
    name = t[87:-14] # home
    df = pd.read_table(t)
    acc = df.iloc[-1].values[0].split(" ")[-1]
    result = result.append({"dataset":name, "best_acc":acc}, ignore_index=True)


print(result)
# print(result.dataset)
# print(result.best_acc)

print("Finished!!!\a")
