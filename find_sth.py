# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:09:16 2023

@author: Yu Zhenyu

If you have a dream ...
"""
import numpy as np
import pandas as pd
import os
import glob
import PIL

path = r"D:\Data\2transferLearning\WangPei\Class-Aware\log\Class_Aware\Office31\新建文件夹\\"
path_out = r""

tag1 = "lr = 0.01"
tag2 = "entropy_tradeoff = 0.2"
tag3 = "feature_normal = True"

files = glob.glob(path + "*.txt")
for f in files:
    if "dslr-amazon" not in f:
        continue
    count = 0
    df = pd.read_table(f)
    for i in range(len(df)):
        tmp = df.iloc[i].values[0]
        if tag1 in tmp or tag2 in tmp or tag3 in tmp:
            count = count + 1
    if count >= 3:
        print(f.split("\\")[-1])
    
    
print("Finished!!!\a")
