# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:40:01 2016

@author: pm001
"""
import pandas as pd
from collections import Counter
import math as m


def cal_index_values(list_of_col,count,row_count):
    absolute_percentages = []
    relative_percentages = []
    for i in range(len(list_of_col)):
        absolute_percentages.append(float((float(1)/float(count))*100))
        relative_percentages.append(float((float(list_of_col[i])/float(row_count)*100)))
    return absolute_percentages,relative_percentages    
        
        
data_read = pd.read_excel("C:\\Users\\PM001\\Datasets\\adult_test.xlsx",sheetname =0 )
score = []
#lst = [1, 3, 5, 6, 7, 8, 9, 13, 14]
lst = [1]
data_read = data_read.iloc[:,lst]
column_count = len(data_read.columns)
row_count = len(data_read.index)
outlier_cols = []
for i in range(column_count):
    score = []
    for j in range(len(data_read)):
        outlier_cols.append(data_read.iloc[j,i])
    values_relative = Counter(outlier_cols).values()
    print "The unique values are : " + str(values_relative)
    values_absolute = len(values_relative)
    print "The number of unique values are:"+ str(values_absolute)
    cal_absolute,cal_relative = cal_index_values(values_relative,values_absolute,row_count)
    print cal_absolute
    print cal_relative
    calculation = [abs(cal_absolute - cal_relative) for cal_absolute, cal_relative in zip(cal_absolute, cal_relative)]
    print calculation 
    score.append(sum(calculation))
print score    