# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:11:57 2016

@author: pm001
"""
import pandas as pd
import numpy as np
from scipy.stats import skew

def skewness(arrays):
    return skew(arrays)
lst = [0, 2, 4, 10, 11, 12]
data_read = pd.read_excel("C:\\Users\\PM001\\Datasets\\adult_test.xlsx",sheetname =0 )
data_read = data_read.iloc[:,lst]
column_count = len(data_read.columns)
row_count = len(data_read.index)
skewed_values=[]
for i in range(column_count):
    skewed_cal = []
    for j in range((row_count)):
        skewed_cal.append(data_read.iloc[j,i])
    skewed_values.append(skewness(np.asanyarray(skewed_cal)))
    
#print skewed_values
print skewed_values

####################################sknewess_calculation_################################################    

assign_confidence_score = []
for i in range(len(skewed_values)):
    if(skewed_values[i] < -1 or skewed_values[i] > 1 ):
        assign_confidence_score.append(0)
    elif((skewed_values[i] < -0.5 and skewed_values[i] > -1 ) or (skewed_values[i] > 0.5 and skewed_values[i] < 1)):
        if(skewed_values[i] > 0):
            value_pos  = ((skewed_values[i] - 0.5) * ((100 -0 )/(1-0.5))) + 0
            assign_confidence_score.append(100-value_pos)
        elif(skewed_values[i] < 0):
            value_neg = (((skewed_values[i]) - (-1)) * ((100 - 0 )/((-0.5)-(-1)))) + 0
            assign_confidence_score.append(100-value_neg)
    elif(skewed_values < 0.5 or skewed_values > -0.5):
        assign_confidence_score.append(100)

print assign_confidence_score        

print np.mean(assign_confidence_score)
        
    