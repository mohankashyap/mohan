# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:45:42 2016

@author: pm001
"""
import pandas as pd
import numpy as np
from collections import Counter

assign_confidence_score_outlier = []
outlier_indices = []    
val = []    

def is_outlier(value, p25, p75):
    """Check if value is an outlier
    """
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return value <= lower or value >= upper
 
 
def get_indices_of_outliers(values):
    """Get outlier indices (if any)
    """
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
     
    indices_of_outliers = []
    count_ind = 0
    for ind, value in enumerate(values):
        if is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)
            count_ind = count_ind + 1
    return indices_of_outliers

def cal_index_values(list_of_col,count):
    for i in range(len(list_of_col)):
        if(list_of_col[i] > 0.85 * count):
            return 0;
            break;
        else:
            break;
        
    
    



#[0, 2, 4, 10, 11, 12]

outlier_cols = []
data_read = pd.read_excel("C:\\Users\\PM001\\Datasets\\adult_test.xlsx",sheetname =0 )
lst = [0, 2, 4, 10, 11, 12]
data_read = data_read.iloc[:,lst]
column_count = len(data_read.columns)
row_count = len(data_read.index)






for i in range(column_count):
    for j in range(len(data_read)):
        outlier_cols.append(data_read.iloc[j,i])
    values = Counter(outlier_cols).values()
    cal_val = cal_index_values(values,row_count)
    if(cal_val == 0):
        print i
        assign_confidence_score_outlier.append(cal_val)
    else:
        val.append(len(get_indices_of_outliers(np.asanyarray(outlier_cols))))    
        assign_confidence_score_outlier.append(100 - (100 * float(float((len(get_indices_of_outliers(np.asanyarray(outlier_cols)))))/float(row_count))))
    #outlier_indices.append(indices_of_outliers)
#np.asarray(get_indices_of_outliers    
#print val  
print assign_confidence_score_outlier









