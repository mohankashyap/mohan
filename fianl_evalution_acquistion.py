from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn import metrics
import pandas as pd
import numpy as np
#from sklearn.svm import SVC
from compiler.ast import flatten
from openpyxl import Workbook, load_workbook
from scipy.stats.stats import pearsonr   
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.chunk.regexp import RegexpParser
from nltk.tag.util import *
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import os
import re
#from tsne import bh_sne
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import math as m
from sklearn.preprocessing import normalize
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.cross_validation import train_test_split
from datetime import datetime
import pickle
start_time = datetime.now()
label_triple = []
################################single random permutation checking#################################################################
df = pd.read_excel('/home/mohan/mtech/data_new/acq_up.xlsx',sheetname=0)
#data = df.iloc[:,1:3]
train_test = df.iloc[:,[1,2]]
for j in range(np.shape(train_test)[1]):
    for i in range(np.shape(train_test)[0]):
        string = train_test.iloc[i,j]
        if(type(string) == float):
            if(str(float(string)).lower() == 'nan'):             
                train_test.iloc[i,j] = df.iloc[i,1]
data = np.asarray(train_test)
#data = np.random.permutation(data)
train_data,test_data = train_test_split(data , test_size=0.0, random_state=42)
train_check = train_data
print train_check[0,0]
print "type of original data"
print type(train_data)
print "original train_data"
print train_data[0,0]
print "original label"
print train_data[0,1]
cv = KFold(n=train_data.shape[0],  # total number of samples
           n_folds=5,           # number of folds the dataset is divided into
           random_state=12345,shuffle=True)
print "length of cv"
print cv
print '#########################test_data########################################################'

print '##########################################################################################'
count_neg_train = 0
count_pos_train = 0
count_pos_test = 0
count_neg_test = 0
for i in range(len(train_data)):
    if(train_data[i,1]=='ACQ'):
        train_data[i,1] = 1
        count_pos_train = count_pos_train + 1
    else:
        train_data[i,1] = 0
        count_neg_train = count_neg_train + 1
        
label_train = train_data[:,1] 

for i in range(len(test_data)):
    if(test_data[i,1]=='ACQ'):
        test_data[i,1] = 1
        count_pos_test = count_pos_test + 1 
        
    else:
        test_data[i,1] = 0
        count_pos_test = count_neg_test + 1
        
label_test = test_data[:,1]
#vctr =  CountVectorizer(stop_words='english',min_df = 1)
vctr2 = HashingVectorizer(stop_words='english')
vctr = TfidfVectorizer(stop_words='english')
count_pos = 0
count_neg = 0

######################################################################################################
train = []
test = []
for i in range(len(train_data)):
    string = train_data[i,0]
    string = vctr.build_preprocessor()(string.lower()) 
    string = vctr.build_tokenizer()(string.lower())
    train.append(' '.join(string))

for i in range(len(test_data)):
    string = test_data[i,0]
    string = vctr.build_preprocessor()(string.lower()) 
    string = vctr.build_tokenizer()(string.lower())
    test.append(' '.join(string)) 

######################################################################################################
train_data1 = vctr.fit_transform(train).toarray()
#X_test = vctr.transform(test).toarray()
y_train = np.asarray(label_train, dtype="|S6")
y_train = y_train.astype(int)
clf1 =   GradientBoostingClassifier(n_estimators = 100)
clf2 =   AdaBoostClassifier(n_estimators = 100)
clf3 =   RandomForestClassifier(n_estimators = 100)
print "type of train_data"
print type(train_data1)
print "type of y_train"
print type(y_train)
print "type of cv"
print type(cv)
scores1 = cross_val_score(clf1, train_data1 , y_train, cv=cv, scoring='mean_absolute_error')
scores2 = cross_val_score(clf2, train_data1 , y_train, cv=cv, scoring='mean_absolute_error')
scores3 = cross_val_score(clf3, train_data1 , y_train, cv=cv, scoring='mean_absolute_error')
scores4 = cross_val_score(clf1, train_data1 , y_train, cv=cv, scoring='f1')
scores5 = cross_val_score(clf2, train_data1 , y_train, cv=cv, scoring='f1')
scores6 = cross_val_score(clf3, train_data1 , y_train, cv=cv, scoring='f1')
score1=[]
score2=[]
score3=[]
score4=[]
score5=[]
score6=[]
val_3classifiers = {}
for score in scores1:
    print "score"
    print score
    score1.append(score)
for score in scores4:
    print "score"
    print score
    score4.append(score)
print "mean error for classifier 1 is"
print  np.mean(score1)
print "minimum error for classifier 1 is"
print  max(score1)
val1, idx1 = max((val1, idx1) for (idx1, val1) in enumerate(score1))
print "minimum error and index for classifier1"
print val1
print idx1
###################################################################################################################################
print "mean f1_score of classifier1"
print np.mean(score4)
print "max f1 and its index for classifier1"
valf1, idxf1 = max((valf1, idxf1) for (idxf1, valf1) in enumerate(score4))
print valf1
print idxf1
########################################################################################################################################
for score in scores2:
    print "score"
    print score
    score2.append(score)
for score in scores5:
    print "score"
    print score
    score5.append(score)
print "mean error for classifier 2 is"
print  np.mean(score2)
print "minimum error for classifier 2 is"
print  max(score2)
val2, idx2 = max((val2, idx2) for (idx2, val2) in enumerate(score2))
print "minimum error and index for classifier2"
print val2
print idx2
###################################################################################################################################
print "mean f1_score of classifier2"
print np.mean(score5)
print "max f2 and its index for classifier2"
valf2, idxf2 = max((valf2, idxf2) for (idxf2, valf2) in enumerate(score5))
print valf2
print idxf2
###########################################################################################################################################    
for score in scores3:
    print "score"
    print score
    score3.append(score)
for score in scores6:
    print "score"
    print score
    score6.append(score)
print "mean error for classifier 3 is"
print  np.mean(score3)
print "minimum error for classifier 3 is"
print  max(score3)
val3, idx3 = max((val3, idx3) for (idx3, val3) in enumerate(score3))
print "minimum error and index for classifier3"
print val3
print idx3
###################################################################################################################################
print "mean f1_score of classifier3"
print np.mean(score6)
print "max f3 and its index for classifier3"
valf3, idxf3 = max((valf3, idxf3) for (idxf3, valf3) in enumerate(score6))
print valf3
print idxf3


###########################################################################################################################################
val_class = {}
val_class[valf1]=idxf1
val_class[valf2]=idxf2
val_class[valf3]=idxf3
print "the minimum value of class and its index is"
for val in val_class:
    if(val == max(val_class.keys())):
       k = val
       print k,val_class[k]
#print "the global minimum error"
print "the best f1"
print k
#print "the global minimum index"
print "best f1 index"
print val_class[k]
required_train_data = []
required_test_data = []
count = 0
for train, test1 in cv:
    #print "normal indices are"
    #print train
    #print test
    if(count == val_class[k]):
       #print "a"
       #print "required indices"
       #print("%s %s" % (train, test))
       #print "required data set"
       #print "length of the train data and its data set"
       #print train_check[train,0]
       #print "label of data is"
       #print train_check[train,1] 
       required_train_data.append(train)
       required_train_data.append(test1)
    count = count + 1
#print " the count value is"
#print count
#print "the concate"
required_train_data = np.concatenate(required_train_data)
#print "length of requored_train_data "
#print len(required_train_data)
#print required_train_data[0]
train_data[required_train_data,0]
label_train1 = []
train_check=[]
for i in range(len(required_train_data)):
    #print "a"
    string = train_data[required_train_data[i],0]
    #print string
    label_train1.append(train_data[required_train_data[i],1])
    string = vctr.build_preprocessor()(string.lower()) 
    string = vctr.build_tokenizer()(string.lower())
    train_check.append(' '.join(string))
print "train_check of data change"
print train_check[0]
train_data2 = vctr.fit_transform(train_check).toarray()
###########################################################################################################################################
input_data_matrix = open('/home/mohan/mtech/New_pickle_files/fitted_matrix_ACQ.pkl', 'wb')
pickle.dump(vctr, input_data_matrix)
input_data_matrix.close()

#print "X_test"
#print X_test
y_train1 = np.asarray(label_train1, dtype="|S6")
y_train1 = y_train1.astype(int)
test_data_labels=y_train1
#print "real labels"
#print y_train1 
#clf1 =   GradientBoostingClassifier(n_estimators = 2000)
#clf2 =   AdaBoostClassifier(n_estimators = 2000)
#clf3 =   RandomForestClassifier()
print "the test data is"
print test[1:3]
###########################################################################################################################################
clf1.fit(train_data2,y_train1)
output1 = open('/home/mohan/mtech/New_pickle_files/clf1_ACQ.pkl', 'wb')
pickle.dump(clf1, output1)
output1.close()

#print "grad_label"
#print grad_label
##########################################################################################
clf2.fit(train_data2,y_train1)
output2 = open('/home/mohan/mtech/New_pickle_files/clf2_ACQ.pkl', 'wb')
pickle.dump(clf2, output2)
output2.close()
#print "ada_label"
#print ada_label
###########################################################################################

clf3.fit(train_data2,y_train1)
output3 = open('/home/mohan/mtech/New_pickle_files/clf3_ACQ.pkl', 'wb')
pickle.dump(clf3, output3)
output3.close()
###############################################################################################
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


