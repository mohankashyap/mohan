#from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.svm import SVC
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
start_time = datetime.now()
label_triple = []
################################single random permutation checking#################################################################
df = pd.read_excel('/home/mohan/Downloads/Theano-Tutorials-master/vnsp_write_features.xlsx',sheetname=1)
#data = df.iloc[:,1:3]
train_test = df.iloc[:,[1,2,3,4,5,6]]
for j in range(np.shape(train_test)[1]):
    for i in range(np.shape(train_test)[0]):
        string = train_test.iloc[i,j]
        if(type(string) == float):
            if(str(float(string)).lower() == 'nan'):             
                train_test.iloc[i,j] = df.iloc[i,1]
data = np.asarray(train_test)
#data = np.random.permutation(data)
train_data,test_data = train_test_split(data , test_size=0.2, random_state=42)
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
           random_state=12345)
print "length of cv"
print cv
print '#########################test_data########################################################'

print '##########################################################################################'

for i in range(len(train_data)):
    if(train_data[i,1]=='VNSP'):
        train_data[i,1] = 1
        
    else:
        train_data[i,1] = 0
        
label_train = train_data[:,1] 

for i in range(len(test_data)):
    if(test_data[i,1]=='VNSP'):
        test_data[i,1] = 1
        
    else:
        test_data[i,1] = 0
        
label_test = test_data[:,1]
vctr =  CountVectorizer(stop_words='english',min_df = 1)
vctr2 = HashingVectorizer(stop_words='english')
vctr1 = TfidfVectorizer(stop_words='english')
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
clf1 =   GradientBoostingClassifier(n_estimators = 1000)
clf2 =   AdaBoostClassifier(n_estimators = 1000)
clf3 =   RandomForestClassifier()
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
val_class[val1]=idx1
val_class[val2]=idx2
val_class[val3]=idx3
print "the minimum value of class and its index is"
for val in val_class:
    if(val == max(val_class.keys())):
       k = val
       print k,val_class[k]
print "the global minimum error"
print k
print "the global minimum index"
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

train_data2 = vctr.fit_transform(train_check).toarray()
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
X_test1 = vctr.transform(test).toarray()
grad_label = clf1.predict(X_test1)
grad_label = grad_label.astype(int)
#print "grad_label"
#print grad_label
##########################################################################################
clf2.fit(train_data2,y_train1)
X_test2 = vctr.transform(test).toarray()
ada_label = clf2.predict(X_test2)
ada_label = ada_label.astype(int)
#print "ada_label"
#print ada_label
###########################################################################################

clf3.fit(train_data2,y_train1)
X_test3 = vctr.transform(test).toarray()
rand_label = clf3.predict(X_test3)
rand_label = rand_label.astype(int)
labelgot = []
for i in range(len(rand_label)):
    avg = float(float(grad_label[i]+ada_label[i]+grad_label[i])/(float(3)))
    labelgot.append(avg)
print "labels averaged"
print labelgot
for i in range(len(labelgot)):
    if(labelgot[i] == 0.3333333333333333):
       labelgot[i] = 0.0
    elif(labelgot[i] == 0.6666666666666666):
       labelgot[i] = 1.0
    elif(labelgot[i] == 1):
       labelgot[i] = 1.0
    elif(labelgot[i] == 0):
       labelgot[i] = 0.0
print "############################################################ developed label set ##################################################"
print "developed label set type "

print labelgot

#print "rand_label"
#print rand_label


#required_labels = increasing_measure(grad_label,ada_label,rand_label)
#print "required meta voting labels"
print "labelgot real format"
print labelgot
required_labels = np.asarray(labelgot,dtype=np.int8)
print "converting into np.int"
print required_labels
required_labels = np.asarray(required_labels,dtype="|S6")
print "array format of labels"
print required_labels
#print "array type labels"
#print required_labels
#print "int type required labels"
required_labels = required_labels.astype(int)
test_data_labels = label_test.astype(int)
##########################################################################################################################################
print "the obtained labels"
print required_labels
print "the accuracy is"
print np.mean(required_labels == label_test)
print "the F1_Score is"
print f1_score(test_data_labels,required_labels)
print "accuracy"
print np.mean(test_data_labels == required_labels)
abs1 = np.mean(test_data_labels == required_labels)
print "mean absoulute error"
print abs(1-abs1)
#acc.append(np.mean(test_data_labels == p_labels))
fn = 0
tn = 0
tp = 0
fp = 0
for i in range(len(test_data_labels)):
    if(test_data_labels[i] == 1 and required_labels[i] == 1):
        tp = tp + 1
    elif(test_data_labels[i] == 0 and required_labels[i] == 1):
        fp = fp + 1   
    elif(test_data_labels[i] == 0 and required_labels[i] == 0):
        tn = tn + 1        
    elif(test_data_labels[i] == 1 and required_labels[i] == 0):
        fn = fn + 1
cm = confusion_matrix(test_data_labels, required_labels)


##################################################################################################
#X_2d = bh_sne(X)
#print scatter(X_2d[:, 0], X_2d[:, 1], c=y)
#cm = confusion_matrix(test_data_labels, p_labels)

#print (cm)
#print "number of positives"
#print count_pos
#print "number of negatives"
#print count_neg

#true_pos_rate = float(float(tp)/float(count_pos))
#false_pos_rate = float(float(fp)/float(count_neg))

#print "true_pos_rate"
#print true_pos_rate
#print "true_neg_rate"
#print false_pos_rate




##############################plotting of confusion matrix#####################################################################
plt.matshow(cm);
plt.title('Confusion matrix');
plt.colorbar();
plt.ylabel('True label');
plt.xlabel('Predicted label');

pylab.show();
print "roc score"
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(test_data_labels, required_labels)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#print roc_auc_score(test_data_labels,  p_labels)

print "tp" 
print tp
print "fp"
print fp
print "tn"
print tn
print "fn"
print fn

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


################################################################################################

# ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']#
#########################################################################################################################################

#####



