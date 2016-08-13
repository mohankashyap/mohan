from sklearn.naive_bayes import MultinomialNB
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
from sklearn.cross_validation import train_test_split
label_triple = []


#def increasing_measure(grad_label,ada_label,rand_label):
 #   for i in range(len(grad_label)):
              
    

























################################single random permutation checking#################################################################
df = pd.read_excel('/home/mohan/Downloads/Theano-Tutorials-master/newbackup.xlsx',sheetname=1)

data = df.iloc[:,1:3]
#print "length of data"
#print len(data)
data = np.asarray(data)
#print "type of train_data"
#print type(train_data)
#print train_data
data = np.random.permutation(data)
train_data,test_data = train_test_split(data , test_size=0.3, random_state=42)
#print "train data"
#print len(train_data[:,0])
#print train_data[:,1]
#print '##########################################################################################'

print '#########################test_data########################################################'

print '##########################################################################################'

#print "test_data"
#print len(test_data[:,0])
#print test_data[:,1]
#test_data = df1.iloc[:,1:3]
#test_data = np.asarray(test_data)
#test_data = np.random.permutation(test_data)
#print "test_data"
#print test_data
for i in range(len(train_data)):
    if(train_data[i,1]=='JOB'):
        train_data[i,1] = 1
        
    else:
        train_data[i,1] = 0
        
label_train = train_data[:,1] 
#print "labels"
#print len(label_train)
#print label_train

for i in range(len(test_data)):
    if(test_data[i,1]=='JOB'):
        test_data[i,1] = 1
        
    else:
        test_data[i,1] = 0
        
label_test = test_data[:,1]
#print "lenght label_test"
#print len(label_test)
#print label_test
#data = df.iloc[:,1:3]
#data = np.asarray(data)
#print len(data)
#print np.random.shuffle(data)
#train_data = df.iloc[0:527,1]
#print "traindata"
#print len(train_data)
#test_data = df.iloc[527:753,1]
#print "testdata"
#print len(test_data)
#test_data = df1.iloc[:,1]
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
    #print string,i
    string = vctr.build_preprocessor()(string.lower()) 
    string = vctr.build_tokenizer()(string.lower())
    train.append(' '.join(string))

for i in range(len(test_data)):
    string = test_data[i,0]
    string = vctr.build_preprocessor()(string.lower()) 
    string = vctr.build_tokenizer()(string.lower())
    test.append(' '.join(string)) 
#print "len of the normalized test data obtained"    
#print len(test)  
######################################################################################################
train_data = vctr.fit_transform(train).toarray()
#print vctr1.inverse_transform(train_data)
y_train = np.asarray(label_train, dtype="|S6")
clf1 =   GradientBoostingClassifier(n_estimators = 660)
clf2 =   AdaBoostClassifier(n_estimators = 660)
clf3 =   RandomForestClassifier()
test_data_labels = np.asarray(label_test,dtype="|S6")
test_data_labels = test_data_labels.astype(int)
print "test_data_labels"
print test_data_labels
##################################################################################################
clf1.fit(train_data,y_train)
X_test = vctr.transform(test).toarray()
grad_label = clf1.predict(X_test)
grad_label = grad_label.astype(int)
#print "grad_label"
#print grad_label
##########################################################################################
clf2.fit(train_data,y_train)
X_test = vctr.transform(test).toarray()
ada_label = clf2.predict(X_test)
ada_label = ada_label.astype(int)
#print "ada_label"
#print ada_label
###########################################################################################

clf3.fit(train_data,y_train)
X_test = vctr.transform(test).toarray()
rand_label = clf3.predict(X_test)
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


print "f1_score"
print f1_score(test_data_labels,required_labels)
#print "real_accuracy"
#print clf.score(X_test,required_labels)
print "accuracy"
print np.mean(test_data_labels == required_labels)
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

