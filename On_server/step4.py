# -*- coding: utf-8 -*-
"""Created on Sun Jan 04 23:24:47 2015@author: Saurabh"""

import sys
sys.path.append('c:\\anaconda\\lib\\site-packages') 
sys.path.append('c:\\Users\\Saurabh\\Anaconda\\Scripts')
import pandas as pd, numpy as np 
from scipy import interpolate
import sqlite3 as sql, pandas.io.sql as pd_sql
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


d1  =pd.read_csv("E:\\Data3\\app_rej2.csv" , encoding='utf8')

#dtype(d1.dti)

app_rej2 = d1.copy()
pd.unique(app_rej2.dti)

for i in xrange(len(app_rej2.dti)):
    app_rej2.dti[i]=float(app_rej2.dti[i])

app_rej2.dti.head()

app_rej2[['dti' , 'emp_length']] = app_rej2[['dti', 'emp_length']].astype(float)

for i in xrange(len(app_rej2.list_d)):
    dtt = app_rej2.list_d[i]
    if dtt[2] =='-':
        app_rej2.list_d[i] = datetime.strptime(app_rej2.list_d[i], '%d-%m-%Y')
    else:
        app_rej2.list_d[i] = datetime.strptime(app_rej2.list_d[i], '%Y-%m-%d')


app_rej2.list_d.head()

addr_state = pd.Series(app_rej2.addr_state.unique())
addr_state 

addr_state_1 = addr_state.reset_index(level=None, drop=True)
addr_state_2=addr_state_1.to_dict()
addr_state_dict = dict (zip(addr_state_2.values(),addr_state_2.keys()))


#app_rej2.to_csv('app_rej2.csv', sep=',',encoding='utf8')

app_rej2['addr_state_1']=0

for i in xrange(len(app_rej2.addr_state)):
    app_rej2.addr_state_1[i]=addr_state_dict.get(app_rej2.addr_state[i])
    app_rej2.addr_state_1[i] = float(app_rej2.addr_state_1[i])
    
print app_rej2.addr_state_1.head()    

#app_rej2.to_csv('app_rej3.csv', sep=',',encoding='utf8')


####%$$$@@ New 
basic_mer = app_rej2.copy()
basic_mer.addr_state_1.head()

#fill na, define features and target numpy arrays
#numerical_features = app_rej2.get(['loan_amnt', 'dti','emp_length','addr_state_1','list_d'])
numerical_features = app_rej2.get(['loan_amnt', 'dti','emp_length','addr_state_1'])
numerical_features_BKP = numerical_features.copy()

numerical_features = numerical_features_BKP.copy()
numerical_features['dti']=np.sqrt(numerical_features['dti'])
#numerical_features['loan_amnt']=sqrt(numerical_features['loan_amnt'])

features_array = numerical_features.fillna(numerical_features.dropna().median()).values
target = basic_mer.approved.values

features_train, features_test, target_train, target_test = train_test_split(features_array, target, test_size=0.30, random_state=0)

# Random Forest
#clf = tree.DecisionTreeClassifier()
#clf_pred = clf.fit(features_train, target_train)
#clf_pred = clf.predict(features_train, target_train)

#evaluate accuracy Random Forest
print("\n\nRandom Forest of Loan Dataset on Numerical Features\n\n")

print(classification_report(target_train, clf_pred,
                            target_names=['not Approved', 'Approved']))



##### Neural Network Model


### Gaussian Naive Bayes
gnb = GaussianNB()
gnb_pred = gnb.fit(features_train, target_train).predict(features_train)


#evaluate accuracy Naive Bayes
print("\n\nLogistic regression of Loan Dataset on Numerical Features\n\n")

print(classification_report(target_train, gnb_pred,
                            target_names=['not Approved', 'Approved']))

target_test_predicted = gnb.predict(features_test)
print(classification_report(target_test, target_test_predicted,
                            target_names=['not Approved', 'Approved']))


# generate evaluation metrics Naive Bayes
print "Accurcy" , metrics.accuracy_score(target_test, target_test_predicted)
print metrics.roc_auc_score(target_test, target_test_predicted)

print "Confusion matrix on Test"
print metrics.confusion_matrix(target_test,target_test_predicted),"\n"

print "Confusion matrix on Train"
print metrics.confusion_matrix(target_train,gnb_pred),"\n"
print "Accurcy on Train" , metrics.accuracy_score(target_train, gnb_pred)
print metrics.roc_auc_score(target_test, target_test_predicted)


### SVM
from sklearn import svm

clf_svm = svm.SVC()
clf_svm_pred = clf_svm.fit(features_train, target_train).predict(features_train)


#evaluate accuracy Naive Bayes
print("\n\nLogistic regression of Loan Dataset on Numerical Features\n\n")

print(classification_report(target_train, clf_svm_pred,
                            target_names=['not Approved', 'Approved']))



# train logistic regression, evaluate on test
lr = LogisticRegression(C=1)
lr.fit(features_train, target_train)
model2 = lr.fit(features_train, target_train)
target_predicted = lr.predict(features_test)


#evaluate accuracy
print("\n\nLogistic regression of Loan Dataset on Numerical Features\n\n")

print(classification_report(target_test, target_predicted,
                            target_names=['not Approved', 'Approved']))


# generate evaluation metrics
print "Accurcy" , metrics.accuracy_score(target_test, target_predicted)
print metrics.roc_auc_score(target_test, target_predicted)

print "Confusion matrix"
print metrics.confusion_matrix(target_test,target_predicted),"\n"


print "Hence the equation of Logistics Regression is as: "
print lr.coef_ , lr.intercept_

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(target_test, target_predicted)

from sklearn.metrics import log_loss
log_loss(target_test, target_predicted)

from sklearn.metrics import roc_auc_score
roc_auc_score(target_test, target_predicted)

from sklearn.metrics import explained_variance_score
explained_variance_score(target_test, target_predicted)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(target_test, target_predicted)

from sklearn.metrics import r2_score
target_predicted_train = lr.predict(features_train)
r2_score(target_train, target_predicted_train)
print metrics.confusion_matrix(target_train,target_predicted_train),"\n"
print "In Sample Accurcy" , metrics.accuracy_score(target_train, target_predicted_train)

from sklearn.metrics import mean_absolute_error
print "In Sample Absolute error" , mean_absolute_error(target_train, target_predicted_train)

print(classification_report(target_train, target_predicted_train,
                            target_names=['not Approved', 'Approved']))




