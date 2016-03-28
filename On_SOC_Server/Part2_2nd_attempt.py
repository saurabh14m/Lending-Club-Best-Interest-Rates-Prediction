# -*- coding: utf-8 -*-
"""Created on Wed Feb 11 10:44:18 2015@author: Saurabh"""

import sys ##
import pandas as pd , matplotlib 
import numpy as np , scipy as sp
from scipy import stats
from scipy import interpolate
import sqlite3 as sql, pandas.io.sql as pd_sql
import statsmodels.api as sm
import re    
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import *
from sklearn. cross_validation import cross_val_score
import pickle

# Conneting to  SQLITE DB
con = sql.connect('/root/data_/data03.db', timeout=10) 
d1 = pd_sql.read_frame("select * from loan1", con)
r1 = pd_sql.read_frame("select * from reject1", con)
cur = con.cursor()
cur.close()                         ## Done with DB work CLOSE
con.close()


########## Correcting Data Types

import re    
d1.interest_rate = re.sub('%', '', str(d1.int_rate))

xxx = d1.int_rate
yyy = xxx #yyy=[]
for i in range(len(xxx)):
    if xxx[i] is None:
        xxx[i]='0'
    if xxx[i] == '':
        xxx[i]='0'
    xxx[i]=re.sub('%', '', str(xxx[i]))
    yyy[i]=float(xxx.values[i])
    d1.int_rate[i]=float(xxx.values[i])

#d1.int_rate = float(yyy)
d1.int_rate = d1.int_rate.astype(float)

## Selecting only float Data 
Interger_Float_columns = []
for i , cols in enumerate(d1.columns):
    if d1.dtypes[i] in ('float64' , 'int64'):
        print cols
        Interger_Float_columns.append(cols)

d1_bkp1 = d1.copy()        
d1 = d1[Interger_Float_columns]

## Drop NAs in the outcome variables
d1_bkp2 = d1.copy()
d1 = d1[np.isfinite(d1['int_rate'])]

d1.describe()
d1['loan_amnt'].describe()

Sparse_columns_can_be_dropped = ['mths_since_last_delinq' , 'mths_since_last_record' , 'mths_since_recent_revol_delinq' , 'mths_since_recent_bc_dlq' , 'mths_since_last_major_derog' , 'mths_since_recent_inq' , 'mo_sin_old_il_acct' , 'num_tl_120dpd_2m' ] 

## Remove Unnecessary columns
d1.drop(Sparse_columns_can_be_dropped, axis=1, inplace=True)

d2 = d1.dropna()
len(d2)



## Preparing Test and Train Data
target_a = 5
c = range(0,len(d1.columns))
c.remove(target_a)                  ## Place at which the target variable exist

numerical_features = d1.get(c)      ## Extract the dataframe to the list

features_array = numerical_features.fillna(numerical_features.dropna().median()).values

target = d1[d1.columns[target_a]].values

features_train, features_test, target_train, target_test = train_test_split(features_array, target, test_size=0.30, random_state=0)


#### LASSO

from sklearn.linear_model import Lasso, lasso_path, enet_path
features_train /= features_train.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
features_test /= features_test.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(features_train, target_train, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    features_train, features_train, eps, positive=True, fit_intercept=False)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    features_train, features_train, eps=eps, l1_ratio=0.8, fit_intercept=False)

clf = linear_model.Lasso(alpha = 0.1)
clf.fit(features_train,target_train)
clf.predict(features_test)

# See which coefficients are relevant in the model
clf.coef_

from sklearn.metrics import r2_score
# sklearn has an r2_score method. 
score = r2_score(target_test, clf.predict(features_test))
print score

# 
from sklearn import svm

clf_svr_rbf= svm.SVR(kernel='rbf')

clf_svr_rbf.fit(features_train,target_train)

clf_svr_rbf.kernel

svr_predictions = clf_svr_rbf.predict(features_test)

# sklearn has an r2_score method. 
score = r2_score(target_test, svr_predictions)
print score

pickle.dump(clf, open('my_model_clf.p', 'wb'))

########################################################
## Gradient Boosting Regressors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(features_train,target_train)

########### Feature Selection
##

from sklearn.feature_selection import RFE
from sklearn.svm import SVC

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(features_train,target_train)
print(model.feature_importances_)

Feature_importance = pd.DataFrame(model.feature_importances_ , d2.columns[c])

Feature_imp2 = Feature_importance[Feature_importance>0.02]


###########################
import statsmodels.api as sm

def reg_m(y, X):
    results = sm.OLS(y, X).fit()
    return results

print reg_m(features_train,target_train).summary()

[34, 220, 21.0, 31.0, 0.02279168915, 0.03376773732, 0.02132549584, 0, 0, 0, 0, 0, 0.02203499463, 0, 0, 0, 0, 0, 0.0201141505, 0, 0, 0.02112456782, 0, 0.02241847706, 0.02303996838, 0, 0.03066001366, 0.02965026877, 0.03137614547, 0.04540111651, 0, 0, 0, 0.02966477947, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.077564276]


