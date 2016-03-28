#Load the data from previous:
import sys
sys.path.append('c:\\anaconda\\lib\\site-packages') 
sys.path.append('c:\\Users\\Saurabh\\Anaconda\\Scripts')
##
import pandas as pd , matplotlib ,matplotlib.pyplot as plt
import numpy as np , scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
from scipy import interpolate
import sqlite3 as sql, pandas.io.sql as pd_sql
import statsmodels.api as sm
import re    
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import *
from sklearn. cross_validation import cross_val_score

# Conneting to  SQLITE DB
con = sql.connect('E:\\Data3\\data03.db', timeout=10) 
d1 = pd_sql.read_frame("select * from loan1", con)
r1 = pd_sql.read_frame("select * from reject1", con)
cur = con.cursor()
cur.close()                         ## Done with DB work CLOSE
con.close()


########## Correcting Data Types

import re    
d1.interest_rate = re.sub('%', '', str(d1.int_rate))

xxx = d1.int_rate
yyy = xxx
#yyy=[]
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

clf = linear_model.SGDRegressor()

## Just cheking whats inside this array.
for i, line in enumerate(features_array):
    print line
    if i >2:
        break 

scaler = StandardScaler()
X_train = scaler.fit_transform(features_train)  # compute mean, std and transform training data as well
X_test = scaler.transform(features_test)        # Transform test set


## Building the MODEL
clf.fit(features_train, target_train)

## evaluate model
print clf.coef_


## Predicting using the model
predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)


scores = cross_val_score(clf, X_train, target_train, cv=5)
print scores.mean(), scores

print 'Test set r-squared score', clf.score(X_test, target_test)

prediction_df = pd.DataFrame(predict_train, target_train)


############

from sklearn.cross_validation import *

def train_and_evaluate(clf, X_train, y_train):
    
    clf.fit(X_train, y_train)
    
    print "Coefficient of determination on training set:",clf.score(X_train, y_train)
    
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print "Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores)


clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None,  random_state=42)

train_and_evaluate(clf_sgd,X_train,target_train)




###############

from sklearn import svm
clf_svr= svm.SVR(kernel='linear')
train_and_evaluate(clf_svr,X_train,target_train)



# Function for Fitting our data to Linear model

features_train, features_test, target_train, target_test 

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(features_train, target_train)
predict_outcome = regr.predict(features_train)
predictions = {}
predictions['intercept'] = regr.intercept_
predictions['coefficient'] = regr.coef_
predictions['predicted_value'] = predict_outcome


result = linear_model_main(features_train, features_test, target_train)
print "Intercept value " `, result['intercept']
print "coefficient" , result['coefficient']
print "Predicted value: ",result['predicted_value']

print 'Test set R-squared score', regr.score(features_test, target_test)


## Predicting using the model
predict_train = regr.predict(features_train)
predict_test = regr.predict(features_test)

prediction_train_df = pd.DataFrame(predict_train, target_train)
prediction_train_df.to_csv('Prediction_on_Train.csv')

prediction_test_df = pd.DataFrame(predict_test, target_test)
prediction_test_df.to_csv('Prediction_on_Test.csv')

### FEATURE SELECTION
fs=SelectKBest(score_func=f_regression,k=5)
X_new=fs.fit_transform(features_train,target_train)
feature_dict =  zip(fs.get_support(),d1.columns)
##print feature_dict 
for i ,diction in enumerate(feature_dict) :
    if diction[0] == True:
        print "Important Feauture ", i , " \n", diction[1]




