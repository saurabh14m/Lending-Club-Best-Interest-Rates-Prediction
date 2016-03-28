# -*- coding: utf-8 -*-
"""Created on Sun Dec 14 21:00:01 2014@author: Saurabh"""
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
import random

##Write this to a new DB -- One time Thing
con = sql.connect('db04_app_rej.db', timeout=10) 
cur = con.cursor() #Declaring Cursor
app_rej = pd_sql.read_frame("select * from app_rej", con) ## Read SQL lite
cur.close() ## Done with DB work CLOSE
con.close()

## Removing Missing
app_rej2 = app_rej.dropna()

## Checking -- how many 
del1 = app_rej2[app_rej2.approved==1]
app_rej2['approved_num']=0

del2 = app_rej2[app_rej2.approved==0]
ind = random.sample(xrange(len(del2)),  len(del1)) 

del3 = del2.ix[ind]

basic_mer = del3.append(del1, ignore_index=True)
basic_mer['approved_num'] = None
basic_mer = basic_mer.ix[:,(0,1,2,4,5,6,7,8)]

basic_mer.dti = float(basic_mer.dti)

##m THe code below doesnt work
basic_mer.dti[basic_mer.dti=='nan'] = 0
len(unique(basic_mer.dti))

## downloaded it into excel and find the row number mannuallly
basic_mer.dti[39725]='0.0'
basic_mer.dti[105009]='0.0'
basic_mer.dti[115356]='0.0'
basic_mer.dti[190387]='0.0'
basic_mer.dti[200017]='0.0'
basic_mer.dti[236428]='0.0'

basic_mer.dti[basic_mer.dti=='-1'] = 0

xdti=[]
for i in xrange(len(basic_mer.dti)):
    basic_mer.dti[i] = float(basic_mer.dti[i])
    
basic_mer.dti = float(basic_mer.dti)



emp_dict = {}
emp_dict = {'1 year' : 1, 
           '10+ years' : 10 ,
           '2 years' : 2,
           '3 years':3 , 
            '4 years':4,
            '5 years':5, 
            '6 years':6, 
            '7 years':7, 
            '8 years':8, 
            '9 years':9,
            '< 1 year':0, 
            'n/a':11}

for i in xrange(len(basic_mer.emp_length)):
    basic_mer.emp_length[i]=emp_dict.get(basic_mer.emp_length[i])

basic_mer.emp_length[str(basic_mer.emp_length) == None]=11
basic_mer.emp_length[str(basic_mer.emp_length) == '']=11

for i in xrange(len(basic_mer.emp_length)):
    basic_mer.emp_length[i]=int(basic_mer.emp_length[i])

basic_mer.to_csv('saurabh.csv', sep=',',encoding='utf8')

#fill na, define features and target numpy arrays
numerical_features = basic_mer.get(['loan_amnt', 'dti','emp_length'])
numerical_features.emp_length = int(numerical_features.emp_length)
T2 = [map(int, x) for x in numerical_features.emp_length]


features_array = numerical_features.fillna(numerical_features.dropna().median()).values
target = basic_mer.approved.values

features_train, features_test, target_train, target_test = train_test_split(features_array, target, test_size=0.40, random_state=0)

# train logistic regression, evaluate on test
lr = LogisticRegression(C=1)
lr.fit(features_train, target_train)
target_predicted = lr.predict(features_test)


#evaluate accuracy
print("\n\nLogistic regression of Titanic Dataset on Numerical Features\n\n")

print(classification_report(target_test, target_predicted,
                            target_names=['not Approved', 'Approved']))


##Data Final Exploration
#app_rej.groupby('approved').mean()

# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('approved ~ loan_amnt + dti + C(list_d)+ C(purpose) + C(addr_city) + C(addr_state) + C(emp_length)',
                  app_rej, return_type="dataframe")

print X.columns

main.data=app_rej.approved
main.target=
#### 
SPLIT_PERC = 0.25
split_size = int(len(main.data)*SPLIT_PERC) 
X_train = main.data[:split_size] 
X_test = main.data[split_size:]  
y_train = main.target[:split_size] 
y_test = main.target[split_size:]


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)




###### Chutiyapa

count=0
for i in unique(app_rej2.emp_length):
    currentid = str(i)
    key = str(count)
    mydict[key].append(currentid)
    count = count+1

rand_stocks = np.random.randint(0, len(app_rej), size=1000000) 
stock = app_rej.ix[rand_stocks]
stock.to_csv('Sample_100K.csv',encoding='utf8')


stock.head()
len(stock)
app_rej = stock.copy()



correlation = basic_mer.corr()

correlation.to_csv("correlatatiom.csv", sep=',')

basic_mer.dti=basic_mer.dti.values()

int(basic_mer.dti)
unique(basic_mer.dti)

basic_mer.dti[basic_mer.dti=='nan']=0


