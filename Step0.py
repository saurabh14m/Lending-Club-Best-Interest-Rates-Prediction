# -*- coding: utf-8 -*-
"""Created on Fri Nov 28 12:08:29 2014 @author: Saurabh """
import sys
sys.path.append('c:\\anaconda\\lib\\site-packages') 
sys.path.append('c:\\Users\\Saurabh\\Anaconda\\Scripts')
##
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

#Running Scipy
from scipy import stats
import scipy as sp
##
d1  =pd.read_csv("F:\\DecisionSc\\LoanStats3a.csv" , encoding='utf8')
d2  =pd.read_csv("F:\\DecisionSc\\LoanStats3a-2.csv", encoding='utf8')
d3  =pd.read_csv("F:\\DecisionSc\\LoanStats3b.csv", encoding='utf8')
d4  =pd.read_csv("F:\\DecisionSc\\LoanStats3c.csv", encoding='utf8')

file_merged = pd.merge(d3, d2, how='inner', on="svc_no", copy=True)

dx  =pd.read_csv("F:\\DecisionSc\\MyData.csv" , encoding='utf8')

### Merged to a Single file
d13 = d1.append(d3)
d134 = d13.append(d4)

## Initial Exploration
d134.describe()
d134.describe().transpose()
d1=d134.copy() ## Just Copying merged file to d1 dataframe which is use later on



## Identifying Numeric , Float and Character variables
d1_Numeric = d1.loc[:, d1.dtypes == np.float64]
d1_object =  d1.loc[:, d1.dtypes != np.float64]

## Doing Basic statistics functions
n, min_max, mean, var, skew, kurt = stats.describe(d1_Numeric)
d1_Numeric.describe()

#We can Run the basic exploaration functions
sp.stats.describe(d1_Numeric.values, axis=0)[source]

####
##STage 2  --->  Creating a DB
####

import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
from scipy import interpolate
import sqlite3 as sql
import pandas.io.sql as pd_sql
import statsmodels.api as sm

# Conneting to  SQLITE DB
con = sql.connect('F:\\DecisionSc\\data03.db', timeout=10) 

## Read SQL lite
d1 = pd_sql.read_frame("select * from loan1", con)
#%timeit -n100 -r10 pd_sql.read_frame("select * from tbl", con) #Read SQLITE DB to DF

#Declaring Cursor
cur = con.cursor()

# Droping table
cur.execute("drop table loan1;")
cur.execute("select * from loan1")

#Writting to Database
pd_sql.write_frame(d1, "loan1", con)

con.text_factory = str

## Done with DB work CLOSE
cur.close()
con.close()

## Automatic way of writing Dataframe to SQLlite
d1.to_sql('tbl1', con, flavor='sqlite', schema=None, if_exists='fail', index=True)

# Conneting to postgress
import sys
sys.path.append('c:\\anaconda\\lib\\site-packages')
import psycopg2
try:
    conn = psycopg2.connect(database="postgres", user="postgres", password="Password", host="127.0.0.1", port="1024")
except:
    print "I am unable to connect to the database"

# Defined Cursor for postgres
cur = conn.cursor()
pd_sql.write_frame(d1, "loan1", conn)
cur.execute(";")

### Method to read a table into dataframe of pandas :
#df2 = pandas.read_sql("SELECT * from mio_tv_service_level", conn)
#df2.head() ## Create the main Table
#cur.execute("DROP TABLE IF EXISTS loan_data")

#### Creating the DB for Rejection Loan Files

r1  =pd.read_csv("G:\\DecisionSc\\RejectStatsA.csv" , encoding='utf8')
r2  =pd.read_csv("G:\\DecisionSc\\RejectStatsB.csv", encoding='utf8')

r12 = r1.append(r2)
len(r12)==len(r1)+len(r2)
# Conneting to  SQLITE DB and writting out Reject merged file
con = sql.connect('data03.db', timeout=10) 
cur = con.cursor() #Declaring Cursor
pd_sql.write_frame(r12, "reject1", con) #Writting to Database
rej = pd_sql.read_frame("select count(*) from reject1", con) ## Read SQL lite
cur.close() ## Done with DB work CLOSE
con.close()

##Save the merged file to a physical CSV file
r12.to_csv('Reject._merged.csv',encoding='utf8')

# Renaming the coulms to what we have in the Loan DB
r12.columns= ['loan_amnt','list_d','title','Risk_Score','dti','addr_city','addr_state','emp_length','Policy Code']


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

d1.int_rate = float(yyy)
d1.int_rate = d1.int_rate.astype(float)

