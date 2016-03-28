# -*- coding: utf-8 -*-
"""Created on Sat Dec 13 14:24:14 2014 @author: Saurabh"""

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

# Conneting to  SQLITE DB
con = sql.connect('data03.db', timeout=10) 
d1 = pd_sql.read_frame("select * from loan1", con)
r1 = pd_sql.read_frame("select * from reject1", con)
cur = con.cursor()
cur.close()## Done with DB work CLOSE
con.close()

#d1.columns
r1.columns

## Subsetting a DataFrame
d11 = d1.ix[:, (2,17,23,32,25,26,11)]
r11 = r1.ix[:,(0,1,2,4,5,6,7)]

r11.columns = d11.columns
r11.head()
d11.head()

for i in range(len(r11.dti)):
    r11.dti[i] = r11.dti[i].rstrip('%')

r11.dti = r11.dti.strip('%')
#r11.dti = re.sub('% ', '', str(r11.dti[i]))

## Created the outcome varaiable 
r11['approved']=0
d11['approved']=1


## Merge the two files
basic_mer = r11.append(d11, ignore_index=True)

for i in range(len(basic_mer)):
    basic_mer.addr_city[i] = str(basic_mer.addr_city[i]).lower().replace(' ','')
    basic_mer.addr_state[i]= str(basic_mer.addr_state[i]).lower().replace(' ','')



## Replace nulls with zeros
basic_mer.= NaN


##Write this to a new DB -- One time Thing
#con = sql.connect('db04_app_rej.db', timeout=10) 
#cur = con.cursor() #Declaring Cursor
#pd_sql.write_frame(basic_mer, "app_rej", con) #Writting to Database
#rej = pd_sql.read_frame("select count(*) from app_rej", con) ## Read SQL lite
#cur.close() ## Done with DB work CLOSE
#con.close()





