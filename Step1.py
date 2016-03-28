# -*- coding: utf-8 -*-
"""Created on Sun Dec 07 12:05:42 2014@author: Saurabh"""

#Assuming have run Step0 Successfully

correlation = d1.corr()
covariance = d1.cov()

covariance.to_csv('G:\\DecisionSc\\Outputs\\covariance.csv')
correlation.to_csv('G:\\DecisionSc\\Outputs\\correlation.csv')

