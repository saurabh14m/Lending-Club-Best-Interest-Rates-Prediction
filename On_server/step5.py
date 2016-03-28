# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 01:57:11 2015

@author: Saurabh
"""

%matplotlib inline
import pymc as pm
import numpy as np


ndims = 2
nobs = 20

xtrue = np.random.normal(scale=2., size=1)
ytrue = np.random.normal(loc=np.exp(xtrue), scale=1, size=(ndims, 1))
zdata = np.random.normal(loc=xtrue + ytrue, scale=.75, size=(ndims, nobs))


with pm.Model() as model:
    x = pm.Normal('x', mu=0., sd=1)
    y = pm.Normal('y', mu=pm.exp(x), sd=2., shape=(ndims, 1)) # here, shape is telling us it's a vector rather than a scalar.
    z = pm.Normal('z', mu=x + y, sd=.75, observed=zdata) # shape is inferred from zdata

import pymc
import numpy as np


n = 5*np.ones(4,dtype=int)
x = np.array([.86,.3,.05,.73])


alpha = pymc.Normal('alpha',mu=0,tau=.01)
beta = pymc.Normal('beta',mu=0,tau=.01)


@pymc.deterministic
def theta(a=alpha, b=beta):
	    """theta = logit^{−1}(a+b)"""
            return pymc.invlogit(a+b*x)


d = pymc.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),\
                         observed=True)
                         
                         
                         