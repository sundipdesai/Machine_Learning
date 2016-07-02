# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:06:19 2016
@author: Sundip Desai
Naive Bayes
-----------
Rev 1:
The following set of functions employ
the necessary data processing and algorithms
to achieve Naive Bayes classification
Rev 2 to do:
1. Modularize for N classes
2. Employ other distribution options (Multinomial, Bernoulli, etc.)
"""
import numpy as np

# Load data
# Takes last column as the target vector
def loadData(name):
   data = np.loadtxt(name, delimiter=',')
   inputs = data[:,:-1] 
   targets = data[:,inputs.shape[1]]   
   return data, inputs, targets
   
# Segment data for 2 classes <ad-hoc>
# Need to modularize this!!
def segmentData(x, y):
    class1 = np.where(y == 2)
    class2 = np.where(y == 4)
    
    class1Inputs = x[class1]
    class2Inputs = x[class2]
    
    return class1Inputs, class2Inputs
    
# Compute mu and sigma
def computeMuAndSigma(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    
    return mu, sigma
    
# Compute probabilites from test data
# using Gaussian Distribution  
def computeProbability(mu, sigma, data):
    p=1/(sigma*np.sqrt(2*np.pi))
    p=p*np.exp(-((data-mu)**2)/(2*sigma**2))

    return p
    
# Accumulate probabilities from different distribuions
# and determine which class has the highes probability
# of being true (i.e., argmax (prod(probabilities)))    
def getClass(p1, p2, targets):
    
    # p1 is associated with class 1
    # p2 is associaed with class 2
    
    # compute product of feature probabilities
    p1p=np.prod(p1,axis=1)
    p2p=np.prod(p2,axis=1)

    ct=0
    for i in range(len(p1p)):
        if (p1p[i] > p2p[i]):
            isClass = 2
        else:
            isClass = 4
    
        if (isClass == targets[i]):
            ct+=1

    predictRate=float(ct)/len(targets)*100   
print "Success rate of Naive Bayes is: ", predictRate
