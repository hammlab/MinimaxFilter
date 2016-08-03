# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 23:55:16 2015

@author: hammj
"""

import numpy as np
#import numpy.random.laplace
#import numpy.random.normal
import matplotlib.pylab as plt

# Random multivariate Laplace distribution
def sample(invbeta,D,N) :

    if invbeta==0.0:
        return np.zeros((D,N))
    Z = np.random.laplace(scale=invbeta, size=(1,N))    
    X = np.random.normal(size=(D,N))
    X = X/np.kron(np.ones((D,1)),np.sqrt((X**2).sum(axis=0).reshape((1,N)))); 
    # uniform on unit sphere
    Y = np.kron(np.ones((D,1)),Z)*X
    
    return Y
    
    
def selftest():
    D = 2
    N = 1000
    invbeta = 1.0
    #samples = randmvlap.sample(invbeta,D,N)    
    samples = sample(invbeta,D,N)    
    
    plt.figure(1)
    plt.clf()
    plt.plot(samples[0,:],samples[1,:],'.')
    plt.show()
