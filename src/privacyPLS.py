# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:53:05 2016

@author: hammj
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
#from scipy.optimize import minimize


## \max_u \mathrm{Cov}(u'X,Y1)^2 - \lambda\mathrm{Cov}(u'X,Y2)^2, s.t. u'u =1

def run(X,y1,y2,dmax=None):

    D,N = X.shape
    y1_unique,J1 = np.unique(y1, return_inverse=True)
    ny1 = y1_unique.size

    y2_unique,J2 = np.unique(y2, return_inverse=True)
    ny2 = y2_unique.size

    Y1 = np.zeros((ny1,N))
    Y2 = np.zeros((ny2,N))
    Y1[J1,range(N)] = 1.
    Y2[J2,range(N)] = 1.
    
    XY2 = np.dot(X,Y2.T) # D x ny2
    XY2Y2X = np.dot(XY2,XY2.T) # D x D
    XX = np.dot(X,X.T) # D x D
    
    P = np.zeros((D,0))
    Sj = np.dot(X,Y1.T) #  D x ny1

    if dmax==None:
        dmax = D

    for d in range(dmax):
        if d>0: 
            invPP = np.linalg.pinv(np.dot(P.T,P))
            Sj -= np.dot(np.dot(np.dot(P,invPP),P.T),Sj) 

        C = np.dot(Sj,Sj.T) - XY2Y2X
        C = 0.5*(C+C.T)
        dd,E = scipy.linalg.eigh(C,eigvals=(D-1,D-1)) # ascending order
        
        assert np.isnan(dd).any()==False
        assert np.iscomplex(dd).any()==False
        #dd = dd[::-1] #
        #E = E[:,::-1]
        wj = E#E[:,0] # D x 1
        pj = np.dot(XX,wj) / np.dot(np.dot(wj.T,XX),wj) # D x 1
        P = np.hstack((P,pj.reshape((D,1)))) #  D x d
    
        
    #P = P/np.tile(np.sqrt((P**2).sum(axis=0,keepdims=True)),(D,1))
    #% They need not be orthogonal.
    return P            
    

'''
def selftest1():
    
    # Generate data
    D0 = 5
    K1 = 2
    K2 = 2
    NperClass = 500
    N = NperClass*K1*K2
    #l = 1.0e-3
    X = np.zeros((D0,NperClass,K1,K2))
    y1 = np.zeros((NperClass,K1,K2),dtype=int)
    y2 = np.zeros((NperClass,K1,K2),dtype=int)
    bias1 = np.random.normal(scale=1.0,size=(D0,K1))
    bias2 = np.random.normal(scale=1.0,size=(D0,K2))    
    for k1 in range(K1):
        for k2 in range(K2):
            X[:,:,k1,k2] = \
                np.random.normal(scale=0.25, size=(D0,NperClass)) \
                + np.kron(np.ones((1,NperClass)),bias1[:,k1].reshape((D0,1))) \
                + np.kron(np.ones((1,NperClass)),bias2[:,k2].reshape((D0,1)))
            y1[:,k1,k2] = k1*np.ones((NperClass,))
            y2[:,k1,k2] = k2*np.ones((NperClass,))

    X = X.reshape((D0,N))
    y1 = y1.reshape((N,))
    y2 = y2.reshape((N,))
    
    P = run(X,y1,y2)
    print dd

    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(dd)
    plt.subplot(1,2,2)
    plt.imshow(E, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()    
    
'''    
    
    
    
