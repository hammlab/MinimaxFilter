# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:27:15 2016

@author: hammj
"""

import numpy as np

class FilterAlg:
    pass
    '''
    def __init__(self,X,hyperparams):
        self.X = X
        self.hyperparams = hyperparams
        return
        
    def g(self,u):
        pass
        return
        
    '''
        
    
class Linear(FilterAlg):

    '''
    def __init__(self, X, d):
        self.X = X
        self.d = d
    
    def reset(self, X, d):
        self.X = X
        self.d = d
    '''

    @staticmethod        
    def init(hparams):
        D = hparams['D']        
        d = hparams['d']
        # random normal
        u = np.random.normal(size=(D*d,))

        return u
        
        
    @staticmethod        
    def g(u,X,hparams):
        d = hparams['d']
        #l = hparams['l']
        D,N = X.shape
        W = u.reshape((D,d))
        
        return np.dot(W.T,X)

    @staticmethod        
    def dgdu(u,X,hparams): # u.size x d x N = D*d x d*N
        d = hparams['d']
        D,N = X.shape
        # Jacobian: u.size x d x N
        #d = hparams['d']
        #l = hparams['l']
        #g(X;u) = [u1...ud]'X     
        # dgiduj = I[i=j]*X
        dg = np.zeros((D,d,d,N))
        for i in range(d):
            dg[:,i,i,:] = X
        
        return dg.reshape((D*d, d, N))



        