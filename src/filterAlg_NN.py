# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:25:07 2016

@author: hammj
"""

import numpy as np
from scipy.optimize import minimize
import sparse_autoencoder    


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
        

class NN1(FilterAlg):

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
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
    
        W = [[] for i in range(nlayers)] 
        # W[i][:,0] is the bias term
        # W[i] = nhs[l] x nhs[l-1]
        W[0] = np.random.normal(size=(nhs[0],1+D))
        for l in range(1,nlayers):
            W[l] = 1E0*np.random.normal(size=(nhs[l],1+nhs[l-1]))
    
        # X => W[0]: nhs[0]x(1+D) => a[0]: nhs[0] => .... 
        #   =>  W[nlayers-1]: nhs[nlayers-1] x (1+nhs[nlayers-2])
        #   => a[nlayers-1] = out
    
        return NN1.packW(W,hparams)

  
    @staticmethod        
    def parseW(w,hparams):
        D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
            
        W = [[] for i in range(nlayers)] # one more than nlayers
            
        cnt = 0
        for l in range(nlayers):
            if l==0:
                nt = nhs[0]
                nb = D+1
            else:
                nt = nhs[l]
                nb = nhs[l-1]+1
               
            nu = nt*nb
            W[l] = w[cnt:cnt+nu].reshape((nt,nb))
            cnt += nu
        
        assert cnt==w.size
        
        return W

        
    @staticmethod        
    def packW(W,hparams):
        
        #D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
            
        m = 0
        for l in range(nlayers):
            m += W[l].size
            
        w = np.zeros((m,))
        cnt = 0
        for l in range(nlayers):
            nu = W[l].size        
            w[cnt:cnt+nu] = W[l].flatten()
            cnt += nu
            
        assert cnt==m
    
        return w


    @staticmethod        
    def g(w,X,hparams):
        D,N = X.shape
        #D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
        #nhs = hparams['nhs']
        W = NN1.parseW(w,hparams)
        a = NN1.forward(W,X,hparams)
        
        return a[nlayers-1]
        
        
    @staticmethod        
    def forward(W,X,hparams):
        D,N = X.shape
        #D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
        
        a = [[] for i in range(nlayers)] #a[layer]: nhs[layer] x N
        for l in range(nlayers):
            # net = nhs[i] x N
            if (l==0):
                net = np.dot(W[l],np.vstack((np.ones((1,N)),X))) 
            else:
                net = np.dot(W[l],np.vstack((np.ones((1,N)),a[l-1])))
                
            if (hparams['activation']=='sigmoid'):
                indp = np.where(net>=0)
                indn = np.where(net<0)
                ta = np.zeros(net.shape)
                ta[indp] = 1./(1.+np.exp(-net[indp]))
                ta[indn] = np.exp(net[indn])/(1.+np.exp(net[indn]))
                a[l] = ta
                #a[l] = 1./(1.+np.exp(-net))
            #elif (hparams['activation']=='relu'):
            #    a[l] = np.maximum(0, net)
            #elif (hparams['activation']=='linear'):
            #    a[l] = net
            else :
                print 'Unknown activation'
                exit()

        return a
        

    @staticmethod        
    def dgdu(w,X,hparams): # u.size x d x N 

        D,N = X.shape
        #D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        d = nhs[-1]
        
        W = NN1.parseW(w,hparams)

        # Forward pass
        a = NN1.forward(W,X,hparams)
        
        # Compute Jacobian by back backpropagation
        # J[k,ij] = dok/dWij = dok/dni * dni/dWij
        # = delk(i) * aj, where delk(i)=dok/dni
        # Backprop:
        # delk(i) = dok/dni = sum_l dok/dnl* dnl/dni 
        # = a'(ni) sum_l Wli delk(l)
        
        #sparsityError = 0
        
        m = 0
        for l in range(nlayers):
            m += W[l].size
        
        J = np.zeros((m,d,N)) 
        delk = [[] for l in range(nlayers+1)] # delk[l]: nhs[l] x N
        I = np.eye(d)
        
        for k in range(d):
            #% Compute delk(i) = dok/dni by backprop
            
            #% output layer 
            l = nlayers-1
            if (hparams['activation']=='sigmoid'):
                delk[l] = np.tile(I[k,:].reshape((d,1)),(1,N))*a[l]*(1. - a[l])
            #elif (hparams['activation']=='linear'):
            #    delk[l] = np.tile(I[k,:].reshape((d,1)),(1,N))
            else :
                print 'Unknown activation'
                exit()
            #% hidden layer
            for l in range(nlayers-2,-1,-1):
                #% Derivative of the activation function
                if (hparams['activation']=='sigmoid'):
                    d_act = a[l]*(1.-a[l])
                #elif (hparams['activation']=='relu'):
                #    d_act = np.zeros((nhs[l],N))
                #    d_act[a[l]>0.] = 1.
                else :
                    print 'Unknown activation'
                    exit()
                # a[l]: nhs[l] x N
                # d_act: nhs[l] x N
                # W[l]:nhs[l] x nhs[l-1]
                # delk(i) = a'(ni) sum_l Wli delk(l) = a'(ni) W:i'*delk:
                delk[l] = d_act*np.dot(W[l+1][:,1:].T,delk[l+1]) # Bishop (5.56)
                # delk[l]: nhs[l] x N = nhs[l]xN  .* (nhs[l]xnhs[l+1] nhs[l+1]xN)
                
                #if(nn.dropoutFraction>0)
                #    delk{i} = delk{i} .* [ones(size(delk{i},1),1) nn.dropOutMask{i}];
            
            # Jacobian from del
            cnt = 0
            for l in range(nlayers):# 1:(n-2)
                nt,nb = W[l].shape # nt = nhs[l]
                nu = nt*nb; 
                # J[k,ij] = dok/Wij = delk(i) * aj : nhs[l] x N    nhs[l] x N
                #print delk[l].shape
                #print W[l].shape
                if l==0:
                    J[cnt:cnt+nu,k,:] = (np.tile(delk[l].reshape((nt,1,N)),(1,nb,1)) \
                        *np.tile(np.vstack((np.ones((1,N)),X)).reshape((1,nb,N)),(nt,1,1))).reshape((nt*nb,N))
                else:
                    J[cnt:cnt+nu,k,:] = (np.tile(delk[l].reshape((nt,1,N)),(1,nb,1)) \
                        *np.tile(np.vstack((np.ones((1,N)),a[l-1])).reshape((1,nb,N)),(nt,1,1))).reshape((nt*nb,N))
                cnt += nu
            
            assert cnt==m
   
            # u.size x d x N = D*d x d*N
    
        return J#.reshape()
   

    
    @staticmethod        
    def initByAutoencoder(X,hparams):
        D = hparams['D']
        nhs = hparams['nhs'] # nlayers x 1
        nlayers = len(nhs)
        #d = nhs[-1]
    
        W = NN1.parseW(NN1.init(hparams),hparams)

        sparsity_param = 0.1  # desired average activation of the hidden units.
        lambda_ = 1e-4  # weight decay parameter
        beta = 3  # weight of sparsity penalty term
        options_ = {'maxiter': 400, 'disp': False}
        
        # Train autoencoder layer-by-layer
        for l in range(nlayers):
            if l==0:
                x0 = sparse_autoencoder.initialize(nhs[l], D)
                cost = lambda x: sparse_autoencoder.sparse_autoencoder_cost\
                    (x, D, nhs[l], lambda_, sparsity_param, beta, X)
                while True:
                    res = minimize(cost, x0, method='L-BFGS-B', jac=True, options=options_)
                    if np.isnan(res.x).any() == False:
                        break;
                        
                W[l][:,1:] = res.x[0:nhs[l]*D].reshape(nhs[l],D)
                W[l][:,0] = res.x[2*nhs[l]*D:2*nhs[l]*D+nhs[l]]
                feat = sparse_autoencoder.sparse_autoencoder(res.x,nhs[l],D,X)
            else:
                x0 = sparse_autoencoder.initialize(nhs[l], nhs[l-1])
                cost = lambda x: sparse_autoencoder.sparse_autoencoder_cost\
                    (x, nhs[l-1], nhs[l], lambda_, sparsity_param, beta, feat)
                while True:
                    res = minimize(cost, x0, method='L-BFGS-B', jac=True, options=options_)
                    if np.isnan(res.x).any() == False:
                        break;
                W[l][:,1:] = res.x[0:nhs[l]*nhs[l-1]].reshape(nhs[l],nhs[l-1])
                W[l][:,0] = res.x[2*nhs[l]*nhs[l-1]:2*nhs[l]*nhs[l-1]+nhs[l]]
                feat = sparse_autoencoder.sparse_autoencoder(res.x,nhs[l],nhs[l-1],feat)
        
        

        return NN1.packW(W,hparams)

    
   
    @staticmethod        
    def selftest1():
        
        D = 5
        N = 100
        d = 3
        hparams = {'D':D, 'nhs':[5,5,d],'activation':'sigmoid'}#relu'}
        X = np.random.normal(size=(D,N))

        w = NN1.init(hparams)   
        w = NN1.initByAutoencoder(X,hparams)
        
        W = NN1.parseW(w,hparams)
        a = NN1.forward(W,X,hparams)
        J = NN1.dgdu(w,X,hparams) # u.size x d x N 
        #print J
        
        # Simple training experiment?
        
        
            

        