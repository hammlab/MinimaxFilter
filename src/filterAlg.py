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



class NN1(FilterAlg):

    # two-layer ReLU network, with m hidden units
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
        nlayers = hparams['nlayers']
        nhs = hparams['nhs'] # nlayers x 1
    
        W = [[] for i in range(nlayers+1)] # one more than nlayers
        # W[i][:,0] is the bias term
        # W[i] = nhs[l] x nhs[l-1]
        W[0] = np.random.normal(size=(nhs[0],1+D))
        for l in range(1,nlayers):
            W[l] = np.random.normal(size=(nhs[l],1+nhs[l-1]))
        W[nlayers] = np.random.normal(size=(d,1+nhs[nlayers-1]))

        #b = [[] for i in range(nlayers)] 
        #for i in range(nlayers-1):
        #    b[i] = np.random.normal(size=(nhs[i],))
        #b[nlayers-1] = np.random.normal(size=(d,))
    
        # net nj = sum_i Wji ai  
    
        # X => W[0]: nhs[0]x(1+D) => a[0]: nhs[0] => .... 
        #   =>  W[nlayers-1]: nhs[nlayers-1] x (1+nhs[nlayers-2]) => a[nlayers-1]
        #   =>  W[nlayers]: d x (1+nhs[nlayers-1]) => out
    
        return NN1.packW(W,hparams)

    @staticmethod        
    def parseW(w,hparams):
        D = hparams['D']
        d = hparams['d']
        nlayers = hparams['nlayers']
        nhs = hparams['nhs'] # nlayers x 1
            
        W = [[] for i in range(nlayers+1)] # one more than nlayers
            
        cnt = 0
        for l in range(nlayers+1):# 1:(n-2)
            if l==0:
                nt = nhs[0]
                nb = D+1
            elif l==nlayers:
                nt = d
                nb = nhs[l-1]+1
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
        #d = hparams['d']
        nlayers = hparams['nlayers']
        #nhs = hparams['nhs'] # nlayers x 1
            
        m = 0
        for l in range(nlayers+1):
            m += W[l].size
            
        w = np.zeros((m,))
        cnt = 0
        for l in range(nlayers+1):# 1:(n-2)
            nu = W[l].size        
            w[cnt:cnt+nu] = W[l].flatten()
            cnt += nu
            
        assert cnt==m
    
        return w

    @staticmethod        
    def g(w,X,hparams):
        #d = hparams['d']
        #l = hparams['l']
        D,N = X.shape
        nlayers = hparams['nlayers']
        #nhs = hparams['nhs']
        #d = hparams['d']
        W = NN1.parseW(w,hparams)
        a = NN1.forward(W,X,hparams)
        
        return a[nlayers]
        
        
    @staticmethod        
    def forward(W,X,hparams):
        #d = hparams['d']
        #l = hparams['l']
        D,N = X.shape
        nlayers = hparams['nlayers']
        #nhs = hparams['nhs']
        #d = hparams['d']
        
        a = [[] for i in range(nlayers+1)] #a[layer]: nhs[layer] x N
        for l in range(nlayers):
            # net = nhs[i] x N
            if (l==0):
                net = np.dot(W[l],np.vstack((np.ones((1,N)),X))) 
            else:
                net = np.dot(W[l],np.vstack((np.ones((1,N)),a[l-1])))
            if (hparams['activation']=='sigmoid'):
                a[l] = 1. / (1.+np.exp(-net))
            elif (hparams['activation']=='relu'):
                a[l] = np.maximum(0, net)
            else:
                print 'Unknown activation'
                exit()
        
        l = nlayers  
        #print a[l-1].shape
        net = np.dot(W[l],np.vstack((np.ones((1,N)),a[l-1])))
        if (hparams['output']=='sigmoid'):
            print 'Not impletemented yet'
            exit()
        elif (hparams['output']=='linear'):
            a[l] = net
        elif (hparams['output']=='softmax'):
            print 'Not impletemented yet'
            exit()
        else :
            print 'Unknown activation'
            exit()

        return a
        

    @staticmethod        
    def dgdu(w,X,hparams): # u.size x d x N 

        D,N = X.shape
        #l = hparams['l']
        nlayers = hparams['nlayers']
        nhs = hparams['nhs']
        d = hparams['d']
        
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
        for l in range(nlayers+1):
            m += W[l].size
        
        J = np.zeros((m,d,N)) 
        delk = [[] for l in range(nlayers+1)] # delk[l]: nhs[l] x N
        I = np.eye(d)
        
        for k in range(d):
            #% Compute delk(i) = dok/dni by backprop
            
            #% output layer 
            l = nlayers
            if (hparams['output']=='sigmoid'):
                delk[l] = np.tile(I[k,:].reshape((d,1)),(1,N))*a[l]*(1. - a[l])
            elif (hparams['output']=='linear'):
                delk[l] = np.tile(I[k,:].reshape((d,1)),(1,N))
            elif (hparams['output']=='softmax'):
                print 'Not impletemented yet'
                exit()
            else :
                print 'Unknown activation'
                exit()
            #% hidden layer
            for l in range(nlayers-1,-1,-1):#(n:= (n - 1) : -1 : 2
                #% Derivative of the activation function
                if (hparams['activation']=='sigmoid'):
                    d_act = a[l]*(1. - a[l])
                elif (hparams['activation']=='relu'):
                    d_act = np.zeros((nhs[l],N))
                    d_act[a[l]>0.] = 1.
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
            for l in range(nlayers+1):# 1:(n-2)
                nt,nb = W[l].shape # nt = nhs[l]
                nu = nt*nb; 
                # J[k,ij] = dok/Wij = delk(i) * aj : nhs[l] x N    nhs[l] x N
                #print delk[l].shape
                #print W[l].shape
                if l==0:
                    #print delk[l].shape
                    #print X.shape
                    #print nt
                    #print nb
                    J[cnt:cnt+nu,k,:] = (np.tile(delk[l].reshape((nt,1,N)),(1,nb,1)) \
                        *np.tile(np.vstack((np.ones((1,N)),X)).reshape((1,nb,N)),(nt,1,1))).reshape((nt*nb,N))
                else:
                    J[cnt:cnt+nu,k,:] = (np.tile(delk[l].reshape((nt,1,N)),(1,nb,1)) \
                        *np.tile(np.vstack((np.ones((1,N)),a[l-1])).reshape((1,nb,N)),(nt,1,1))).reshape((nt*nb,N))
                cnt += nu
            
            assert cnt==m
            '''            
            l = nlayers-1
            nt,nb = W[i].shape
            nu = nt*nb
            J[k,cnt:cnt+nu,:] = (np.tile((delk[l+1].T).reshape((nt,1,N)),(1,nb,1)) \
                *np.tile((a[l].T).reshape((1,nb,N)),(nt,1,1))).reshape((nt*nb, N))
            '''
    
            # u.size x d x N = D*d x d*N
    
        return J#.reshape()
   

    '''
    @staticmethod        
    def updateW(W,X,y,hparams):
        
        W -= 


        return W
    '''
   
    @staticmethod        
    def selftest1():
        
        D = 5
        N = 100
        hparams = {'D':D, 'd':3, 'nlayers':2, 'nhs':[30,30],'output':'linear', 'activation':'relu'}
        X = np.random.normal(size=(D,N))

        w = NN1.init(hparams)        
        W = NN1.parse(w,hparams)
        a = NN1.forward(W,X,hparams)
        J = NN1.dgdu(w,X,hparams) # u.size x d x N 
        #print J
        
        # Simple training experiment?
        
        
            

        