# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 22:05:37 2015

@author: hammj
"""

import numpy as np
from scipy.optimize import minimize

import kiwiel



def run(W0,W1,W2,rho,maxiter_main,*args):
    #X,y1,y2,\
    #filt0,alg1,alg2,hparams0,hparams1,hparams2):

    # util: min_u min_w f1(u,w) = min_u -max_w -f1(u,w) 
    # priv: max_u min_v f2(u,v) = -min_u -min_v f2(u,v) = -min_u max_v -f2(u,v)

    # Joint task:  min_u [-rho*max_w -f1(u,w) + max_v -f2(u,v)] = min_u Phi(u)
    # where Phi(u) = -rho*max_w -f1(u,w) + max_v -f2(u,v)
    # = -rho Phi1(u) + Phi2(u), where 
    # Phi1(u) = max_w -f1, Phi2(u) = max_v -f2
     
    # Also let f(u,wv) = rho*f1(u,w) - f2(u,v). 
    # Note, max_uv f(u,wv) = rho*max_w f1(u,w) + max_v -f2(u,v)
    # is not the same as Phi(u) = -rho*Phi1(u) + Phi2(u) 
    
    u = W0.flatten()
    w = W1.flatten()
    v = W2.flatten()
    
    '''#% check gradients
    if 0
        u = randn(size(u)); v = randn(size(v)); w = randn(size(v));
        check_gradient(@(u)ftemp1(u,w,v,X(:,1:100),y1(:,1:100),y2(:,1:100),K1,K2,rho,lambda0,lambda1,lambda2),u(:));
        % 
        q = randn(size(u));
        check_gradient(@(v)f_lin(u,v,q,X(:,1:100),y1(:,1:100),K1,lambda0,lambda1),v(:));
        % 
    end
    '''
    
    u,wv = kiwiel.run(u,(w,v),maxiter_main,_f_joint,_Phi_joint,_Phi_lin_joint,rho,*args)
    #u,v,Phiu = kiwiel.run(u,v,maxiter_main,f,Phi,Philin)#,args)
    w = wv[0]
    v = wv[1]
    
    #% Check dfdx
    #function [fval,dfdu] = ftemp1(u,w,v,X,y1,y2,K1,K2,rho,lambda0,lambda1,lambda2)
    #    [fval,~,dfdu] = f_joint(u,w,v,X,y1,y2,K1,K2,rho,lambda0,lambda1,lambda2);
    #end
    return (u,w,v)


def _f_joint(u,wv,\
        rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):        

    # f(u,wv) = rho*f1(u,w) - f2(u,v). 
    D,N = X.shape
    w = wv[0]
    v = wv[1]
    G = filt0.g(u,X,hparams0)
    d = G.shape[0]
    f1 = alg1.f(w,G,y1,hparams1)
    f2 = alg2.f(v,G,y2,hparams2)

    # dgdu: u.size x d x N
    dgdu = filt0.dgdu(u,X,hparams0).reshape((u.size,d*N)) # u.size x d x N

    df1 = alg1.dfdu(w,G,y1,dgdu,hparams1)
    df2 = alg2.dfdu(v,G,y2,dgdu,hparams2)        

    fval = rho*f1 - f2
    dfdu = rho*df1 - df2

    assert np.isnan(fval)==False
    assert np.isnan(dfdu).any()==False
    return (fval,dfdu)


def _Phi_joint(u,wv,maxiter,\
        rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):        
    # Phi(u) = -rho*max_w -f1(u,w) + max_v -f2(u,v)
    # = -rho Phi1(u) + Phi2(u), where 
    # Phi1(u) = max_w -f1, Phi2(u) = max_v -f2
    
    w = wv[0]
    v = wv[1]

    # Phi1(u) = max_w -f_util(u,w) = -min_w f_util(u,w)
    G = filt0.g(u,X,hparams0)
    res = minimize(alg1.f, w, args=(G,y1,hparams1),\
        method='BFGS', jac=alg1.dfdv, options={'disp':False, 'maxiter':maxiter})    
    w = res.x
    Phiu1 = -res.fun
    
    # Phi2(u) = max_v -f_priv(u,v) = -min_v f_priv(u,v)
    res = minimize(alg2.f, v, args=(G,y2,hparams2),\
        method='BFGS',jac=alg2.dfdv, options={'disp':False, 'maxiter':maxiter})    
    v = res.x
    Phiu2 = -res.fun

    Phiu = -rho*Phiu1 + Phiu2

    assert np.isnan(w).any()==False
    assert np.isnan(v).any()==False
    assert np.isnan(Phiu)==False

    return ((w,v),Phiu)


def _Phi_lin_joint(u,wv,q,maxiter,\
    rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):        

    w = wv[0]
    v = wv[1]

    G = filt0.g(u,X,hparams0)
    d,N = G.shape
    dgdu = filt0.dgdu(u,X,hparams0).reshape((u.size,d*N)) # u.size x d x N
    
    # Phi_lin(u) = -rho*max_w -f1(u,w) + max_v -f2(u,v)
    # = -rho Phi1lin(u) + Phi2lin(u), where 
    # Phi1lin(u) = max_w -f1lin, Phi2lin(u) = max_v -f2lin
    
    res = minimize(alg1.flin, w, args=(q,G,y1,dgdu,hparams1), \
        method='BFGS',jac=alg1.dflindv, options={'disp':False, 'maxiter':maxiter})    
    w = res.x
    Phiu1 = -res.fun


    res = minimize(alg2.flin, v, args=(q,G,y2,dgdu,hparams2),\
        method='BFGS',jac=alg2.dflindv, options={'disp':False, 'maxiter':maxiter})    
    v = res.x
    Phiu2 = -res.fun

    Phiu = -rho*Phiu1 + Phiu2
    
    assert np.isnan(w).any()==False
    assert np.isnan(v).any()==False
    assert np.isnan(Phiu)==False

    return ((w,v),Phiu)



def selftest1():
    
    import privacyLDA
    from filterAlg import Linear
    from learningAlg import mlogreg 
    
    # Generate data
    D0 = 5
    K1 = 2
    K2 = 3
    NperClass = 100
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
                np.random.normal(scale=1.0, size=(D0,NperClass)) \
                + np.tile(bias1[:,k1].reshape((D0,1)),(1,NperClass)) \
                + np.tile(bias2[:,k2].reshape((D0,1)),(1,NperClass))
            y1[:,k1,k2] = k1*np.ones((NperClass,))
            y2[:,k1,k2] = k2*np.ones((NperClass,))
    
    X = X.reshape((D0,N))
    y1 = y1.reshape((N,))
    y2 = y2.reshape((N,))
    Ntrain = np.floor(N/2.)
    #Ntest = N - Ntrain
    ind = np.random.choice(range(N),size=(N,),replace=False)
    ind_train = ind[:Ntrain]
    ind_test = ind[Ntrain:]
    
    ###########################################################################
    
    maxiter = 30
    maxiter_main = 1
    maxiter_final = 50
    rho = 1.
    lambda0 = 1e-8
    lambda1 = 1e-8
    lambda2 = 1e-8
    
    d = 2
    
    hparams0 = {'d':d, 'l':lambda0, 'D':D0}
    hparams1 = {'K':K1, 'l':lambda1, 'd':d}
    hparams2 = {'K':K2,'l':lambda2, 'd':d}
    
    if True:
        U,dd = privacyLDA.run(X[:,ind_train],y1[ind_train],y2[ind_train])
        w0 = U[:,0:d].flatten()
    else:
        w0 = Linear.init(hparams0)
    #print (W0**2).sum()    
    w1 = mlogreg.init(hparams1)
    w2 = mlogreg.init(hparams2)
    
    
    for iter in range(maxiter):
        #print (W0**2).sum()
        G_train = Linear.g(w0,X[:,ind_train],hparams0)
       
        # Full training
        tW1,f1 = mlogreg.train(G_train,y1[ind_train],hparams1,None,maxiter_final)
        tW2,f2 = mlogreg.train(G_train,y2[ind_train],hparams2,None,maxiter_final)
        
        # Testing error
        G_test = Linear.g(w0,X[:,ind_test],hparams0)
        
        rate1,_ = mlogreg.accuracy(tW1,G_test,y1[ind_test])
        rate2,_ = mlogreg.accuracy(tW2,G_test,y2[ind_test])
    
        print 'rate_tar= %.2f, rate_subj= %.2f' % (rate1,rate2)
    
        # run one iteration
        w0,w1,w2 = run(w0,w1,w2,rho,maxiter_main,\
            X[:,ind_train],y1[ind_train],y2[ind_train],\
            Linear,mlogreg,mlogreg,\
            hparams0,hparams1,hparams2)
    
        val,_ = _f_joint(w0,(w1,w2),\
            rho,X[:,ind_train],y1[ind_train],y2[ind_train],\
            Linear,mlogreg,mlogreg,\
            hparams0,hparams1,hparams2)
            
        print 'val=', val, '\n'



def selftest2():
    
    from filterAlg import NN1
    from learningAlg import mlogreg 
    
    # Generate data
    D0 = 5
    K1 = 2
    K2 = 3
    NperClass = 100
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
                np.random.normal(scale=1.0, size=(D0,NperClass)) \
                + np.tile(bias1[:,k1].reshape((D0,1)),(1,NperClass)) \
                + np.tile(bias2[:,k2].reshape((D0,1)),(1,NperClass))
            y1[:,k1,k2] = k1*np.ones((NperClass,))
            y2[:,k1,k2] = k2*np.ones((NperClass,))
    
    X = X.reshape((D0,N))
    y1 = y1.reshape((N,))
    y2 = y2.reshape((N,))
    Ntrain = np.floor(N/2.)
    #Ntest = N - Ntrain
    ind = np.random.choice(range(N),size=(N,),replace=False)
    ind_train = ind[:Ntrain]
    ind_test = ind[Ntrain:]
    
    ###########################################################################
    
    maxiter = 30
    maxiter_main = 1
    maxiter_final = 50
    rho = 1.
    lambda0 = 1e-8
    lambda1 = 1e-8
    lambda2 = 1e-8
    
    d = 2
    
    hparams0 = {'D':D0, 'd':d, 'nlayers':2, 'nhs':[3,3], 'output':'linear',\
        'activation':'relu', 'l':lambda0}
    hparams1 = {'K':K1, 'l':lambda1, 'd':d}
    hparams2 = {'K':K2,'l':lambda2, 'd':d}
    
    w0 = NN1.init(hparams0)
    w1 = mlogreg.init(hparams1)
    w2 = mlogreg.init(hparams2)


    for iter in range(maxiter):
        #print (W0**2).sum()
        G_train = NN1.g(w0,X[:,ind_train],hparams0)
       
        # Full training
        tW1,f1 = mlogreg.train(G_train,y1[ind_train],hparams1,None,maxiter_final)
        tW2,f2 = mlogreg.train(G_train,y2[ind_train],hparams2,None,maxiter_final)
        
        # Testing error
        G_test = NN1.g(w0,X[:,ind_test],hparams0)
        
        rate1,_ = mlogreg.accuracy(tW1,G_test,y1[ind_test])
        rate2,_ = mlogreg.accuracy(tW2,G_test,y2[ind_test])
    
        print 'rate_tar= %.2f, rate_subj= %.2f' % (rate1,rate2)
    
        # run one iteration
        w0,w1,w2 = run(w0,w1,w2,rho,maxiter_main,\
            X[:,ind_train],y1[ind_train],y2[ind_train],\
            NN1,mlogreg,mlogreg,\
            hparams0,hparams1,hparams2)
    
        val,_ = _f_joint(w0,(w1,w2),\
            rho,X[:,ind_train],y1[ind_train],y2[ind_train],\
            NN1,mlogreg,mlogreg,\
            hparams0,hparams1,hparams2)
            
        print 'val=', val, '\n'
    