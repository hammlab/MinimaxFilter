# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 23:19:55 2016

@author: hammj
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import line_search


def run1(u,wv,maxiter,f,dfdu,Phi,Phi_lin,rho,\
    X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):

    '''
    min_u  Phi(u), where
    Phi(u) = -rho max_w -f_util(u,w) + max_v -f_priv(u,v)
    = rho min_w f_util(u,w) + max_v -f_priv(u,v)
    '''
    #eta = 1e0
    D,N = X.shape
    #d = hparams0['d']
    maxiter_Phi = 50
    #c = 1E-4
    #sigma = 0.5
    #maxiter_linesearch = 30
    
    for iter in range(maxiter):
        # 1. Update w, v
        #w -= eta*alg1.dfdv(w,G,y1,hparams1)
        #v -= eta*alg2.dfdv(v,G,y2,hparams2)
        # or fully optimize
        wv,Phiu = Phi(u,wv,maxiter_Phi,rho,\
            X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        
        # 2. Update u
        # Hopefuly, by an analog of Danskin's theorem,
        # df/du = rho*df_util(u,wh)/du  -df_priv(u,vh)/du, where  
        # wh = argmin f_util, vh = argmin f_priv are UNIQUE solutions
        dfdu_ = dfdu(u,wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        q = -dfdu_
        # SGD: u -= eta*dfdu_
        # Or Line search
        res = line_search(f,dfdu,u,q,gfk=dfdu_,\
            args=(wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2))
        al = res[0]

        if (al==None):
            print 'No improvement in line search!'
        else:
            u += al*q

    #    Danskin's theorem:
    #   if -f_priv is convex in u (it's not) and hat(v) is unique (it's possible),
    #  then dPhi_v(u)/du = -rho*dPhi(u,wh)

    return (u,wv)


def run2(u,wv,maxiter,f,dfdu,Phi,Phi_lin,rho,\
    X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):

    print 'Not working for some reason'
    exit()
    
    '''
    min_u  Phi(u), where
    Phi(u) = -rho max_w -f_util(u,w) + max_v -f_priv(u,v)
    = rho min_w f_util(u,w) + max_v -f_priv(u,v)
    '''

    D,N = X.shape
    #d = hparams0['d']
    maxiter_Phi = 5
    maxiter_outer = 5
    
    for iter in range(maxiter):
        # 1. Update w, v
        #w -= eta*alg1.dfdv(w,G,y1,hparams1)
        #v -= eta*alg2.dfdv(v,G,y2,hparams2)
        # or fully optimize
        wv,Phiu = Phi(u,wv,maxiter_Phi,rho,\
            X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        
        # 2. Update u
        # min_u  (rho*f_util(u,wh) - f_priv(u,vh))
        # wh = argmin f_util, vh = argmin f_priv are UNIQUE solutions
        res = minimize(f, u, args=(wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2),\
            method='BFGS', jac=dfdu, options={'disp':False, 'maxiter':maxiter_outer})    
        u = res.x
        #Phiu = -res.fun

    return (u,wv)

