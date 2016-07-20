# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:31:31 2016

@author: hammj
"""

import numpy as np

def run(u,v,maxiter_main,f,dfdu,Phi,Phi_lin,*args):

    
    #% \min_{u \in \X} \max_{y \in \Y} f(u,y)
    #% where \X \subset \R^n and \Y \subset \R^m are convex compact sets and 
    #% f(u,y), \nabla_x f(u,y) are continuous with respect to \X and \Y.
    #% Also assume for y \in \Y and x_1,x_2 \in \X, \nabla_x f is Lipschitz continuous in u,
    #% \|\nabla_x f(x_1,y) - \nabla_x f(x_2,y) \| \leq K \|x_1 - x_2\|
    #% where K > 0 is a constant.
    #% Problem is reformulated as 
    #% \min_{u \in \X} \Phi(u), where \Phi(u) = \max_{y\in\Y} f(u,y).
    #% 
    #% Kiwiel's algorithm: linear approximation to the max function
    #% f^l_k (d,y) = f(x_k,y) + <\nabla_x f(x_k,y), d>
    #% Phi^l_k(d) = \max_{y\in\Y} f^l_k(d,y).
    #% At x_k the AA evaluate the descent direction d_k in finite number of iterations by
    #% \min_{d \in R^n} \Phi^l_k(d) + 1/2\|d\|^2.
    
    #% Step 0. Initialization: select x0, y0; set k=0 and
    #% 1) termination accuracy 1 >> xi >=0 (xi=1E-6)
    #% 2) line search parameter c in (0,1), (c = 1E-4)
    #% 3) stepsize factor sigma i in (0,1), (sigma = 0.5)
    #% linear approximation parameter m in (0,1) (m = 2E-4)
    #% Step 1. Solve the maximization at current point xk:
    #% Phi(xk) = max_y f(xk,y)
    #% Step 2. Direction-finding subproblem: Set u = xk and use auxiliary algorithm
    #% (AA) with parameters xi >= 0 and m until it terminates, returning dk and Psi^l_k.
    #% If Psi^l_k >= -xi, the solution has been reached: stop.
    #% Step 3. Line search: compute the stepsize ak using
    #% ak = max {a | Phi(xk + a*dk) - Phi(xk) <= c a Psi_k, a = (sigma)^i, i=0,1,2,..}
    #% Set xk+1 = xk + ak*dk, k = k + 1, go to Step 1.
    
    #% Step 0. Initialization: 
    xi = 1E-6
    c = 1E-4
    sigma = 0.5
    m = 2E-4
    
    maxiter_aa = 200
    maxiter_Phi = 200
    maxiter_Phi_lin = 200
    maxiter_linesearch = 30

    
    for it in range(maxiter_main):
        #% Step 1. Solve the maximization at current point xk
        
        v_,Phiu = Phi(u,v,maxiter_Phi,*args)
        tPhiu = Phiu
        
        #%fprintf('Main iter=%d/%d, Phiu=%f\n',iter,maxiter_main,Phiu);
        #% Step 2. Direction-finding subproblem 
        q,Psi = _AuxiliaryAlgorithm(u,v_,\
                Phiu,xi,m,maxiter_aa,maxiter_Phi_lin,f,dfdu,Phi_lin,*args)
        if Psi >= -xi:
            break
            
        #% Step 3. Line search
        if True: #% standard
            al = 1.0
            for i in range(maxiter_linesearch):#%while 1
                _,tPhiu = Phi(u+al*q,v_,maxiter_Phi,*args)
                if tPhiu - Phiu <= c*al*Psi:
                    break
                al *= sigma
            
            if tPhiu - Phiu > c*al*Psi:
                print 'No improvement in line search!'
                #return (u,v)

            u += al*q
        else: #%  Grassmann manifold
            pass

        v = v_
    #Phiu = tPhiu

    return (u,v)


def _AuxiliaryAlgorithm(u,v,Phiu,xi,m,maxiter_aa,maxiter_Phi_lin,f,dfdu,Phi_lin,*args):

    #% Auxiliary Algorithm (AA) (requires input values: xk in \R^n, Phi(xk), xi >=0, m in (0,1)
    #% Step 0. Initialization: set u = xk, Phi(u) = Phi(xk), select any w in Y, set
    #% p0 = \nabla_x f(u,w), \Theta_0 = f(u,w), i=1.
    #% Step 1. Find the number mu_i that solves
    #% min_{\mu in R} { 1/2||(1-mu)p_{i-1} + mu\nabla_x f(u,yi)||^2 - (1-mu)Theta_{i-1} - mu f(u,yi) }
    #% Set
    #% pi = (1-mu_i)p_{i-1} + mu_i \nabla_x f(u,y_i), \Theta_i = (1-\mu_i)\Theta_{i-1} + \mu_i f(u,y_i);
    #% \Psi_i = -\{ \|p_i\|^2 + \Phi(u) - \Theta_i \}.
    #% If \Psi_i \geq -\xi then go to Step 3.
    #% Step 2. Primal optimality testing: set d_i = -p_i. Compute
    #% y_{i+1} = \arg\max_y  \{f(u,y) + \langle \nabla_x f(u,y), d_i \rangle \}.
    #% 
    #% If f(u,y_{i+1}) + \langle \nabla_x (u,y_{i+1}), d_i \rangle - \Phi(u) \leq m \Psi_i
    #% then, go to Step 3. Else, set i = i + 1, and go to Step 1.
    #% Step 3. Stop returning d_k = -p_i and \Psi^l_k = \Psi_i.
    
    #%Step 0. Initialization:
    #% Phiu = Phiu; % given as argument
    #%y = ymax; % given as arguments
    
    fval = f(u,v,*args)
    dfdu_ = dfdu(u,v,*args)    
    p = dfdu_
    t = fval 
    Psi = 0.0
    assert np.isnan(p).any()==False

    for it in range(maxiter_aa):
        #%fprintf('Aux: %d/%d, Psi=%f\n',iter,maxiter_aa,Psi);
        #%Step 1. Find the number mu_i that solves
        #%min_{\mu in R} { 1/2||(1-mu)p_{i-1} + mu\nabla_x f(u,yi)||^2 - (1-mu)Theta_{i-1} - mu f(u,yi) }
        #%1/2*|mu(g-p) + p|^2 + mu(t-f) - t = 1/2*mu^2(g-p)'(g-p)+mu(g-p)'p + 1/2*p'p + mu(t-f)
        #%= mu^2 (1/2 |g-p|^2) + mu((g-p)'p + t-f) + const
        #%=> mu = -((g-p)'p + t-f)/(|g-p|^2)
        fval = f(u,v,*args)
        dfdu_ = dfdu(u,v,*args)    
    
        if it==0:
            mu = .5
        else:
            mu = -(np.dot(dfdu_-p,p) + t-fval)/np.dot(dfdu_-p,dfdu_-p)
        
        p = (1.-mu)*p + mu*dfdu_
        t = (1.-mu)*t + mu*fval
        Psi = -(np.dot(p,p) + Phiu - t)
        if Psi >= -xi:
            break
    
        #%Step 2. Primal optimality testing:
        q = -p
    
        v,tPhiu = Phi_lin(u,v,q,maxiter_Phi_lin,*args)
        if tPhiu - Phiu <= m*Psi:
            break

    q = -p

    assert np.isnan(q).any()==False
    
    return (q,Psi)



def selftest1():
    
    # Solve min_u max_v f(u,v) using Kiwiel's method, where
    # f(u,v) = |u|^2 - |v|^2 -2*u'v = 2u'u - (v+u)'(v+u)
    
    # Create local functions and pass them to kiwiel.run()
    def f(u,v,*args):
        # f(u,v) = |u|^2 - |v|^2 -2*u'v = 2u'u - (v+u)'(v+u)
        #u = u.flatten()
        fval = 2*np.dot(u,u) - np.dot(v+u,v+u)
        return fval

    
    def dfdu(u,v,*args):
        # f(u,v) = |u|^2 - |v|^2 -2*u'v = 2u'u - (v+u)'(v+u)
        #u = u.flatten()
        dfdu_ = 2*u - 2*v
        return dfdu_
        
    
    def flin(u,v,q,*args):
        # f_(v;xk,v) = f(xk,v) + dfdu(xk,v)'*q.    
        # = u'u-v'v-2u'v + 2(u-v)'q
    
        #fval,dfdu = f(u,v,*args)
        #return fval + np.dot(dfdu,q)
        return np.dot(u,u) - np.dot(v,v) -2*np.dot(u,v) + 2*np.dot(u-v,q)
    
        
    def Phi(u,v,*args): 
        # max_v f(u,v) = max_v 2u'u -(v+u)'(v+u)
        # = 2 u'u 
        #u = u.flatten()
    
        return (-u, 2*np.dot(u,u))
    
        
    def Philin(u,v,q,*args):
        # f_(v;xk,v) = f(xk,v) + dfdu(xk,v)'*q.    
        # = u'u-v'v-2u'v + 2(u-v)'q
        
        # Philin = max_v flin(q,v) = max_v (u'u-v'v-2u'v + 2(u-v)'q)
        # = max_v ( - (v'v +2v'(u+q) +(u+q)'(u+q)) + (u+q)'(u+q) + u'u + 2u'q
        # = (u+q)'(u+q) + u'u + 2u'q = 2u'u + 4u'q + q'q
    
        #u = u.flatten()
        #v = v.flatten()
        return (-(u+q), 2*np.dot(u,u) + 4*np.dot(u,q) + np.dot(q,q))
    
    
    
    maxiter = 10
    maxiter_main = 1
    
    #args = [[]]
    
    D = 10
    u = np.random.normal(size=(D,))
    v = np.random.normal(size=(D,))
    
    for iter in range(maxiter):
        
        u,v = run(u,v,maxiter_main,f,dfdu,Phi,Philin)#,args)
        #print u,v
        fval = f(u,v)
        print fval
    
        
        
        
        
    