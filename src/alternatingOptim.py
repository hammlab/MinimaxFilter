'''
Copyright 2016 Jihun Hamm
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License
'''

import numpy as np
#from scipy.optimize import minimize
from scipy.optimize import line_search


def run(u,wv,maxiter,f,dfdu,Phi,Phi_lin,rho,\
    X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):

    '''
    min_u  Phi(u), where
    Phi(u) = -rho max_w -f_util(u,w) + max_v -f_priv(u,v)
    = rho min_w f_util(u,w) + max_v -f_priv(u,v)
    '''
    D,N = X.shape
    #d = hparams0['d']
    maxiter_Phi = 50
    #method = 'rmsprop'
    method = 'linesearch'
    eta = 1E-3
    #c = 1E-4
    #sigma = 0.5
    #maxiter_linesearch = 30
    if not hasattr(run,'tsum'):
        run.tsum=np.zeros(u.shape)
   
    for iter in range(maxiter):
        # 1. Update w, v
        #w -= eta*alg1.dfdv(w,G,y1,hparams1)
        #v -= eta*alg2.dfdv(v,G,y2,hparams2)
        # or fully optimize
        Phi_,wv = Phi(u,wv,maxiter_Phi,rho,\
            X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        
        # 2. Update u
        # Hopefuly, by an analog of Danskin's theorem,
        # df/du = rho*df_util(u,wh)/du  -df_priv(u,vh)/du, where  
        # wh = argmin f_util, vh = argmin f_priv are UNIQUE solutions
        dfdu_ = dfdu(u,wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        q = -dfdu_
        if method=='linesearch':
            res = line_search(f,dfdu,u,q,gfk=dfdu_,\
                args=(wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2))
            al = res[0]
    
            if (al==None):
                print 'No improvement in line search!'
            else:
                u += al*q
        elif method=='const':
            #eta = 1.
            u += eta*q
        elif method=='rmsprop':
            r = .9
            #eta = 1E-3
            run.tsum = r*run.tsum + (1.-r)*(q**2)
            u += eta*q/(np.sqrt(run.tsum)+1E-20)
        else:
            print 'unimplemented method'
        #elif False: # Adagrad
        #    eta = 1E-3
        #    tsum += q**2
        #    u += eta*q/(np.sqrt(tsum)+1E-20)
        #elif False: #Adam
        #    beta1 = .9
        #    beta2 = .999
        #    m = beta1*m + (1-beta1)*q
        #    v = beta2*v + (1-beta2)*(q**2)
        #    u += c0 * m / (np.sqrt(v) + 1E-20)

        
    #    Danskin's theorem:
    #   if -f_priv is convex in u (it's not) and hat(v) is unique (it's possible),
    #  then dPhi_v(u)/du = -rho*dPhi(u,wh)

    return (u,wv)

#def init(u):
#    run.tsum = np.zeros(u.shape)
    

'''
def run3(u,wv,maxiter,f,dfdu,Phi,dPhidu,rho,\
    X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):

    D,N = X.shape
    maxiter_Phi = 50
    
   
    res = minimize(Phi,u,args=(wv,maxiter_Phi,rho,\
            X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2),\
            method='BFGS', jac=dPhidu, options={'disp':False, 'maxiter':maxiter})    
    u = res.x
    
    return (u,wv)
'''

'''
def run2(u,wv,maxiter,f,dfdu,Phi,Phi_lin,rho,\
    X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2):

    print 'Not working for some reason'
    exit()
    
    D,N = X.shape
    #d = hparams0['d']
    maxiter_Phi = 5
    maxiter_outer = 5
    
    for iter in range(maxiter):
        # 1. Update w, v
        #w -= eta*alg1.dfdv(w,G,y1,hparams1)
        #v -= eta*alg2.dfdv(v,G,y2,hparams2)
        # or fully optimize
        Phi_,wv = Phi(u,wv,maxiter_Phi,rho,\
            X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2)
        
        # 2. Update u
        # min_u  (rho*f_util(u,wh) - f_priv(u,vh))
        # wh = argmin f_util, vh = argmin f_priv are UNIQUE solutions
        res = minimize(f, u, args=(wv,rho,X,y1,y2,filt0,alg1,alg2,hparams0,hparams1,hparams2),\
            method='BFGS', jac=dfdu, options={'disp':False, 'maxiter':maxiter_outer})    
        u = res.x
        #Phiu = -res.fun

    return (u,wv)
'''
