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
from scipy.optimize import minimize
#from scipy.optimize import check_grad



class LearningAlg:
    #def __init__(self):
    #    self.data = []

    '''
    def init
    
    def train(G,y,hparams,W0=None,maxiter=300):

        return (W, val)

    def accuracy(W,Gtest,ytest):

        return (rate, ncorrect)

    def test(W,G):
        
        return (ypred,pyx)
        

    def f(W,G,y,hparams):

        return fval
        
    def dfdv(W,G,y,hparams):

    def dfdu(W,G,y,dgdu,hparams): # u.size x v.size = D*d x d*k

    def flin(v,q,G,y,dgdu,hparams):
        # f_(v;xk,v) = f(xk,v) + dfdu(xk,v)'*q.
        
        return fval

    def dflindv(v,q,G,y,dgdu,hparams):
        # df_/dy(v;xk,v) = df/dy(xk,v) + d2f/dxdy(xk,v)'q
        df = dfdv + np.dot(d2fdudv.T,q)
        return df
        

    def f_neg(v,G,y,hparams):
        return -f(v,G,y,hparams)

    def dfdv_neg(v,G,y,hparams):
        return -dfdv(v,G,y,hparams)

    def flin_neg(v,q,G,y,dgdu,hparams):
        return -flin(v,q,G,y,dgdu,hparams)
        
    def dflindv_neg(v,q,G,y,dgdu,hparams):
        return -.dflindv(v,q,G,y,dgdu,hparams)
    
    # private methods
    
    def _d2fdudv(W,G,y,dgdu,hparams): # u.size x v.size = D*d x d*k
        return d2f

    def _dldg(W,G,y,hparams): # d x N
        # f1(u,w) = 1/N sum_i l(g(x_i;u);w) #+ l1*||w||^2
        return dl # d x N

    def _d2ldgdv(W,G,y,hparams): # d x N x v.size = d x N x d*K
        return d2l 

    '''

    
########################################################################################

class mlogreg(LearningAlg):
    
    
    @staticmethod        
    def init(hparams):
        K = hparams['K']        
        d = hparams['d']
        # random normal
        w = np.random.normal(size=(d*K,))

        return w
    
    @staticmethod    
    def train(G,y,hparams,W0=None,maxiter=300):
        K = hparams['K']
        #l = hparams['l']
        d,N = G.shape
        if W0==None :
            W = np.random.normal(size=(d*K,))
        else:
            W = W0.flatten()
            
        res = minimize(mlogreg.f, W, args=(G,y,hparams),\
            method='BFGS',jac=mlogreg.dfdv, options={'disp':False, 'maxiter':maxiter})    
        
        W = res.x.reshape((d,K))
        return (W, res.fun)


    @staticmethod
    def accuracy(W,Gtest,ytest):
        
        ypred,_ = mlogreg.test(W,Gtest)
        ind_correct = np.where(ypred==ytest)[0]    
        ncorrect = ind_correct.size
        rate = float(ncorrect) / float(ypred.size)
        return (rate, ncorrect)

        
    @staticmethod
    def test(W,G):
        
        d,K = W.shape
        d,N = G.shape
    
        #ypred = np.zeros((N,),dtype=int)    
        #pyx = np.zeros((K,N))
        
        WG = np.dot(W.T,G)
        #expWG = np.exp(WG) # K x N
        expWG = np.exp(WG-np.tile(WG.max(axis=0,keepdims=True),(K,1)))
        sumexpWG = expWG.sum(axis=0,keepdims=True) #% 1 x N
        
        pyx = expWG / np.tile(sumexpWG,(K,1))     
        ypred = np.argmax(pyx, axis=0)
    
        return (ypred,pyx)
        

    @staticmethod
    def f(W,G,y,hparams):
        # f = -1/N*sum_t log(exp(w(yt)'gt)/sum_k exp(wk'gt)) + l*||W||
        # = -1/N*sum_t [w(yt)'*gt - log(sum_k exp(wk'gt))] + l*||W||
        # = -1/N*sum(sum(W(:,y).*G,1),2) + 1/N*sum(log(sumexpWG),2) + l*sum(sum(W.^2));

        #K,l = hparams
        K = hparams['K']
        l = hparams['l']
        d,N = G.shape
        W = W.reshape((d,K))
        
        WG = np.dot(W.T,G) # K x N
        WG -= np.kron(np.ones((K,1)),WG.max(axis=0).reshape(1,N))
        #WG_max = WG.max(axis=0).reshape((1,N))
        #expWG = np.exp(WG-np.kron(np.ones((K,1)),WG_max)) # K x N
        expWG = np.exp(WG) # K x N
        sumexpWG = expWG.sum(axis=0) # N x 1
        WyG = WG[y,range(N)]
        #WyG -= WG_max
        
        fval = -1.0/N*(WyG).sum() \
            + 1.0/N*np.log(sumexpWG).sum() \
            + l*(W**2).sum()#(axis=(0,1))
    
        return fval
        

    @staticmethod
    def dfdv(W,G,y,hparams):
        # df/dwk = -1/N*sum(x(:,y==k),2) + 1/N*sum_t exp(wk'xt)*xt/(sum_k exp(wk'xt))] + l*2*wk
        K = hparams['K']
        l = hparams['l']
        d,N = G.shape
        shapeW = W.shape
        W = W.reshape((d,K))
        
        WG = np.dot(W.T,G) # K x N
        WG -= np.kron(np.ones((K,1)),WG.max(axis=0).reshape(1,N))
        expWG = np.exp(WG) # K x N
        sumexpWG = expWG.sum(axis=0) # N x 1
        df = np.zeros((d,K))
        for k in range(K):
            indk = np.where(y==k)[0]    
            df[:,k] = -1./N*G[:,indk].sum(axis=1).reshape((d,)) \
                + 1./N*np.dot(G,(expWG[k,:]/sumexpWG).T).reshape((d,)) \
                + 2.*l*W[:,k].reshape((d,))
        
        assert np.isnan(df).any()==False        
        
        return df.reshape(shapeW)



    @staticmethod
    def dfdu(W,G,y,dgdu,hparams): # u.size
        # d2fdudv = d/dv (df/du)
        # = d/dv (1/N*sum(dli/dgi*dgidu))
        # = 1/N*sum (d2li/dgidv*dgidu )
        # dgdu : u.size x d x N
        # d2li/dgidv = d x N x v.size
        #K = hparams['K']
        #Dd = dgdu.shape
        d,N = G.shape
        df = 1./N*np.dot(dgdu, mlogreg._dldg(W,G,y,hparams).reshape((d*N,)))
        assert np.isnan(df).any()==False
        
        return df



    
    @staticmethod    
    def flin(v,q,G,y,dgdu,hparams):
        #K = hparams['K']
        #l = hparams['l']
        d,N = G.shape
        # f_(v;xk,v) = f(xk,v) + dfdu(xk,v)'*q.
        dfdu = 1./N*np.dot(dgdu,mlogreg._dldg(v,G,y,hparams).reshape((d*N,))) 
        
        fval_ = mlogreg.f(v,G,y,hparams) + np.dot(dfdu.T,q)
        
        assert np.isnan(fval_)==False
        
        return fval_


    @staticmethod    
    def dflindv(v,q,G,y,dgdu,hparams):
        #K = hparams['K']
        #l = hparams['l']
        d,N = G.shape
        
        # df_/dy(v;xk,v) = df/dy(xk,v) + d2f/dxdy(xk,v)'q
        dfdv = mlogreg.dfdv(v,G,y,hparams)
        d2fdudv = mlogreg._d2fdudv(v,G,y,dgdu,hparams)  # u.size x v.size = D*d x d*k

        df = dfdv + np.dot(d2fdudv.T,q)

        assert np.isnan(df).any()==False
        
        return df
        
    
    @staticmethod    
    def f_neg(v,G,y,hparams):
        return -mlogreg.f(v,G,y,hparams)


    @staticmethod    
    def dfdv_neg(v,G,y,hparams):
        return -mlogreg.dfdv(v,G,y,hparams)


    @staticmethod    
    def flin_neg(v,q,G,y,dgdu,hparams):
        return -mlogreg.flin(v,q,G,y,dgdu,hparams)
        

    @staticmethod    
    def dflindv_neg(v,q,G,y,dgdu,hparams):
        return -mlogreg.dflindv(v,q,G,y,dgdu,hparams)
    
###################################################################################
    
    @staticmethod
    def _d2fdudv(W,G,y,dgdu,hparams): # u.size x v.size = D*d x d*k
        # d2fdudv = d/dv (df/du)
        # = d/dv (1/N*sum(dli/dgi*dgidu))
        # = 1/N*sum (d2li/dgidv*dgidu )
        # dgdu : u.size x d x N
        # d2li/dgidv = d x N x v.size
        K = hparams['K']
        #Dd = dgdu.shape
        d,N = G.shape
        d2f = 1./N*np.dot(dgdu, mlogreg._d2ldgdv(W,G,y,hparams).reshape((d*N,d*K)))
        assert np.isnan(d2f).any()==False
        
        return d2f


    @staticmethod    
    def _dldg(W,G,y,hparams): # d x N
        # f1(u,w) = 1/N sum_i l(g(x_i;u);w) #+ l1*||w||^2
        # df1du = 1/N sum_i dldg*dgdu
    
        # f = -1/N*sum_t log(exp(w(yt)'gt)/sum_k exp(wk'gt)) + l*||W||^2
        # = -1/N*sum_t [w(yt)'*gt - log(sum_k exp(wk'gt))] + l*||W||^2
        # = -1/N*sum(sum(W(:,y).*G,1),2) + 1/N*sum(log(sumexpWG),2) + l*sum(sum(W.^2));

        K = hparams['K']
        #l = hparams['l']
        d,N = G.shape
        #shapeW = W.shape
        W = W.reshape((d,K))

        # l = -log(exp(w(yi)'gi)/sum_k exp(wk'gi)) + lamb*||W||^2
        #   = -w(yi)'*gi + log(sum_k exp(wk'*gi)) + lamb*||W||^2
        # dldg = -w(yi) + sum_k wk*exp(wk'*gi) / (sum_k exp(wk'*gi))

        WG = np.dot(W.T,G)
        #expWG = np.exp(WG) # K x N
        expWG = np.exp(WG-np.tile(WG.max(axis=0,keepdims=True),(K,1)))
        sumexpWG = expWG.sum(axis=0,keepdims=True) #% 1 x N
        sumWexpWG = np.dot(W,expWG) # dxK x KxN = d x N
        
        dl = -W[:,y] + sumWexpWG/np.tile(sumexpWG,(d,1))

        assert np.isnan(dl).any()==False

        return dl # d x N


    @staticmethod    
    def _d2ldgdv(W,G,y,hparams): # d x N x v.size = d x N x d*K

        K = hparams['K']
        #l = hparams['l']
        d,N = G.shape
        W = W.reshape((d,K))

        # l = -w(y)'*g + log(sum_l exp(wl'*g)) + l*||W||^2
        # dldg = -w(y) + sum_l wl*exp(wl'*g)) / sum_l exp(wl'*g)
        #      = -w(y) + sum_l wl*al, where al = exp(wl'*g)) / sum_l exp(wl'*g)
        # dal/dwk = I[l==k]*g*exp(wl'g)/sum - g*exp(wl'g)exp(wk'g)/sum^2  (=g*al*ak)
        #         = I[l==k]*g*al - g*al*ak
        # d2ldgdwk = -eye*I[y==k] + sum_l [eye*I[l==k]*al + wl*dal/dwk]
        #          = -eye*I[y==k] + eye*ak + sum_l wl*dal/dwk
        #          = -eye*I[y==k] + eye*ak + g*wk'*ak - g*ak*sum_l wl'*al

        # Compare with d2ldwkdg:
        # l = -w(y)'*g + log(sum_l exp(wl'*g)) + l*||W||^2
        # dldwk = -g*eye*I[y==k] + g*exp(wk'*g) /sum_l exp() + 2*l*wk
        #       = -g*eye*I[y==k] + g*ak + 2*l*wk
        # dak/dg = wk*ak - exp(wk'g)*sum_l wl*exp(wl'g)/sum^2
        #        = wk*ak - ak*sum_l wl*al
        # d2ldwkdg = -eye*I[y==k] + eye*ak + g*dak/dg 
        #          = -eye*I[y==k] + eye*ak + g*(wk'*ak - ak*sum_l wl*al)


        WG = np.dot(W.T,G) # K x N
        expWG = np.exp(WG-np.tile(WG.max(axis=0,keepdims=True),(K,1)))
        sumexpWG = expWG.sum(axis=0) #% 1 x N
        A = expWG/np.tile(sumexpWG,(K,1)) # K x N
        #sumWexpWG = np.dot(W,expWG) #% d x N
        sumWA = np.dot(W,A) #% d x N
            
        d2l = np.zeros((d,N,d,K))
        for i in range(N):
            for k in range(K):
                d2l[:,i,:,k] = -np.eye(d)*(np.float(y[i]==k)-A[k,i]) \
                    + np.outer(G[:,i],W[:,k]-sumWA[:,i])*A[k,i] 
                    
        '''
        for k in range(K):
            d2l[:,:,:,k] = -np.asarray(y==k,np.double) \
                + np.tile(np.eye(d).reshape((d,1,d)),(1,N,1))*np.tile(normexpWG[k,:].reshape((1,N,1)),(d,1,d)) \
                + np.tile(np.outer(G,W[:,k]).reshape((d,1,d)),(1,N,1))*np.tile(normexpWG[k,:].reshape((1,N,1)),(d,1,d)) \
                - np.tile(np.outer(G,W[:,k]).reshape((d,1,d)),(1,N,1))*np.tile((normexpWG[k,:]**2).reshape((1,N,1)),(d,1,d)) \
        '''

        assert np.isnan(d2l).any()==False        

        return d2l 
