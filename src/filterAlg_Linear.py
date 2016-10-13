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



        
