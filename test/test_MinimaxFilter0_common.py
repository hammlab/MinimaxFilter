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

import privacyLDA
import privacyPLS

from filterAlg_Linear import Linear
from learningAlg import mlogreg 
import minimaxFilter


#########################################################################################################

def train_and_test(G,y1,y2,ind_train,ind_test,maxiter_final,hparams1,hparams2):
    tW1,f1 = mlogreg.train(G[:,ind_train].squeeze(),y1[:,ind_train].squeeze(),hparams1,None,maxiter_final)
    tW2,f2 = mlogreg.train(G[:,ind_train].squeeze(),y2[:,ind_train].squeeze(),hparams2,None,maxiter_final)

    rate1,_ = mlogreg.accuracy(tW1,G[:,ind_test].squeeze(),y1[:,ind_test].squeeze())
    rate2,_ = mlogreg.accuracy(tW2,G[:,ind_test].squeeze(),y2[:,ind_test].squeeze())
    
    return (rate1,rate2)




def runTest(method,ntrials,ds,ind_train,ind_test,X,y1,y2,K1,K2,\
    maxiter_final=50,rho=10.,maxiter_minimax=10,lambda0=1E-6,lambda1=1E-6,lambda2=1E-6):

    # method = 'rand' | 'pca' | 'pls' | 'lda' | 'kiwiel' | 'alt'    
    
    if (method=='kiwiel' or method=='alt'):
        print 'Minimax'
        
        rates1 = np.nan*np.ones((len(ds),ntrials))
        rates2 = np.nan*np.ones((len(ds),ntrials))
        W0s = [[[] for i in range(ntrials)] for j in range(len(ds))]
        
        #maxiter = 5
        for trial in range(ntrials):
            #% Init by LDA
            U,dd = privacyLDA.run(X[:,ind_train[trial][0]].squeeze(),y1[:,ind_train[trial][0]].squeeze(),y2[:,ind_train[trial][0]].squeeze())
            for j in range(len(ds)):

                d = ds[j]
                W0 = U[:,0:d]
                W1 = np.zeros((d,K1))
                W2 = np.zeros((d,K2))
    
                hparams0 = {'d':d, 'l':lambda0}
                hparams1 = {'K':K1, 'l':lambda1}
                hparams2 = {'K':K2,'l':lambda2}
    
                for iter in range(maxiter_minimax):
                    if True:#iter==maxiter_minimax-1:
                        
                        G = Linear.g(W0,X,hparams0)
                        r1,r2 = train_and_test(G,y1,y2,ind_train[trial][0],ind_test[trial][0],maxiter_final,hparams1,hparams2)
                        print '%s: rho=%f, d=%d, trial=%d, rate1=%f, rate2=%f\n' % \
                            (method, rho,d,trial,r1,r2)
    
                        rates1[j,trial] = r1
                        rates2[j,trial] = r2
                        W0s[j][trial] = W0
                    
                    W0,W1,W2 = minimaxFilter.run(W0,W1,W2,rho,method,1,\
                        X[:,ind_train[trial][0]].squeeze(), \
                        y1[:,ind_train[trial][0]].squeeze(),\
                        y2[:,ind_train[trial][0]].squeeze(),\
                        Linear,mlogreg,mlogreg,\
                        hparams0,hparams1,hparams2)
                        
        return (rates1,rates2,W0s)
        

    elif (method=='rand' or method=='pca' or method=='pls' or method=='lda' ):

        rates1 = np.nan*np.ones((len(ds),ntrials))
        rates2 = np.nan*np.ones((len(ds),ntrials))
        W0s = [[[] for i in range(ntrials)] for j in range(len(ds))]

        D = X.shape[0]
        
        for trial in range(ntrials):
            if method=='rand': # Random full-rank matrix
                U,S,V = np.linalg.svd(np.random.normal(size=(D,D)))
            elif method=='pca': # SVD
                U,S,V = np.linalg.svd(X[:,ind_train[trial][0]].squeeze())
            elif method=='pls':
                U = privacyPLS.run(X[:,ind_train[trial][0]].squeeze(),y1[:,ind_train[trial][0]].squeeze(),y2[:,ind_train[trial][0]].squeeze(),ds[-1])
            elif method=='lda':
                U,dd = privacyLDA.run(X[:,ind_train[trial][0]].squeeze(),y1[:,ind_train[trial][0]].squeeze(),y2[:,ind_train[trial][0]].squeeze())
                
            for j in range(len(ds)):
                d = ds[j]
                W0 = U[:,:d]
                W1 = np.zeros((d,K1))
                W2 = np.zeros((d,K2))
                
                hparams0 = {'d':d, 'l':lambda0}
                hparams1 = {'K':K1, 'l':lambda1}
                hparams2 = {'K':K2,'l':lambda2}
                
                G = Linear.g(W0,X,hparams0)
                r1,r2 = train_and_test(G,y1,y2,ind_train[trial][0],ind_test[trial][0],maxiter_final,hparams1,hparams2)
                print '%s: d=%d, trial=%d, rate1=%f, rate2=%f\n' % \
                    (method,d,trial,r1,r2)

                rates1[j,trial] = r1
                rates2[j,trial] = r2
                W0s[j][trial] = W0

        return (rates1,rates2,W0s)

    else:
        
        print 'Unknown method: ', method
        exit(0)


