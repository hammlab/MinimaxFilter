# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 21:58:27 2016

@author: hammj
"""

import scipy.io
import numpy as np

from test_MinimaxFilter0_common import runTest



mat = scipy.io.loadmat('genki.mat')
#nsubjs = np.asscalar(mat['nsubjs'])
K1 = np.asscalar(mat['K1'])
K2 = np.asscalar(mat['K2'])

D = np.asscalar(mat['D'])
Ntrain = np.asscalar(mat['Ntrain'])
Ntest = np.asscalar(mat['Ntest'])
N = Ntrain + Ntest
y1_train = mat['y1_train']-1
y1_test = mat['y1_test']-1
y1 = np.hstack((y1_train,y1_test))
del y1_train, y1_test
y2_train = mat['y2_train']-1
y2_test = mat['y2_test']-1
y2 = np.hstack((y2_train,y2_test))
del y2_train, y2_test

Xtrain = mat['Xtrain']
Xtest = mat['Xtest']
X = np.hstack((Xtrain,Xtest))
del Xtrain, Xtest

ind_train_dom1 = [[range(Ntrain)]]
ind_test_dom1 = [[range(Ntrain,Ntrain+Ntest)]]

rates1_ddd = mat['rates1_ddd_dom1'][0][0]
rates2_ddd = mat['rates2_ddd_dom1'][0][0]

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntrials = 1
ds = [10]

## Random (orthogoal) projection
rates1_rand,rates2_rand,_ = runTest('rand',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## PCA init
rates1_pca,rates2_pca,_ = runTest('pca',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## PLS init
rates1_pls,rates2_pls,_ = runTest('pls',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## LDA init
#rates1_lda,rates2_lda,_ = runTest('lda',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## Minimax - kiwiel
#rates1_kiwiel,rates2_kiwiel,W0_kiwiel = \
#   runTest('kiwiel',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## Minimax - alternating
rates1_alt,rates2_alt,W0_alt = \
    runTest('alt',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)


np.savez('test_all_genki.npz',\
    W0_alt=[W0_alt], \
    rates1_rand=[rates1_rand],\
    rates2_rand=[rates2_rand],\
    rates1_pca=[rates1_pca],\
    rates2_pca=[rates2_pca],\
    rates1_pls=[rates1_pls],\
    rates2_pls=[rates2_pls],\
    rates1_alt=[rates1_alt],\
    rates2_alt=[rates2_alt]\
    )

'''
rand: d=10, trial=0, rate1=0.705000, rate2=0.705000

pca: d=10, trial=0, rate1=0.840000, rate2=0.665000

pls: d=10, trial=0, rate1=0.850000, rate2=0.685000

alt: rho=10.000000, d=10, trial=0, rate1=0.825000, rate2=0.520000

'''

