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
import matplotlib.pyplot as plt
import scipy.linalg
#from scipy.optimize import minimize

def run(X,y1,y2):
    # function [E,dd] = privacyLDA(X,y1,y2)
    # %max_W0 tr(W0'*C1*W0) - tr(W0'*C2*W0)
    D,N = X.shape
    y1_unique = np.unique(y1)
    #print y1_unique
    #[y1_unique,~,J1] = unique(y1);
    #ny1 = y1_unique.size

    y2_unique = np.unique(y2)
    #print y2_unique
    #[y2_unique,~,J2] = unique(y2);
    #ny2 = y2_unique.size

    C1 = np.zeros((D,D))
    C2 = np.zeros((D,D))
    mu = X.mean(axis=1).reshape((D,1))
    #print mu.shape

    for k in np.nditer(y1_unique):
        indk = np.where(y1==k)[0]        
        muk = X[:,indk].mean(axis=1).reshape((D,1))
        #muk -= np.kron(np.ones((1,len(indk))),mu)
        #%C1 = C1 + ny1*(muk-mu)*(muk-mu)';
        C1 = C1 + len(indk)*np.dot(muk-mu,(muk-mu).T)
    
    for k in np.nditer(y2_unique):
        indk = np.where(y2==k)[0]        
        muk = X[:,indk].mean(axis=1).reshape((D,1))
        #muk -= np.kron(np.ones((1,len(indk))),mu)
        #%C1 = C1 + ny1*(muk-mu)*(muk-mu)';
        C2 = C2 + len(indk)*np.dot(muk-mu,(muk-mu).T)
    
    C1 = C1 + 1e-8*np.trace(C1)*np.eye(D)# 
    C2 = C2 + 1e-8*np.trace(C2)*np.eye(D)#
    C1 = 0.5*(C1+C1.T)#;% + 1E-8*trace(C1)*eye(D); 
    C2 = 0.5*(C2+C2.T)#;% + 1E-8*trace(C2)*eye(D);
    

    dd,E = scipy.linalg.eigh(C1,C2) # ascending order
    
    #print dd.shape
    #print E.shape
    
    assert np.isnan(dd).any()==False
    assert np.iscomplex(dd).any()==False
    #[dd,ind] = sort(diag(dd),'descend'); 
    #print dd
    dd = dd[::-1] #
    E = E[:,::-1]
    
    E = E/np.tile(np.sqrt((E**2).sum(axis=0,keepdims=True)),(D,1))
    #% They need not be orthogonal.

    #print dd.shape
    #print E.shape

    return (E,dd)
    
    
def selftest1():
    
    # Generate data
    D0 = 5
    K1 = 2
    K2 = 2
    NperClass = 500
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
                np.random.normal(scale=0.25, size=(D0,NperClass)) \
                + np.kron(np.ones((1,NperClass)),bias1[:,k1].reshape((D0,1))) \
                + np.kron(np.ones((1,NperClass)),bias2[:,k2].reshape((D0,1)))
            y1[:,k1,k2] = k1*np.ones((NperClass,))
            y2[:,k1,k2] = k2*np.ones((NperClass,))

    X = X.reshape((D0,N))
    y1 = y1.reshape((N,))
    y2 = y2.reshape((N,))
    
    E,dd = run(X,y1,y2)
    print dd

    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(dd)
    plt.subplot(1,2,2)
    plt.imshow(E, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()
