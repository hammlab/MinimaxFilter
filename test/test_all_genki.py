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
rates1_lda,rates2_lda,_ = runTest('lda',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## Minimax - kiwiel
#rates1_kiwiel,rates2_kiwiel,W0_kiwiel = \
#   runTest('kiwiel',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)
## Minimax - alternating
rates1_alt,rates2_alt,W0_alt = \
    runTest('alt',ntrials,ds,ind_train_dom1,ind_test_dom1,X,y1,y2,K1,K2)



##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Figures

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


loaded = np.load('test_NN_genki.npz')
rates1_minimax2=loaded['rates1_minimax2'][0]
rates2_minimax2=loaded['rates2_minimax2'][0]


methods = ('rand proj','pca proj','pls proj','lda proj','ddd','minimax1','minimax2')
methods2 = ('Rand','PCA','PPLS','PLDA','DDD','Minimax1','Minimax2')
cols = ('b','g','r','c','m','c','c')

offset = 0#-0.005

plt.figure(1)
plt.close()
plt.figure(1)
for j in range(len(ds)):
    plt.subplot(3,4,j+1) 
    plt.title('d = %d' % (ds[j]),fontsize=8)
    plt.hold(True)
    
    if not j==0:
        # rand proj
        x = np.nanmean(rates2_rand[j,:])
        y = np.nanmean(rates1_rand[j,:])
        assert np.isnan(x)==False
        assert np.isnan(y)==False
        plt.plot(x,y,'o', markerfacecolor=cols[0],markeredgecolor='k',markersize=2)
        plt.text(x,y+offset,methods2[0],verticalalignment='top',horizontalalignment='left',fontsize=6,color='k')

    # pca proj
    x = np.nanmean(rates2_pca[j,:])
    y = np.nanmean(rates1_pca[j,:])
    assert np.isnan(x)==False
    assert np.isnan(y)==False
    plt.plot(x,y,'o', markerfacecolor=cols[1],markeredgecolor='k',markersize=2)
    plt.text(x,y+offset,methods2[1],verticalalignment='bottom',horizontalalignment='left',fontsize=6,color='k')

    # pls proj
    x = np.nanmean(rates2_pls[j,:])
    y = np.nanmean(rates1_pls[j,:])
    assert np.isnan(x)==False
    assert np.isnan(y)==False
    plt.plot(x,y,'o', markerfacecolor=cols[2],markeredgecolor='k',markersize=2)
    plt.text(x,y+offset,methods2[2],verticalalignment='bottom',horizontalalignment='left',fontsize=6,color='k')

    #%% ddd proj
    x = np.nanmean(rates2_ddd)
    y = np.nanmean(rates1_ddd)
    assert np.isnan(x)==False
    assert np.isnan(y)==False
    plt.plot(x,y,'o', markerfacecolor=cols[4],markeredgecolor='k',markersize=2)
    plt.text(x,y+offset,methods2[4],verticalalignment='top',horizontalalignment='left',fontsize=6,color='k')
    
    # minimax
    x = np.nanmean(rates2_alt[j,:])
    y = np.nanmean(rates1_alt[j,:])
    assert np.isnan(x)==False
    assert np.isnan(y)==False
    plt.plot(x,y,'o', markerfacecolor=cols[5],markeredgecolor='k',markersize=2)
    plt.text(x,y+offset,methods2[5],verticalalignment='top',horizontalalignment='center',fontsize=6,color='k')

    # minimax2
    x = np.nanmean(rates2_minimax2[0,0])
    y = np.nanmean(rates1_minimax2[0,0])
    assert np.isnan(x)==False
    assert np.isnan(y)==False
    plt.plot(x,y,'o', markerfacecolor=cols[6],markeredgecolor='k',markersize=2)
    plt.text(x,y+offset,methods2[6],verticalalignment='top',horizontalalignment='center',fontsize=6,color='k')

    # grid lines
    for c in np.arange(-1.,1.,.1):
        x0 = -1.
        y0 = x0 + c
        x1 = 2.
        y1 = x1 + c
        plt.plot((x0,x1),(y0,y1),'k:',linewidth=.4)

    
    plt.xlim(.4, 1.)
    plt.ylim(.75, .9)
    plt.tick_params(labelsize=5)    
    if j==0:
        plt.xlabel('Gender recognition accuracy',fontsize=8)
        plt.ylabel('Expression recognition accuracy',fontsize=8)
    
#%set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[]);
plt.show(block=False)        



with PdfPages('test_all_genki.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')






