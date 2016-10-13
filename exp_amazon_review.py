# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:36:58 2016

@author: zellinger
"""

from __future__ import print_function

import os
import numpy as np

from utils.amazon_review_utils import split_data, perc_acc, plot_activations
from utils.amazon_review_utils import reverse_validation
from utils.amazon_review_utils import make_final_table, load_amazon
from models.amazon_review_mmatch import NN
from models.domain_regularizer import DomainRegularizer

N_FEATURES = 5000
N_TR_SAMPLES = 2000
N_HIDDEN_UNITS = 50
EXP_FOLDER = 'experiments/amazon_review/'
AMAZON_DATA_FILE = 'utils/amazon_dataset/amazon.mat'
MAX_N_EPOCH = 1500
BATCH_SIZE = 300
IND_REP = range(10)
INIT_WEIGHTS_PRE_NAME = 'init_weights_230916_'
FINAL_TABLW_NAME = 'accs_amazon_reviews'

if __name__ == '__main__':
    """ main """
    
    # create folder for model parameters
    if not os.path.exists(EXP_FOLDER):
        print("\nCreating folder "+EXP_FOLDER+"...")
        os.makedirs(EXP_FOLDER)
        
        
    print("\nLoading amazon review data...")
    x, y, offset = load_amazon(N_FEATURES,AMAZON_DATA_FILE)
    domains=['books','dvd','electronics','kitchen']   
    
    print("\nRunning big evaluation...")
    # Can run some days
    accs_big_eval = np.zeros((12,3,len(IND_REP)))
    for rnd_eval in IND_REP:
        init_weights_file_name = INIT_WEIGHTS_PRE_NAME+str(rnd_eval)
        rnd_seed = rnd_eval
        
        print("\nTraining three test models (SimpleNN, NNmmd, NNMmatch)...")
        # Split data for domains books->kitchen
        x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                     = split_data(0, 3, x, y, offset, N_TR_SAMPLES,
                                                  rnd_seed)
                                     
        n_hiddens = 5
        # SimpleNN
        simplenn = NN(n_features=N_FEATURES, n_hidden=n_hiddens, folder=EXP_FOLDER,
                      epoch=MAX_N_EPOCH, bsize=BATCH_SIZE, save_weights='nn')
        simplenn.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
        # NN w/ MMD penalty
        mmd_penalty = DomainRegularizer(l=10, name='mmd', beta=0.3)
        nn_mmd = NN(n_features=N_FEATURES, n_hidden=n_hiddens, folder=EXP_FOLDER,
                      epoch=MAX_N_EPOCH, bsize=BATCH_SIZE,
                      domain_regularizer=mmd_penalty, save_weights='mmd')
        nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
        # NN w/ Mmatch penalty
        mmatch_penalty = DomainRegularizer(l=1, name='mmatch')
        nn_mmatch = NN(n_features=N_FEATURES, n_hidden=n_hiddens, folder=EXP_FOLDER,
                      epoch=MAX_N_EPOCH, bsize=BATCH_SIZE,
                      domain_regularizer=mmatch_penalty,save_weights='mmatch')
        nn_mmatch.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
        
        print("\nCalculating accuracies...")
        print('SimpleNN: '+str(perc_acc(y_t_tst,simplenn.predict(x_t_tst))))
        print('NNmmd   : '+str(perc_acc(y_t_tst,nn_mmd.predict(x_t_tst))))
        print('NNMmatch: '+str(perc_acc(y_t_tst,nn_mmatch.predict(x_t_tst))))         
        
        print("\nPlotting hidden activations...")
        a_nn_s, a_nn_t = simplenn.get_activations(x_s_tst,x_t_tst)
        plot_activations(a_nn_s,a_nn_t,EXP_FOLDER+'act_nn','NN activations w/o regularization')
        a_mmd_s, a_mmd_t = nn_mmd.get_activations(x_s_tst,x_t_tst)
        plot_activations(a_mmd_s,a_mmd_t,EXP_FOLDER+'act_mmd','NN activations w/ mmd-penalty')
        a_mmatch_s, a_mmatch_t = nn_mmatch.get_activations(x_s_tst,x_t_tst)
        plot_activations(a_mmatch_s,a_mmatch_t,EXP_FOLDER+'act_mmatch','NN activations w/ Mmatch')
        print('Generated graphics in '+EXP_FOLDER)    
    
        print("\nCreating initial model weights...")
        # This is important for every model to start at same initial situation
        # Overwrites INIT_WEIGHTS_FILE_NAME!
        # Create different initial weights by setting RANDOM_SEED for split_data
        # function before the weights creation.
        # Alternatively use the pre-computed weight-files (see README.txt).
        nn = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS, folder=EXP_FOLDER,
                epoch=MAX_N_EPOCH, bsize=BATCH_SIZE)
        x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                 = split_data(0, 3, x, y, offset, N_TR_SAMPLES,
                                              rnd_seed)
        nn.create_initial_weights(x_s_tr, y_s_tr, x_t_tr, init_weights_file_name)
    
    
        print("\nStart evaluation inclusive grid-search for mmd (can run some hours)...")
        # Setup
        accuracies_mmd = np.zeros((4,4))
        accuracies_nn = np.zeros((4,4))
        accuracies_mmatch = np.zeros((4,4))
        settings_mmd = np.chararray((4,4),itemsize=100)
        settings_mmd[:] = 'empty'
        for d_s_ind,dom_s in enumerate(domains):
            for d_t_ind,dom_t in enumerate(domains):
                if dom_s==dom_t:
                    continue
                
                print('\nSource domain:'+str(dom_s))
                print('Target domain:'+str(dom_t))
                
                # Split data
                x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst = \
                split_data(d_s_ind,d_t_ind,x,y,offset,N_TR_SAMPLES)
                data = [x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst]
                
                # SimpleNN
                simplenn = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS,
                              folder=EXP_FOLDER, epoch=MAX_N_EPOCH,
                              bsize=BATCH_SIZE, save_weights='nn')
                simplenn.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights=init_weights_file_name)
                accuracies_nn[d_s_ind,d_t_ind] = perc_acc(y_t_tst,
                                                          simplenn.predict(x_t_tst))
                print('acc: '+str(accuracies_nn[d_s_ind,d_t_ind]))
                
                # NN w/ MMD-penalty
                # Grid Search for model parameters of mmd (main running time)
                # via reverse validation (unsupervised hyperparameter selection for
                # transfer learning algorithms, see paper)
                drws = np.logspace(np.log10(0.1),np.log10(500),num=10).tolist()# domain regularizer weights
                betas= np.logspace(np.log10(0.01),np.log10(10),num=10).tolist()# gaussian kernel betas
                accs_mmd = np.zeros((len(drws),len(betas)))            
                for i,weight in enumerate(drws):
                    for j,beta in enumerate(betas):
                        # info
                        print('mmd '+dom_s+'->'+dom_t+' step: '+str(len(betas)*i+j+1)
                        +'/'+str(len(drws)*len(betas))+' dr-weight: '+str(weight)
                        +' ('+str(i+1)+'/'+str(len(drws))+') beta: '+str(beta)
                        +' ('+str(j+1)+'/'+str(len(betas))+')')
                        
                        # define models
                        mmd_penalty = DomainRegularizer(l=weight, name='mmd',
                                                        beta=beta)
                        nn_mmd = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS,
                                    folder=EXP_FOLDER, epoch=MAX_N_EPOCH,
                                    bsize=BATCH_SIZE, domain_regularizer=mmd_penalty,
                                    save_weights='mmd')
                        S=(x_s_tr,y_s_tr)
                        T=x_t_tr
                        # grid search for mmd via reverse validation (see paper)
                        accs_mmd[i,j] = reverse_validation(nn_mmd, init_weights_file_name,
                                                           S, T)
                # Find best mmd setting
                [i,j]=np.unravel_index(accs_mmd.argmax(),accs_mmd.shape)
                # Train best nn w/ mmd model
                mmd_penalty = DomainRegularizer(l=drws[i], name='mmd',
                                                beta=betas[j])
                nn_mmd = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS,
                            folder=EXP_FOLDER, epoch=MAX_N_EPOCH,
                            bsize=BATCH_SIZE, domain_regularizer=mmd_penalty,
                            save_weights='mmd_final')
                nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0, init_weights=init_weights_file_name)
                accuracies_mmd[d_s_ind,d_t_ind] = perc_acc(y_t_tst,
                                                           nn_mmd.predict(x_t_tst))
                settings_mmd[d_s_ind,d_t_ind] = 'domreg-weight: ' + str(drws[i]) \
                                                + 'beta: ' + str(betas[j])
                print('acc: '+str(accuracies_mmd[d_s_ind,d_t_ind]))
                
                # NN w/ moment matching
                mmatch_penalty = DomainRegularizer(l=1, name='mmatch')
                nn_mmatch = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS,
                               folder=EXP_FOLDER, epoch=MAX_N_EPOCH,
                               bsize=BATCH_SIZE, domain_regularizer=mmatch_penalty,
                               save_weights='mmatch')
                nn_mmatch.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0,
                              init_weights=init_weights_file_name)
                accuracies_mmatch[d_s_ind,d_t_ind] = perc_acc(y_t_tst,
                                                              nn_mmatch.predict(x_t_tst))
                print('acc: '+str(accuracies_mmatch[d_s_ind,d_t_ind]))
                                    
        accs_big_eval[:,0,rnd_eval] = np.delete(accuracies_nn.ravel(),[0,5,10,15])
        accs_big_eval[:,1,rnd_eval] = np.delete(accuracies_mmd.ravel(),[0,5,10,15])
        accs_big_eval[:,2,rnd_eval] = np.delete(accuracies_mmatch.ravel(),[0,5,10,15])
        
    make_final_table(EXP_FOLDER+FINAL_TABLW_NAME, accs_big_eval)