# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:36:58 2016

@author: zellinger
"""

from __future__ import print_function

import os
import numpy as np

from utils.amazon_review_utils import split_data, perc_acc
from utils.amazon_review_utils import load_amazon, plot_sensitivity
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

if __name__ == '__main__':
    """ main """
    
    # create folder for model parameters
    if not os.path.exists(EXP_FOLDER):
        print("\nCreating folder "+EXP_FOLDER+"...")
        os.makedirs(EXP_FOLDER)
        
        
    print("\nLoading amazon review data...")
    x, y, offset = load_amazon(N_FEATURES,AMAZON_DATA_FILE)
    domains=['books','dvd','electronics','kitchen']       

    print("\nParameter sensitivity analysis...")
    drws_mmatch = np.linspace(0,3,11)[1:]
    n_moments = np.array([1,2,3,4,5,6,7])
    tasks = np.arange(0,12)
    accs_mmatch = np.zeros((12,drws_mmatch.shape[0],n_moments.shape[0]))
    task_count = 0
    for d_s in range(4):
        for d_t in range(4):
            if d_s==d_t:
                continue
            x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                         = split_data(d_s, d_t, x, y, offset,
                                                      N_TR_SAMPLES)
            for i,drw in enumerate(drws_mmatch):
                for j,beta in enumerate(n_moments):       
                    mmatch_penalty = DomainRegularizer(l=drw, name='mmatchK', beta=int(beta))
                    nn_mmatch = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS, folder=EXP_FOLDER,
                                epoch=MAX_N_EPOCH, bsize=BATCH_SIZE,
                                domain_regularizer=mmatch_penalty, save_weights='mmd')
                    nn_mmatch.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
                    acc = perc_acc(y_t_tst,nn_mmatch.predict(x_t_tst))
                    print('mmatch '+str(task_count)+' '+str(drw)+' '+str(beta)+' : '+str(acc))
                    accs_mmatch[task_count,i,j]=acc
            task_count+=1
    np.save(EXP_FOLDER+'sensitivity_mmatch.npy',accs_mmatch)
    
    drws_mmd = np.linspace(5,45,11)
    betas = np.linspace(0.3,1.7,7)
    tasks = np.arange(0,12)
    accs_mmd = np.zeros((12,drws_mmd.shape[0], betas.shape[0]))
    task_count = 0
    for d_s in range(4):
        for d_t in range(4):
            if d_s==d_t:
                continue
            x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst \
                                         = split_data(d_s, d_t, x, y, offset,
                                                      N_TR_SAMPLES)
            for i,drw in enumerate(drws_mmd):
                for j,beta in enumerate(betas):       
                    mmd_penalty = DomainRegularizer(l=drw, name='mmd', beta=beta)
                    nn_mmd = NN(n_features=N_FEATURES, n_hidden=N_HIDDEN_UNITS, folder=EXP_FOLDER,
                                epoch=MAX_N_EPOCH, bsize=BATCH_SIZE,
                                domain_regularizer=mmd_penalty, save_weights='mmd')
                    nn_mmd.fit(x_s_tr, y_s_tr, x_t_tr, verbose=0)
                    acc = perc_acc(y_t_tst,nn_mmd.predict(x_t_tst))
                    print('mmd '+str(task_count)+' '+str(drw)+' '+str(beta)+' : '+str(acc))
                    accs_mmd[task_count,i,j]=acc
            task_count+=1
    np.save(EXP_FOLDER+'sensitivity_mmd.npy',accs_mmd)
    
    accs_mmatch = np.load(EXP_FOLDER+'sensitivity_mmatch.npy')
    accs_mmd = np.load(EXP_FOLDER+'sensitivity_mmd.npy')
    plot_sensitivity(EXP_FOLDER+'sensitivity', accs_mmd, accs_mmatch, betas, n_moments)