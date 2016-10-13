# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:36:58 2016

@author: zellinger
"""

from __future__ import print_function

import os
import numpy as np

from models.office_mmatch import NN
from models.domain_regularizer import DomainRegularizer
from utils.office_utils import plot_activations

from keras.preprocessing.image import ImageDataGenerator

EXP_FOLDER = 'experiments/office/'
DATASET_FOLDER = 'utils/office_dataset/'
VGG16_WEIGHTS = 'vgg16_weights.h5'
N_IMAGES_AM = 2817
N_IMAGES_DSLR = 498
N_IMAGES_WC = 795
S_IMAGE = 288
S_BATCH = 2
N_REPETITIONS = 10
CLASS_PLOT = 14

if __name__ == '__main__':
    """ main """
    
    # create folder for model parameters
    if not os.path.exists(EXP_FOLDER):
        print("\nCreating folder "+EXP_FOLDER+"...")
        os.makedirs(EXP_FOLDER)
        
    print("\nLoading office image data...")
    datagen = ImageDataGenerator()
    am_gen = datagen.flow_from_directory(DATASET_FOLDER+'amazon/images',
                                         target_size=(S_IMAGE, S_IMAGE),
                                         batch_size=S_BATCH,
                                         shuffle=False)
    dslr_gen = datagen.flow_from_directory(DATASET_FOLDER+'dslr/images',
                                           target_size=(S_IMAGE, S_IMAGE),
                                           batch_size=S_BATCH,
                                           shuffle=False)
    wc_gen = datagen.flow_from_directory(DATASET_FOLDER+'webcam/images',
                                         target_size=(S_IMAGE, S_IMAGE),
                                         batch_size=S_BATCH,
                                         shuffle=False)
    
    print("\nCreating/Loading image representations via VGG_16 model...")
    nn = NN(EXP_FOLDER)
    x_am, y_am = nn.create_img_repr(DATASET_FOLDER+VGG16_WEIGHTS, am_gen,
                                    'amazon', N_IMAGES_AM)
    x_wc, y_wc = nn.create_img_repr(DATASET_FOLDER+VGG16_WEIGHTS, wc_gen,
                                    'webcam', N_IMAGES_WC)
    x_dslr, y_dslr = nn.create_img_repr(DATASET_FOLDER+VGG16_WEIGHTS, dslr_gen,
                                        'dslr', N_IMAGES_DSLR)    

    print("\nRandom Repetitions...")
    reg = DomainRegularizer(l=1.0, name='mmatch')
    print("wc->dslr:")
    acc_wcdslr = np.array([])
    acc_wcdslr_dr = np.array([])
    for i in range(N_REPETITIONS):
        np.random.seed(i)
        print('--')
        nn_wc = NN(EXP_FOLDER, n_features=256)
        nn_wc_dr = NN(EXP_FOLDER, n_features=256, domain_regularizer=reg)
        nn_wc.fit(x_wc, y_wc, x_dslr)
        nn_wc_dr.fit(x_wc, y_wc, x_dslr)
        acc_tst = nn_wc.evaluate(x_dslr, y_dslr)
        acc_tst_dr = nn_wc_dr.evaluate(x_dslr, y_dslr)
        acc_wcdslr = np.append(acc_wcdslr,acc_tst)
        acc_wcdslr_dr = np.append(acc_wcdslr_dr,acc_tst_dr)
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst=    '+str(acc_tst))
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst-dr= '+str(acc_tst_dr))
    print("\ndslr->wc:")
    acc_dslrwc = np.array([])
    acc_dslrwc_dr = np.array([])
    for i in range(N_REPETITIONS):
        np.random.seed(i)
        print('--')
        nn_dslr = NN(EXP_FOLDER, n_features=256)
        nn_dslr_dr = NN(EXP_FOLDER, n_features=256, domain_regularizer=reg)
        nn_dslr.fit(x_dslr, y_dslr, x_wc)
        nn_dslr_dr.fit(x_dslr, y_dslr, x_wc)
        acc_tst = nn_dslr.evaluate(x_wc, y_wc)
        acc_tst_dr = nn_dslr_dr.evaluate(x_wc, y_wc)
        acc_dslrwc = np.append(acc_dslrwc, acc_tst)
        acc_dslrwc_dr = np.append(acc_dslrwc_dr, acc_tst_dr)
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst=    '+str(acc_tst))
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst-dr= '+str(acc_tst_dr))
    print("\nam->wc:")
    acc_amwc = np.array([])
    acc_amwc_dr = np.array([])
    for i in range(N_REPETITIONS):
        np.random.seed(i)
        print('--')
        nn_am = NN(EXP_FOLDER, n_features=256)
        nn_am_dr = NN(EXP_FOLDER, n_features=256, domain_regularizer=reg)
        nn_am.fit(x_am, y_am, x_wc)
        nn_am_dr.fit(x_am, y_am, x_wc)
        acc_tst = nn_am.evaluate(x_wc, y_wc)
        acc_tst_dr = nn_am_dr.evaluate(x_wc, y_wc)
        acc_amwc = np.append(acc_amwc, acc_tst)
        acc_amwc_dr = np.append(acc_amwc_dr, acc_tst_dr)
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst=    '+str(acc_tst))
        print(str(i+1)+'/'+str(N_REPETITIONS)+' acc-tst-dr= '+str(acc_tst_dr))
     
    print('------------------------------------------------------------------')
    print("am->wc")
    print('SimpleNN acc-tst= '+str(acc_amwc.mean())+'+-'
    +str(acc_amwc.std()))
    print('MMatchNN acc-tst= '+str(acc_amwc_dr.mean())+'+-'
    +str(acc_amwc_dr.std()))
    print("dslr->wc")
    print('SimpleNN acc-tst= '+str(acc_dslrwc.mean())+'+-'
    +str(acc_dslrwc.std()))
    print('MMatchNN acc-tst= '+str(acc_dslrwc_dr.mean())+'+-'
    +str(acc_dslrwc_dr.std()))
    print("wc->dslr")
    print('SimpleNN acc-tst= '+str(acc_wcdslr.mean())+'+-'
    +str(acc_wcdslr.std()))
    print('MMatchNN acc-tst= '+str(acc_wcdslr_dr.mean())+'+-'
    +str(acc_wcdslr_dr.std()))
    print('------------------------------------------------------------------')
    
    print("\nCreate t-SNE grafik...")
    reg = DomainRegularizer(l=0.0, name='mmatch')
    nn_am = NN(EXP_FOLDER, n_features=256, domain_regularizer=reg)
    reg1 = DomainRegularizer(l=1.0, name='mmatch')
    nn_am_dr = NN(EXP_FOLDER, n_features=256, domain_regularizer=reg1)
    nn_am.fit(x_am, y_am, x_wc)
    nn_am_dr.fit(x_am, y_am, x_wc)
    cl_mouse = 14 # class of mouse images
    acc_tst = nn_am.evaluate(x_wc[y_wc.argmax(1)==cl_mouse],
                             y_wc[y_wc.argmax(1)==cl_mouse])
    acc_tst_dr = nn_am_dr.evaluate(x_wc[y_wc.argmax(1)==cl_mouse],
                                   y_wc[y_wc.argmax(1)==cl_mouse])
    print('mouse acc-tst=    '+str(acc_tst))
    print('mouse acc-tst-dr= '+str(acc_tst_dr))
    
    plot_activations(EXP_FOLDER+'tsne_nn', nn_am, x_am, y_am, x_wc, y_wc,
                     lift=True, cl_lift=cl_mouse)
    plot_activations(EXP_FOLDER+'tsne_mmatch', nn_am_dr, x_am, y_am, x_wc,
                     y_wc, lift=True, cl_lift=cl_mouse)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
