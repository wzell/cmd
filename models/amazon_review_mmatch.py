# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 2016

@author: zellinger
"""

import numpy as np
import datetime

np.random.seed(0)

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, SGD
from keras.layers import Dense, Input
from keras.models import Model

# root folder of experiments
# model parameters, evaluated accuracies, etc. get dumped here
EXP_FOLDER = 'experiments/amazon_review'
MAX_N_EPOCH = 1500
BATCH_SIZE = 300


class NN:
    def __init__(self, n_features=5000, n_hidden=50,
                 folder='experiments/amazon_review/', epoch=1500, bsize=300,
                 domain_regularizer=None, save_weights='tmp_weights'):
        self.n_features = n_features
        self.nn = None
        self.encoder = None
        self.n_epoch = epoch
        self.batch_size = bsize
        self.visualize_model = None
        self.n_hidden = n_hidden
        self.domain_regularizer = domain_regularizer
        self.exp_folder = folder
        self.save_weights = save_weights
        
    def create(self):
        # Input
        input_s = Input(shape=(self.n_features,), name='souce_input')
        input_t = Input(shape=(self.n_features,), name='target_input')
        # Layers
        if self.domain_regularizer:
            encoding = Dense(self.n_hidden,
                             activation='sigmoid',
                             name='encoded',
                             activity_regularizer=self.domain_regularizer)
        else:
            encoding = Dense(self.n_hidden,
                             activation='sigmoid',
                             name='encoded')
        prediction = Dense(2,
                           activation='softmax',
                           name='pred')
        # encoding
        encoded_s = encoding(input_s)
        encoded_t = encoding(input_t)
        # prediction
        pred_s = prediction(encoded_s)
        pred_t = prediction(encoded_t)
        
        # Model Definition
        self.nn = Model(input=[input_s,input_t],
                        output=[pred_s,pred_t])
        
        # Optimizer
#        sgd = SGD(lr=self.learn_rate, nesterov=True, decay=1e-6, momentum=0.3)#, decay=1e-6
        adagrad = Adagrad()     
        
        # Model Compilation
        self.nn.compile(loss='categorical_crossentropy',
                        optimizer=adagrad,
                        metrics=['accuracy'],
                        loss_weights=[1.,0.])
        
        # Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,verbose=0)
        checkpointer = ModelCheckpoint(filepath=self.exp_folder+self.save_weights+'.hdf5',
                                       monitor='val_pred_acc_1',
                                       verbose=0, save_best_only=True)
        self.callbacks = [early_stopping,checkpointer]
        
        # Create seperate model for activation visualization
        self.visualize_model = Model(input=[input_s,input_t],
                                     output=[encoded_s,encoded_t])
        
    def fit(self, x_s, y_s, x_t, val_set=None, init_weights=None, verbose=0):
        np.random.seed(0)
        self.create()
        if init_weights:
            self.nn.load_weights(self.exp_folder+init_weights+'.hdf5')
        start = datetime.datetime.now().replace(microsecond=0)
        dummy=np.zeros((x_t.shape[0],1))
        dummy[0]=1
        y_s = to_categorical(y_s.astype(int))
        y_t = to_categorical(dummy.astype(int))
        if not val_set:
            self.nn.fit([x_s,x_t],[y_s,y_t],
                        batch_size=self.batch_size,
                        shuffle=False,
                        nb_epoch=self.n_epoch,
                        callbacks=self.callbacks,
                        verbose=verbose,
                        validation_split=0.3)
        else:
            y_val=to_categorical(val_set[1].astype(int))
            self.nn.fit([x_s,x_t],[y_s,y_t],
                        batch_size=self.batch_size,
                        shuffle=False,
                        nb_epoch=self.n_epoch,
                        callbacks=self.callbacks,
                        verbose=verbose,
                        validation_data=([val_set[0],val_set[0]],
                                         [y_val,y_val]))
        self.load(self.save_weights)# use with checkpointer
        stop = datetime.datetime.now().replace(microsecond=0)
        if verbose:
            print('done in '+str(stop-start))
    
    def predict(self,x):
        y = self.nn.predict([x,x])[1]
        out=np.zeros(y.shape[0])
        for i in range(out.shape[0]):
            out[i]=np.argmax(np.round(y[i,:]))
        return out
        
#    def evaluate(self,x,y):
#        score, acc = self.nn.evaluate(x,y,batch_size=self.batch_size)
#        return score,acc
        
    def load(self,name):
        self.create()
        self.nn.load_weights(self.exp_folder+name+'.hdf5')
        
    def get_activations(self,x_s,x_t):
        return self.visualize_model.predict([x_s,x_t])
        
    def save(self,name):
        self.nn.save_weights(self.exp_folder+name+'.hdf5',overwrite=True)
        
    def create_initial_weights(self,x_s,y_s,x_t,name):
        input_s = Input(shape=(self.n_features,), name='souce_input')
        input_t = Input(shape=(self.n_features,), name='target_input')
        encoding = Dense(self.n_hidden,
                         activation='sigmoid',
                         init='lecun_uniform',
                         name='encoded')
        prediction = Dense(2,
                           activation='softmax',
                           init='lecun_uniform',
                           name='pred')
        encoded_s = encoding(input_s)
        encoded_t = encoding(input_t)
        pred_s = prediction(encoded_s)
        pred_t = prediction(encoded_t)
        
        nn = Model(input=[input_s,input_t],
                        output=[pred_s,pred_t])
        sgd = SGD(0.1)
        nn.compile(loss='categorical_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'],
                   loss_weights=[1.,0.])
        
        dummy=np.zeros((x_t.shape[0],1))
        dummy[0]=1
        y_s = to_categorical(y_s.astype(int))
        y_t = to_categorical(dummy.astype(int))
        nn.fit([x_s,x_t],[y_s,y_t],nb_epoch=1,validation_split=0.3,verbose=0)
        nn.save_weights(self.exp_folder+name+'.hdf5',overwrite=True)
        
    def plot_architecture(self):
        plot(self.nn, to_file=self.exp_folder+'nn_rchitecture.png', show_shapes=True)
        















