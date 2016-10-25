# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 2016

@author: zellinger
"""

from __future__ import print_function

import numpy as np
import datetime
np.random.seed(0)

from os.path import isfile
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, Model

class Batches:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        
    def next_batch(self):
        """
        get next batch
        """
        if self.y.shape[0]>self.batch_size:
            return self.next_batch_smaller(self.x, self.y, self.batch_size)
        else:
            return self.next_batch_bigger()
            
    def next_batch_smaller(self, x, y, batch_size):
        """
        calculate random batch if batchsize of target is smaller than source
        """
        x_batch = np.array([])
        y_batch = np.array([])
        n_min = int(np.min(self.y.sum(0)))
        n_rest = int(batch_size - n_min*y.shape[1])
        if n_rest<0:
            n_min = int(batch_size /y.shape[1])
            n_rest = batch_size %y.shape[1]
        ind_chos = np.array([])
        is_first = True
        # fill with n_min samples per class
        for cl in range(y.shape[1]):
            ind_cl = np.arange(y.shape[0])[y[:,cl]!=0]
            ind_cl_choose = np.random.permutation(np.arange(ind_cl.shape[0]))[:n_min]
            if is_first:
                x_batch = x[ind_cl[ind_cl_choose]]
                y_batch = y[ind_cl[ind_cl_choose]]
                is_first = False
            else:
                x_batch = np.concatenate((x_batch,x[ind_cl[ind_cl_choose]]),axis=0)
                y_batch = np.concatenate((y_batch,y[ind_cl[ind_cl_choose]]),axis=0)
            ind_chos = np.concatenate((ind_chos,ind_cl[ind_cl_choose]))
        # fill with n_rest random samples
        mask = np.ones(x.shape[0],dtype=bool)
        mask[ind_chos.astype(int)] = False
        x_rem = x[mask]
        y_rem = y[mask]
        ind_choose = np.random.permutation(np.arange(x_rem.shape[0]))[:n_rest]
        x_batch = np.concatenate((x_batch,x_rem[ind_choose]),axis=0)
        y_batch = np.concatenate((y_batch,y_rem[ind_choose]),axis=0)
        return x_batch, y_batch
        
    def next_batch_bigger(self):
        """
        calculate random batch if batchsize of target is greater than source
        """
        x_add, y_add = self.next_batch_smaller(self.x, self.y, self.batch_size-self.x.shape[0])
        x_batch = np.concatenate((self.x,x_add),axis=0)
        y_batch = np.concatenate((self.y,y_add),axis=0)
        return x_batch, y_batch
        
            
class NN:
    def __init__(self,
                 folder,
                 n_features=256,
                 max_n_epoch=10000,
                 domain_regularizer=None,
                 save_weights='save_weights'):
        self.nn = None
        self.exp_folder = folder
        self.max_n_epoch = max_n_epoch
        self.n_features = n_features
        self.save_weights = save_weights
        self.domain_regularizer = domain_regularizer
        self.visualize_model = None
        
    def create(self):
        """
        create two layer classifier
        """
        # input
        img_repr_s = Input(shape=(4096,), name='souce_input')
        img_repr_t = Input(shape=(4096,), name='target_input')
        # layers
        if self.domain_regularizer:
            shared_dense = Dense(self.n_features,
                                 name='shared_dense',
                                 activation='sigmoid',
                                 init='he_normal',
                                 activity_regularizer=self.domain_regularizer)
        else:
            shared_dense = Dense(self.n_features,
                                 name='shared_dense',
                                 activation='sigmoid',
                                 init='he_normal')
        classifier = Dense(31,
                           name='clf',
                           activation='softmax')
        # encoding
        s_d_s = shared_dense(img_repr_s)
        s_d_s = Dropout(0.8)(s_d_s)
        s_d_t = shared_dense(img_repr_t)
        s_d_t = Dropout(0.8)(s_d_t)
        # prediction
        pred_s = classifier(s_d_s)
        pred_t = classifier(s_d_t)
        # model definition
        self.nn = Model(input=[img_repr_s, img_repr_t],
                        output=[pred_s, pred_t])
        # model compilation
        self.nn.compile(loss='categorical_crossentropy',
                        optimizer='Adadelta',
                        metrics=['categorical_accuracy'],
                        loss_weights=[1.,0.])
        # seperate model for activation visualization
        self.visualize_model = Model(input=[img_repr_s,img_repr_t],
                                     output=[s_d_s,s_d_t])
        
    def get_activations(self, x_s, x_t):
        return self.visualize_model.predict([x_s,x_t])
        
    def create_img_repr(self, weights_file, gen, save_name, max_n_imgs):
        """
        calculate image representation via VGG_16 neural net
        """
        vgg16 = self.VGG_16(weights_file)
        vgg16_repr = Model(input=vgg16.input,
                           output=[vgg16.get_layer('dense_2').output])
        vgg16_repr.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['categorical_accuracy'])
        
        if not isfile(self.exp_folder+save_name+'_img_repr.npy'):
            print('Calculating image representations of '+str(save_name)+'..')
            repr = np.array([])
            labels = None
            batch_size = 0
            is_first = True
            while True:
                (x, y) = gen.next()
                # crop img
                if x.shape[2]>224:
                    cut = int((x.shape[2]-224)/2)
                    x = x[:,[2,1,0],cut:x.shape[2]-cut,cut:x.shape[2]-cut]
                # calculate img representations
                img_repr = vgg16_repr.predict(x)
                if is_first:
                    batch_size = x.shape[0]
                    is_first = False
                    repr = img_repr
                    labels = y
                else:
                    repr = np.concatenate((repr,img_repr), axis=0)
                    labels = np.concatenate((labels,y))
                if x.shape[0]<batch_size or repr.shape[0] == max_n_imgs:
                    break
            np.save(open(self.exp_folder+save_name+'_img_repr.npy', 'w'), repr)
            np.save(open(self.exp_folder+save_name+'_labels.npy', 'w'), labels)
        else:
            print('Loading image representations of '+str(save_name)+'..')
            repr = np.load(open(self.exp_folder+save_name+'_img_repr.npy'))
            labels = np.load(open(self.exp_folder+save_name+'_labels.npy'))
        return repr, labels
        
    def fit(self, x_s, y_s, x_t, verbose=False, y_t=[], init_weights=None):
        """
        train classifier
        """
        start = datetime.datetime.now().replace(microsecond=0)
        self.create()
#        np.random.seed(0)
        if init_weights:
            self.load(init_weights)
            
        best_acc = 0
        best_loss = 100
        counter = 0
        dummy_y_t =np.zeros((x_t.shape[0],y_s.shape[1]))
        
        iter_batches = Batches(x_s, y_s, x_t.shape[0])
            
        for i in range(self.max_n_epoch):
            x_s_batch, y_s_batch = iter_batches.next_batch()
            metrics = self.nn.train_on_batch([x_s_batch, x_t],
                                             [y_s_batch, dummy_y_t])
            if metrics[3]>best_acc:
                self.save(self.save_weights)
                best_acc = metrics[3]
                best_loss = metrics[1]
                counter = 0
            elif metrics[3]==best_acc and metrics[1]<best_loss:
                self.save(self.save_weights)
                best_loss = metrics[1]
                best_acc = metrics[3]
                counter+=1
            else:
                counter+=1
            if i%20 == 0 and verbose:
                accs = self.nn.evaluate([x_t, x_t],
                                        [y_t, y_t],
                                        verbose = 0)
                print('Batch update %.4d loss= %.4f tr-acc= %.4f tst-acc= %.4f'
                % (i, metrics[1], best_acc, accs[4]))
            if counter>1000:
                break
            
        self.load(self.save_weights)
        stop = datetime.datetime.now().replace(microsecond=0)
        print('done in '+str(stop-start))
        
    def evaluate(self, x, y):
        """
        evaluate classifier
        """
        accs = self.nn.evaluate([x, x],
                                [y, y],
                                verbose = 0)
        return accs[4]
        
    def predict(self, x):
        """
        predict classifier
        """
        return self.nn.predict([x, x])[1]
        
    def save(self, name):
        """
        save weights
        """
        self.nn.save_weights(self.exp_folder+name+'.hdf5',overwrite=True) 
        
    def load(self,name):
        """
        load weights
        """
        self.create()
        self.nn.load_weights(self.exp_folder+name+'.hdf5')
        
    def VGG_16(self, weights_path=None):
        """
        VGG_16 network architecture
        Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional
        networks for large-scale image recognition."
        arXiv preprint arXiv:1409.1556 (2014).
        """
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2), name='conv'))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', name='dense_1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='dense_2'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax', name='dense_3'))
    
        if weights_path:
            model.load_weights(weights_path)
    
        return model
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        