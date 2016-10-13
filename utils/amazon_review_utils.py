# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:36:58 2016

@author: zellinger
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import loadmat

def plot_sensitivity(name, accs_mmd, accs_mmatch, betas, n_moments):
    """
    Plot sensitivity analysis
    """
    base_mmatch = 4
    base_mmd = accs_mmd.mean(1).mean(0).argmax()
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for i in range(12):
        ax1.plot(n_moments,
                 accs_mmatch.mean(1)[i,:]/accs_mmatch.mean(1)[i,base_mmatch])
        ax2.plot(range(1,betas.shape[0]+1),
                       accs_mmd.mean(1)[i,:]/accs_mmd.mean(1)[i,base_mmd])
    ax1.plot(n_moments,
             accs_mmatch.mean(1).mean(0)/accs_mmatch.mean(1).mean(0)[base_mmatch],
             'k--',linewidth=4)
    ax2.plot(n_moments, accs_mmd.mean(1).mean(0)/accs_mmd.mean(1).mean(0)[base_mmd],
             'k--',linewidth=4)     
    ax1.grid(True,linestyle='-',color='0.75')
    ax2.grid(True,linestyle='-',color='0.75')
    plt.sca(ax1)
    plt.xticks(range(1,len(n_moments)+1),n_moments)
    plt.xlabel('number of moments', fontsize=15)
    plt.ylabel('accuracy improvement', fontsize=15)
    plt.sca(ax2)
    plt.xticks(range(1,betas.shape[0]+1),betas.round(1))
    plt.xlabel('kernel parameter', fontsize=15)
    ax1.set_ylim([0.97,1.01])
    plt.savefig(name+'.tif')

def load_amazon(n_features, filename):
    """
    Load amazon reviews
    """
    mat = loadmat(filename)
    
    xx=mat['xx']
    yy=mat['yy']
    offset=mat['offset']
    
    x=xx[:n_features,:].toarray().T#n_samples X n_features
    y=yy.ravel()
    
    return x, y, offset

def make_final_table(name, acc_mat):
    """
    plot final table of amazon review experiment
    """
    table = acc_mat.mean(2)
    final_table = np.round(table,3).astype('str')
    final_std = acc_mat.std(2)
    final_std = np.round(final_std,3).astype('str')
    final_table = np.core.defchararray.add(final_table,np.full_like(final_table,'+-'))
    final_table = np.core.defchararray.add(final_table,final_std)
    
    print(final_table)
    
    col_labels = ('SimpleNN', 'MMD','Mmatch')
    row_labels = ('b->d','b->e','b->k','d->b',
                  'd->e','d->k','e->b',
                  'e->d','e->k','k->b',
                  'k->d','k->e')
    nrows, ncols = 12, len(col_labels)
    hcell, wcell = 1., 3.
    fig=plt.figure(figsize=(ncols*wcell, nrows*hcell))
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_color('w')
        sp.set_zorder(0)
    the_table = ax.table(cellText=final_table,
                         colLabels=col_labels,
                         rowLabels=row_labels,
                         loc='center')
    the_table.set_zorder(10)
    plt.savefig(name+'.tif')

def shuffle(x, y):
    """
    shuffle data (used by split)
    """
    index_shuf = np.arange(x.shape[0])
    np.random.shuffle(index_shuf)
    x=x[index_shuf,:]
    y=y[index_shuf]
    return x,y
    
def perc_acc(y, y_true):
    """
    percentage of right classified
    """
    return 1-np.sum(np.abs(np.round(y).ravel()-y_true.ravel()))/y.shape[0]
    
def reverse_validation(model, init_weights, S, T, name='nn'):
    """
    reverse validation
    
    - Zhong, Erheng, et al. "Cross validation framework to choose amongst models
    and datasets for transfer learning.", Joint European Conference on Machine
    Learning and Knowledge Discovery in Databases. Springer Berlin Heidelberg,
    2010.
    """
    train_perc=0.8
    x_tr_s=S[0][:int(S[0].shape[0]*train_perc),:]
    y_tr_s=S[1][:int(S[1].shape[0]*train_perc)]
    x_val_s=S[0][int(S[0].shape[0]*train_perc):,:]
    y_val_s=S[1][int(S[1].shape[0]*train_perc):]
    x_tr_t=T[:int(T.shape[0]*train_perc),:]
    x_val_t=T[int(T.shape[0]*train_perc):,:]
    # Train model \nu
    model.fit(x_tr_s, y_tr_s, x_tr_t, val_set=(x_val_s, y_val_s), init_weights=init_weights)
    # Save the weights as init for next turn
    model.save('tmp_weights_rv')
    # Predict target labels
    y_pred_t = model.predict(x_tr_t)
    y_pred_val_t = model.predict(x_val_t)
    # Learn reverse classifier \nu_r (load weights)
    # Init with first fitted weights, procedere taken from
    # Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks.",
    # arXiv preprint arXiv:1505.07818 (2015).
    model.fit(x_tr_t, y_pred_t, x_tr_s, val_set=(x_val_t,y_pred_val_t),
              init_weights='tmp_weights_rv')
    # Evaluate reverse classifier
    y_pred_s=model.predict(x_val_s)
    # Calculate accuracy
    acc = perc_acc(y_pred_s,y_val_s)
    # Return reverse validation risk
    return acc
    
def split_data(d_s_ind,d_t_ind,x,y,offset,n_tr_samples,r_seed=0):
    """
    split data (train/validation/test, source/target)
    """
    np.random.seed(r_seed)
    x_s_tr = x[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples,:]
    x_t_tr = x[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples,:]
    x_s_tst = x[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0],:]
    x_t_tst = x[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0],:]
    y_s_tr = y[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples]
    y_t_tr = y[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples]
    y_s_tst = y[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0]]
    y_t_tst = y[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0]]
    x_s_tr,y_s_tr=shuffle(x_s_tr,y_s_tr)
    x_t_tr,y_t_tr=shuffle(x_t_tr,y_t_tr)
    x_s_tst,y_s_tst=shuffle(x_s_tst,y_s_tst)
    x_t_tst,y_t_tst=shuffle(x_t_tst,y_t_tst)
    y_s_tr[y_s_tr==-1]=0
    y_t_tr[y_t_tr==-1]=0
    y_s_tst[y_s_tst==-1]=0
    y_t_tst[y_t_tst==-1]=0
    return x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst
    
def plotpd(a,title,save_name,save=False):
    """
    make use of pandas/seaborn plotting functionality
    """
    n_dim=a.shape[1]-1
    n_rows=1
    n_cols=n_dim
    fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols, sharey=True)
    for k,ax in enumerate(axs.reshape(-1)):
        if k>=n_dim:
            continue
        ax.set_title('dim='+str(k),fontsize=10)
        sns.kdeplot(a[a['domain']==1][k],ax=ax, shade=True, label='target', legend=False)
        sns.kdeplot(a[a['domain']==0][k],ax=ax, shade=True, label='source', legend=False)
    plt.suptitle(title,fontsize=14)
    if save:
        fig.set_figheight(3)
        plt.setp(axs, xticks=[0, 0.5, 1])
        plt.savefig(save_name+'.png')
        
def to_pandas(a_s,a_t):
    """
    define pandas array
    """
    a_s=pd.DataFrame(data=a_s,index=np.arange(a_s.shape[0]),
                 columns=np.arange(a_s.shape[1]))
    a_t=pd.DataFrame(data=a_t,index=np.arange(a_s.shape[0],a_s.shape[0]+a_t.shape[0]),
                     columns=np.arange(a_t.shape[1]))
    # adding class labels 0
    a_s['domain'] = pd.Series(np.zeros(len(a_s[0])),index=a_s.index)
    a_t['domain'] = pd.Series(np.ones(len(a_t[0])),index=a_t.index)
    a = pd.concat([a_s,a_t])
    return a
    
def plot_activations(a_s,a_t,save_name,title='Activations'):
    """
    plot histograms of activations
    """
    a = to_pandas(a_s,a_t)
    plotpd(a,title,save_name,save=True) 
    
    