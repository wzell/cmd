import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_activations(name, model, x_s, y_s, x_t, y_t, lift=False, cl_lift=0):
    """
    plot t-SNE embeddings of amazon->webcam task
    """
    a_s = model.get_activations(x_s, x_s)[1]
    a_t = model.get_activations(x_t, x_t)[1]    
    n_s = a_s.shape[0]
    a = np.concatenate((a_s, a_t),axis=0)
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    a_transf = model.fit_transform(a)
    
    fig=plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.scatter(a_transf[:n_s,0], a_transf[:n_s,1],marker='.',s=200,
                c=y_s.argmax(1), cmap=plt.cm.get_cmap("jet", y_s.shape[1]), alpha=0.2)
    ax.scatter(a_transf[n_s:,0], a_transf[n_s:,1],marker=(5,2),s=200,
                c=y_t.argmax(1), cmap=plt.cm.get_cmap("jet", y_t.shape[1]), alpha=0.2)
    if lift:    
        plt.scatter(a_transf[n_s:,0][y_t.argmax(1)==cl_lift],
                    a_transf[n_s:,1][y_t.argmax(1)==cl_lift],marker=(5,2),s=500,c='k')
        plt.scatter(a_transf[:n_s,0][y_s.argmax(1)==cl_lift],
                    a_transf[:n_s,1][y_s.argmax(1)==cl_lift],marker='.',s=500,c='k')
    ax.axis('off')
    plt.savefig(name+'.tif')