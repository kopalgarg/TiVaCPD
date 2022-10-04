# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import json
import pickle as pkl
import scipy
import scipy.spatial.distance as sd


def load_data(data_path, ind, prefix):
    
    with open(os.path.join(data_path, ''.join([prefix, str(ind), '.pkl'])), 'rb') as f:
            X = pkl.load(f)

    return X

def load_simulated(data_path, ind):
    
    with open(os.path.join(data_path, ''.join(['series_', str(ind), '.pkl'])), 'rb') as f:
            X = pkl.load(f)
    with open(os.path.join(data_path, ''.join(['gt_cor_', str(ind), '.pkl'])), 'rb') as f:
            gt_corr = pkl.load(f)
    with open(os.path.join(data_path, ''.join(['gt_var_', str(ind), '.pkl'])), 'rb') as f:
            gt_var = pkl.load(f)
    with open(os.path.join(data_path, ''.join(['gt_mean_', str(ind), '.pkl'])), 'rb') as f:
            gt_mean = pkl.load(f)

    return X, gt_corr, gt_var, gt_mean

def load_beedance(data_path, ind):
    bee = scipy.io.loadmat(os.path.join(data_path, ''.join(['beedance2-', str(ind), '.mat']))) 
    X = bee['Y']
    y_true = bee['L']  
    return X, y_true  

def load_HAR(data_path, ind):
    
    with open(os.path.join(data_path, ''.join(['HAR_X_', str(ind), '.pkl'])), 'rb') as f:
            X = pkl.load(f)
    with open(os.path.join(data_path, ''.join(['HAR_y_', str(ind), '.pkl'])), 'rb') as f:
            y_true = pkl.load(f)

    return X, y_true

def rmdiag(m):
    return m - np.diag(np.diag(m))

def random_corrmat(K):
    
    x = np.random.randint(-1,2, size=(K, K))
    #np.random.randn(K, K)
    #x = x * x.T
    #x /= np.max(np.abs(x))
    np.fill_diagonal(x, 1.)
    
    return x

def mat2vec(m):
    """
    Function that converts correlation matrix to a vector

    Parameters
    ----------
    m : ndarray
        Correlation matix

    Returns
    ----------
    result : ndarray
        Vector

    """
    K = m.shape[0]
    V = int((((K ** 2) - K) / 2) + K)

    if m.ndim == 2:
        y = np.zeros(V)
        y[0:K] = np.diag(m)

        #force m to by symmetric
        m = np.triu(rmdiag(m))
        m[np.isnan(m)] = 0
        m += m.T

        y[K:] = sd.squareform(rmdiag(m))
    elif m.ndim == 3:
        T = m.shape[2]
        y = np.zeros([T, V])
        for t in np.arange(T):
            y[t, :] = mat2vec(np.squeeze(m[:, :, t]))
    else:
        raise ValueError('Input must be a 2 or 3 dimensional Numpy array')

    return y

def vec2mat(v):
    """
    Function that converts vector back to correlation matrix

    Parameters
    ----------
    result : ndarray
        Vector

    Returns
    ----------
    m : ndarray
        Correlation matix

    """
    if (v.ndim == 1) or (v.shape[0] == 1):
        x = int(0.5*(np.sqrt(8*len(v) + 1) - 1))
        return sd.squareform(v[x:]) + np.diag(v[0:x])
    elif v.ndim == 2:
        a = vec2mat(v[0, :])
        y = np.zeros([a.shape[0], a.shape[1], v.shape[0]])
        y[:, :, 0] = a
        for t in np.arange(1, v.shape[0]):
            y[:, :, t] = vec2mat(v[t, :])
    else:
        raise ValueError('Input must be a 1 or 2 dimensional Numpy array')

    return y