# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import json
import pickle as pkl
import scipy

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