# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
from tvgl import *
from load_data import *
import matplotlib
from performance import *
import roerich
from cpd_methods import *
from performance import *
import fnmatch,os
import pickle as pkl
from scipy.signal import savgol_filter
import warnings
from scipy.signal import peak_prominences
from pyampd.ampd import find_peaks, find_peaks_adaptive

def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

def peak_prominences_(distances):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances, range(len(distances)))
    return all_peak_prom

def post_processing(score, threshold):
        score_peaks = peak_prominences_(np.array(score))[0]
        for j in range(len(score_peaks)):
            if peak_prominences_(np.array(score))[0][j] - peak_prominences_(np.array(score))[0][j-1] >threshold :
                score_peaks[j] = 1
            else:
                score_peaks[j] = 0
        return score_peaks

def main():

    # load the data
    data_path = os.path.join(args.data_path)
    suffix = args.suffix
    match  = "{}_*".format(suffix)
    n_samples = len(fnmatch.filter(os.listdir(data_path), match))
    X_samples = []
    for i in range(n_samples):
        X = load_data(data_path, i, "{}_".format(suffix))
        X_samples.append(X)
            
    # results path
    if not os.path.exists(os.path.join(args.out_path)):
        os.mkdir(os.path.join(args.out_path))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp, args.model_type))
    

    for i in range(0, len(X_samples)):
        
        print(i)
        if args.model_type == 'MMDATVGL_CPD':
            X = X_samples[i]
            
            model = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = args.threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim) 

            mmd_score = shift(model.mmd_score, args.p_wnd_dim)
            corr_score = model.corr_score

            minLength = min(len(mmd_score), len(corr_score)) 
            corr_score = (corr_score)[:minLength]
            mmd_score = mmd_score[:minLength]
            
            # processed combined score
            
            mmd_score_savgol  = mmd_score #savgol_filter(mmd_score, 3, 2) # 2=polynomial order 
            corr_score_savgol =  savgol_filter(model.corr_score[:minLength], 7, 3)
            combined_score_savgol  = np.add(abs(mmd_score_savgol), abs(corr_score_savgol))

            plt.plot(mmd_score_savgol, label = 'mmd_score_savgol')
            plt.plot(corr_score_savgol, label = 'corr_score_savgol')
            plt.plot(combined_score_savgol, label = 'combined_score_savgol')
            plt.legend()
            plt.title(args.exp)
            plt.show()
        if args.model_type == 'KLCPD':
            X = X_samples[i]
            model = KLCPD(X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, epochs=20)
            y_pred = model.scores
            plt.plot(y_pred)
            plt.plot(X)
            plt.show()



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/changing_correlation') # exact data dir, including the name of exp
    parser.add_argument('--out_path', default = './out') # just the main out directory
    parser.add_argument('--max_iters', type = int, default = 1000)
    parser.add_argument('--overlap', type = int, default = 1)
    parser.add_argument('--threshold', type = float, default = .2)
    parser.add_argument('--f_wnd_dim', type = int, default = 10)
    parser.add_argument('--p_wnd_dim', type = int, default = 3)
    parser.add_argument('--exp', default = 'changing_correlation') # used for output path for results
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--score_type', default='combined') # others: combined, correlation, mmdagg
    parser.add_argument('--margin', default = 10)
    parser.add_argument('--suffix', default = 'series')

    args = parser.parse_args()

    main()

        