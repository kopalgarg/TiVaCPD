# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from tvgl import *
from load_data import *
from performance import *
import roerich
from cpd_methods import *
from performance import *
import fnmatch,os
import pickle as pkl
from scipy.signal import savgol_filter
import ruptures as rpt

def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

def main():

    # load the data
    data_path = os.path.join(args.data_path)
    prefix = args.prefix
    match  = "{}_*".format(prefix)
    n_samples = len(fnmatch.filter(os.listdir(data_path), match))
    X_samples = []
    for i in range(n_samples):
        X = load_data(data_path, i, "{}_".format(prefix))
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

            data_path = os.path.join(args.out_path, args.exp)

            X = X_samples[i]
                        
            model = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = args.threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim,
            slice_size=args.slice_size, data_path = data_path, sample = i) 
            
            mmd_score = shift(model.mmd_score, args.p_wnd_dim)
           
            corr_score = model.corr_score

            minLength = min(len(mmd_score), len(corr_score)) 
            corr_score = (corr_score)[:minLength]
            mmd_score = mmd_score[:minLength]

            mmd_score_savgol  = savgol_filter(mmd_score, 11, 1) 
            corr_score_savgol = savgol_filter(corr_score, 11,1)
            combined_score_savgol  = savgol_filter(np.add(abs(mmd_score_savgol), abs(corr_score_savgol)), 11,1)
            
        
            plt.plot(mmd_score, label = 'DistScore')
            plt.plot(corr_score, label = 'CorrScore')
            plt.plot(np.add(abs(mmd_score), abs(corr_score)), label = 'Ensemble')
            plt.legend()
            plt.title(args.exp)
            plt.show()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
            save_data(os.path.join(data_path, ''.join(['mmd_score_', str(i), '.pkl'])), mmd_score)
            save_data(os.path.join(data_path, ''.join(['corr_score_', str(i), '.pkl'])), corr_score)

        elif args.model_type == 'KLCPD':
            X = X_samples[i]
            model = KLCPD(X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, epochs=20)
            y_pred = model.scores
            plt.plot(y_pred)
            plt.plot(X)
            plt.show()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
            save_data(os.path.join(data_path, ''.join(['klcpd_score_', str(i), '.pkl'])), y_pred)

        elif args.model_type == 'GRAPHTIME_CPD':
            X = X_samples[i]
            model = GRAPHTIME_CPD(series = X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, max_iter = 500)
            y_pred = np.zeros((len(X)))
            
            plt.plot(X)
            plt.plot(y_pred, label = 'graphtime')
            plt.legend()
            plt.title(args.exp)
            plt.show()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
            save_data(os.path.join(data_path, ''.join(['graphtime_score_', str(i), '.pkl'])), y_pred)
            
        elif args.model_type  == 'roerich':
            X = X_samples[i]
            model = roerich.OnlineNNClassifier(net='default', scaler="default", metric="KL_sym",
                  periods=1, window_size=10, lag_size=30, step=10, n_epochs=25,
                  lr=0.01, lam=0.0001, optimizer="Adam"
                 )
            y_pred, _ = model.predict(X)

            plt.plot(X)
            plt.plot(y_pred, label = 'roerich')
            plt.legend()
            plt.title(args.exp)
            plt.show()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['roerich_score_', str(i), '.pkl'])), y_pred)
        
        elif args.model_type  == 'ruptures':

            X = X_samples[i]

            n_samples = X.shape[0]
            algo = rpt.Pelt(model="linear").fit(X)
            result = algo.predict(pen=10)

            y_pred=np.zeros(X.shape[0]+1)
            y_pred[result] = 1

            plt.plot(X)
            plt.plot(y_pred, label = 'ruptures')
            plt.legend()
            plt.title(args.exp)
            plt.show()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['ruptures_score_', str(i), '.pkl'])), y_pred)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/grump') # exact data dir, including the name of exp
    parser.add_argument('--out_path', default = './out') # just the main out directory
    parser.add_argument('--max_iters', type = int, default = 1000)
    parser.add_argument('--overlap', type = int, default = 1)
    parser.add_argument('--threshold', type = float, default = .2)
    parser.add_argument('--f_wnd_dim', type = int, default = 5)
    parser.add_argument('--p_wnd_dim', type = int, default = 3)
    parser.add_argument('--exp', default = 'grump') # used for output path for results
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--score_type', default='combined') # others: combined, correlation, mmdagg
    parser.add_argument('--margin', default = 10)
    parser.add_argument('--prefix', default = 'user')
    parser.add_argument('--slice_size', default = 10)

    args = parser.parse_args()

    main()

        