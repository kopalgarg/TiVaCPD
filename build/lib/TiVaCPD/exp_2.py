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

def main():

    # load the data
    if args.data_type in ['simulated_data', 'simulated']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'series_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, gt_cor, gt_var, gt_mean = load_simulated(data_path, i)
            y_true = abs(gt_mean) + abs(gt_var) + abs(gt_cor)
            y_true_spike = y_true.copy()
            for j in range(len(y_true)):
                if y_true[j] != y_true[j-1] and j!=0:
                    y_true_spike[j] = 1
                else:
                    y_true_spike[j] = 0
            y_true = y_true_spike
            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['beewaggle', 'beedance']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'*.mat'))
        X_samples = []
        y_true_samples = []
        for i in range(1,n_samples):
            X, y_true = load_beedance(data_path, i)
            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['HAR', 'har']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'HAR_X_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, y_true = load_HAR(data_path, i)
            y_true_spike = y_true.copy()
            for j in range(len(y_true)):
                if y_true[j] != y_true[j-1]:
                    y_true_spike[j] = 1
                else:
                    y_true_spike[j] = 0
            y_true = y_true_spike
            y_true[0] = 0
            X_samples.append(X)
            y_true_samples.append(y_true)
    # results path
    if not os.path.exists(os.path.join(args.out_path)):
        os.mkdir(os.path.join(args.out_path))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp, args.model_type))
    
    auc_scores_combined =  []
    f1_scores_combined = []
    auc_scores_correlation =  []
    f1_scores_correlation= []
    auc_scores_mmdagg =  []
    f1_scores_mmdagg= []
    auc_scores = []
    f1_scores = []

    for i in range(len(X_samples)):
        print(i)
        if args.model_type == 'MMDATVGL_CPD':
            X = X_samples[i]
            y_true = y_true_samples[i]

            model = MMDATVGL_CPD(X, max_iters = 1000, overlap=args.overlap)
            mmd_score = shift(model.mmd_score, args.p_wnd_dim)
            corr_score = model.corr_score
            minLength = min(len(mmd_score), len(corr_score)) 
            corr_score = (corr_score/2)[:minLength]
            mmd_score = mmd_score[:minLength]
            combined_score = np.add(mmd_score, corr_score)/2
            y_true = y_true[:minLength]
            
            y_pred = mmd_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin, 0.01)
            auc_scores_mmdagg.append(metrics.auc)
            f1_scores_mmdagg.append(metrics.f1) 

            y_pred = combined_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin, 0.01)
            auc_scores_combined.append(metrics.auc)
            f1_scores_combined.append(metrics.f1)

            y_pred = corr_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin, 0.01)
            auc_scores_correlation.append(metrics.auc)
            f1_scores_correlation.append(metrics.f1)  

        elif args.model_type == 'GRAPHTIME_CPD':
            X = X_samples[i]
            y_true = y_true_samples[i]
            model = GRAPHTIME_CPD(series = X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, max_iter = args.max_iters)
            y_pred = np.zeros((len(y_true)))
            for j in range(len(y_true)):
                if j in model.cps:
                    y_pred[j] = 1 
                else:
                    y_pred[j] = 0

            metrics = ComputeMetrics(y_true, y_pred, args.margin, args.threshold, process=False)
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 

        elif args.model_type == 'KLCPD':
            X = X_samples[i]
            y_true = y_true_samples[i]
            model = KLCPD(X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, epochs=20)
            y_pred = model.scores

            metrics = ComputeMetrics(y_true, y_pred, args.margin, args.threshold, model_type='KLCPD')
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 

        elif args.model_type == 'roerich':
            X = X_samples[i]
            y_true = y_true_samples[i]
            model = roerich.OnlineNNClassifier(net='default', scaler="default", metric="KL_sym",
                  periods=1, window_size=10, lag_size=30, step=10, n_epochs=25,
                  lr=0.01, lam=0.0001, optimizer="Adam"
                 )
            y_pred, _ = model.predict(X)
            metrics = ComputeMetrics(y_true, y_pred, args.margin, args.threshold)
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 

    print(args.data_type, args.model_type, args.exp, args.score_type)

    if args.model_type == 'MMDATVGL_CPD':
        print("auc_scores_combined_CI", mean_confidence_interval(auc_scores_combined))
        print("f1_scores_combined_CI", mean_confidence_interval(f1_scores_combined))

        print("auc_scores_correlation_CI", mean_confidence_interval(auc_scores_correlation))
        print("f1_scores_correlation_CI", mean_confidence_interval(f1_scores_correlation))

        print("auc_scores_mmdagg_CI", mean_confidence_interval(auc_scores_mmdagg))
        print("f1_scores_mmdagg_CI", mean_confidence_interval(f1_scores_mmdagg))
    else:
        print(args.data_type, args.model_type, args.exp, args.score_type)
        print("auc_CI", mean_confidence_interval(auc_scores))
        print("f1_CI", mean_confidence_interval(f1_scores))


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/beedance') # exact data dir, including the name of exp
    parser.add_argument('--out_path', default = './out') # just the main out directory
    parser.add_argument('--data_type', default = 'simulated_data') # others: beedance, HAR
    parser.add_argument('--max_iters', type = int, default = 1000)
    parser.add_argument('--overlap', type = int, default = 1)
    parser.add_argument('--threshold', type = float, default = .05)
    parser.add_argument('--f_wnd_dim', type = float, default = 10)
    parser.add_argument('--p_wnd_dim', type = float, default = 3)
    parser.add_argument('--exp', default = '3') # used for output path for results
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--score_type', default='combined') # others: combined, correlation, mmdagg
    parser.add_argument('--margin', default = 5)

    args = parser.parse_args()

    main()

        
