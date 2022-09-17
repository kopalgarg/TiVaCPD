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
 
from cpd_methods import *

def main():

    # load the data
    if args.data_type in ['simulated_data', 'simulated']:
        data_path = args.data_path
        ind = args.ind
        X, gt_corr, gt_var, gt_mean = load_simulated(data_path, ind)
    elif args.data_type in ['har', 'HAR']:
        data_path = args.data_path
        X, y = load_har(data_path, ind)
    else:
        data_path = args.data_path
        X = load_real(data_path, ind)

    # results path
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp))
    exp_path = os.path.join(args.out_path, args.exp)

    if args.model_type == 'MMDATVGL_CPD':
        model = MMDATVGL_CPD(X, threshold=args.threshold, overlap=args.overlap, max_iters=args.max_iters, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim)
        plt = model.visualize_results(X, model.mmd_score, model.corr_score, gt_corr, gt_mean, gt_var, 'mmd', 'tvgl')
        plt.savefig(os.path.join(exp_path, 'cpd_mmdatvgl.png')) 
        with open(os.path.join(exp_path, 'mmda_score.pkl'),'wb') as f:
            pkl.dump((model.mmd_score), f)
        with open(os.path.join(exp_path, 'corr_score.pkl'),'wb') as f:
            pkl.dump((model.corr_score), f)

    elif args.model_type == 'GRAPHTIME_CPD':
        model = GRAPHTIME_CPD(series = X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, max_iter = args.max_iter)
        plt = model.plot_data_with_cps(X, model.cps)
        plt.savefig(os.path.join(exp_path, 'cpd_graphtime.png')) 
        with open(os.path.join(exp_path, 'graphtime_score.pkl'),'wb') as f:
            pkl.dump((model.cps), f)

    elif args.model_type == 'KLCPD':
        model = KLCPD(X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim)
        plt = model.visualize_results(X, model.scores, gt_corr, gt_mean, gt_var, 'klcpd')
        plt.savefig(os.path.join(exp_path, 'cpd_klcpd.png')) 
        with open(os.path.join(exp_path, 'klcpd_score.pkl'),'wb') as f:
            pkl.dump((model.scores), f)

    elif args.model_type == 'KSTBTVGL_CPD':
        model = KSTBTVGL_CPD(X, overlap=1, max_iters=args.max_iter, f_wnd_dim=args.f_wnd_dim, p_wnd_dim=args.p_wnd_dim)
        plt = model.visualize_results(X, model.scores, gt_corr, gt_mean, gt_var, 'kstvgl')
        plt.savefig(os.path.join(exp_path, 'cpd_kstvgl.png')) 
        with open(os.path.join(exp_path, 'kstvgl_score.pkl'),'wb') as f:
            pkl.dump((model.scores), f)

    elif args.model_type == 'KSTB_CPD':
        model = KSTB_CPD(X)
        plt = model.visualize_results(X, model.scores, gt_corr, gt_mean, gt_var, 'kst')
        plt.savefig(os.path.join(exp_path, 'cpd_kst.png')) 
        with open(os.path.join(exp_path, 'kst_score.pkl'),'wb') as f:
            pkl.dump((model.scores), f)

    elif args.model_type == 'MMDA_CPD':
        model = MMDA_CPD(X, threshold=args.threshold, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim)
        plt = model.visualize_results(X, model.scores, gt_corr, gt_mean, gt_var, 'mmda')
        plt.savefig(os.path.join(exp_path, 'cpd_mmda.png')) 
        with open(os.path.join(exp_path, 'mmda_score.pkl'),'wb') as f:
            pkl.dump((model.scores), f)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/')
    parser.add_argument('--out_path', default = './out')
    parser.add_argument('--data_type', default = 'simulated_data')
    parser.add_argument('--max_iters', type = int, default = 500)
    parser.add_argument('--overlap', type = int, default = 10)
    parser.add_argument('--threshold', type = float, default = .1)
    parser.add_argument('--f_wnd_dim', type = float, default = 5)
    parser.add_argument('--p_wnd_dim', type = float, default = 5)
    parser.add_argument('--exp', default = '3')
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--max_iter', type = int, default = 500)
    parser.add_argument('--ind', type = int, default = 0)

    args = parser.parse_args()

    main()

        

