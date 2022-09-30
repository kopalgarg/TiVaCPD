import numpy as np
import pickle as pkl
import os
import random
from math import *
import matplotlib.pyplot as plt
import argparse
from load_data import *
import warnings

def block_dataset(K, T, B=3):
    warnings.simplefilter('ignore')
    block_len = np.ceil(T / B)

    corrs = np.zeros([B, int((K ** 2 - K) / 2 + K)])
    
    Y = np.zeros([T, K])

    for b in np.arange(B):
        randomCorr = random_corrmat(K)
        corrs[b, :] = mat2vec(randomCorr)
    corrs = np.repeat(corrs, block_len, axis=0)
    corrs = corrs[:T, :]

    for t in np.arange(T):
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]) + .1, cov=vec2mat(corrs[t, :]))

    return Y, corrs

def sim_data(n_cp=3):

  # Simulate and plot the signal

  mean = np.asarray([])
  var = np.asarray([])

  series, cor =  block_dataset(K = 3, T = 100, B=n_cp)
  mu = 0; sigma = 0
  mean = np.concatenate((mean, np.repeat(mu, 100)))
  var = np.concatenate((var, np.repeat(sigma, 100)))

  return series, mean, var, cor


def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--exp', default = 'block_correlation')
    parser.add_argument('--num_cp', type = int, default = 3)
    parser.add_argument('--n_samples', type = int, default = 10)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.path, args.exp)): 
        os.mkdir(os.path.join(args.path, args.exp))
    data_path = os.path.join(args.path, args.exp)

    for i in range(args.n_samples):
      X, mean, var, cor = sim_data(n_cp = args.num_cp)

      data_path = os.path.join(args.path, args.exp)

      save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
      save_data(os.path.join(data_path, ''.join(['gt_cor_', str(i), '.pkl'])), cor)
      save_data(os.path.join(data_path, ''.join(['gt_var_', str(i), '.pkl'])), var)
      save_data(os.path.join(data_path, ''.join(['gt_mean_', str(i), '.pkl'])), mean)
