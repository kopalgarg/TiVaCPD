import numpy as np
import pickle as pkl
import os
import random
from math import *
import matplotlib.pyplot as plt
import argparse
from load_data import *
import warnings

def block_dataset(K, T, B=5):
    warnings.simplefilter('ignore')
    block_len = np.ceil(T / B)

    corrs = np.zeros([B, int((K ** 2 - K) / 2 + K)])
    
    Y = np.zeros([T, K])

    for b in np.arange(B):
        randomCorr = random_corrmat(K)
        print(randomCorr)
        corrs[b, :] = mat2vec(randomCorr)
    corrs = np.repeat(corrs, block_len, axis=0)
    corrs = corrs[:T, :]

    for t in np.arange(T):
        Y[t, :] = np.random.multivariate_normal(mean=np.zeros([K]) + .1, cov=vec2mat(corrs[t, :]))

    return Y, corrs

def sim_data(n_cp=3, constant_mean = False, constant_var = False, constant_corr = False):

  # Simulate and plot the signal

  s1 = np.asarray([]).reshape(-1, 1)
  s2 = np.asarray([]).reshape(-1, 1)
  s3 = np.asarray([]).reshape(-1, 1)
  mean = np.asarray([])
  var = np.asarray([])
  cor = np.asarray([])

  for i in range(n_cp):
    
    len = int(random.randrange(50, 100))  # random length of segment

    if constant_mean == False:
      mu = round(random.uniform(1, 3), 2)
      #mu = [1,3,5,3,5,3,1,3,5,3,5,3,1,3,5,3,5,3][i]
    else:
      
      mu = 0
    
    if constant_var == False:
      #sigma = round(random.uniform(.1, .7),2)
      sigma = [.1,.8,.1,.8,.1,.8,.1,.8,.5,.3,.5,.3,.1,.3,.5,.3,.5,.3][i]
    else:
      sigma = 0.1
    
    if constant_corr == False:
      rho = [1, -1, 0, 1, -1, 0, 1, -1][i] #int(random.randrange(-1, 2)) #round(random.uniform(-1,1),2)

    else:
      rho = 0

    mean = np.concatenate((mean, np.repeat(mu, len)))
    var = np.concatenate((var, np.repeat(sigma, len)))
    cor = np.concatenate((cor, np.repeat(rho, len)))

    X1 = np.random.normal(mu, sigma, len).reshape(-1, 1) # s1
    Z1 = np.random.normal(mu, sigma, len).reshape(-1, 1)
    X2 = rho*X1+sqrt(1-np.power(rho,2))*Z1                  # s2
    X3 = np.random.normal(mu, sigma, len).reshape(-1, 1) # s3

    s1 = np.concatenate((s1, X1), axis = 0)
    s2 = np.concatenate((s2, X2), axis = 0)
    s3 = np.concatenate((s3, X3), axis = 0)
  
  series = np.concatenate((s1, s2, s3), axis=1) 
  return series, mean, var, cor


def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--constant_mean', default = 'False')
    parser.add_argument('--constant_var', default = 'False')
    parser.add_argument('--constant_corr', default = 'False')
    parser.add_argument('--exp', default = '0')
    parser.add_argument('--num_cp', type = int, default = 3)
    parser.add_argument('--n_samples', type = int, default = 20)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.path, args.exp)): 
        os.mkdir(os.path.join(args.path, args.exp))
    data_path = os.path.join(args.path, args.exp)

    for i in range(args.n_samples):
      X, mean, var, cor = sim_data(n_cp = args.num_cp, constant_mean = eval(args.constant_mean), 
                                                    constant_var = eval(args.constant_var), 
                                                    constant_corr = eval(args.constant_corr))

      data_path = os.path.join(args.path, args.exp)

      save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
      save_data(os.path.join(data_path, ''.join(['gt_cor_', str(i), '.pkl'])), cor)
      save_data(os.path.join(data_path, ''.join(['gt_var_', str(i), '.pkl'])), var)
      save_data(os.path.join(data_path, ''.join(['gt_mean_', str(i), '.pkl'])), mean)
