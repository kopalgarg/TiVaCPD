# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import json
import numpy as np
import pickle as pkl
from scipy.stats import ks_2samp
import metrics

def univariate_ks(prev, next):
    """
    Univariate 2 Sample Testing with Bonferroni Aggregation

    Arguments:
        prev {vector} -- [n_sample1, dim]
        next {vector} -- [n_sample2, dim]
    Returns:
        p_val -- [p-value]
        t_val -- [t-value, i.e. KS test-statistic]

    """
    p_vals = []
    t_vals = []

  # for each dimension, we conduct a separate KS test
    for i in range(prev.shape[1]):
        feature_tr = prev[:, i]
        feature_te = next[:, i]

        t_val, p_val = None, None
        t_val, p_val = ks_2samp(feature_tr, feature_te)
        p_vals.append(p_val)
        t_vals.append(t_val)

    # apply the Bonferroni correction for the family-wise error rate by picking the minimum
    # p-value from all individual tests
    p_vals = np.array(p_vals)
    t_vals = np.array(t_vals)
    p_val = min(np.min(p_vals), 1.0)
    t_val = np.mean(t_vals)
  
    return p_val, t_val


def mmd_linear(X, Y):
    """
    MMD with Linear Kernel

    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
    Returns:
        mmd_val -- [MMD value]
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    mmd_val=XX.mean() + YY.mean() - 2 * XY.mean()

    return mmd_val

def mmd_rbf(X, Y, gamma=1.0):
    """
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    
    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
        gamma {float} -- [kernel parameter, default: 1.0]
    
    Returns:
        mmd_val {scalar} -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """
    MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    
    Arguments:
        X {matrix} -- [n_sample1, dim]
        Y {matrix} -- [n_sample2, dim]
        degree {int} -- [degree, default: 2)
        gamma {int} -- [gamma, default: 1]
        coef0 {int} -- [constant item, default: 0]

    Returns:
        mmd_val {scalar} -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()
