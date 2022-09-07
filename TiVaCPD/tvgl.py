import numpy as np
from scipy import stats

class TVGL():
    def __init__(self, alpha, beta, penalty_type, slice_size,
            rho=1, max_iters=1e5, e_abs=1e-4, e_rel=1e-4, overlap=1):
        self.alpha = alpha
        self.beta = beta
        self.penalty_type = penalty_type
        self.slice_size = slice_size
        self.rho = rho
        self.max_iters = max_iters
        self.e_abs = e_abs
        self.e_rel = e_rel
        self.overlap = overlap

    def fit(self, X):
        self.covariance_set = gen_s(X, self.slice_size, self.overlap)
        self.precision_set = admm(X, self.covariance_set, self.alpha, self.beta, self.penalty_type,
        self.slice_size, self.rho, self.max_iters, self.e_abs, self.e_rel)

def gen_s(X, slice_size, overlap):
    S = []
    for i in range(0, X.shape[0] - slice_size + 1, overlap):
        X_ = X[i:i + slice_size, :]
        X_ = stats.zscore(X_, axis=0)
        S_ = np.cov(X_.T)
        S.append(np.matrix(S_))
    return S

def initialize_theta(S):
    theta = []
    theta_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        theta.append(theta_)
    return theta

def initialize_z(S):
    z0 = []
    z1 = []
    z2 = []
    z_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        z0.append(z_)
        z1.append(z_)
        z2.append(z_)
    return z0, z1, z2

def initialize_u(S):
    u0 = []
    u1 = []
    u2 = []
    u_ = np.zeros((S[0].shape[0], S[0].shape[1]))
    for i in range(len(S)):
        u0.append(u_)
        u1.append(u_)
        u2.append(u_)
    return u0, u1, u2

def update_theta(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2):
    #update theta
    eta_ = 3 * rho / slice_size
    for i in range(len(theta)):
        A = z0[i] + z1[i] + z2[i] - u0[i] - u1[i] - u2[i]
        if i == 0 or i == len(theta) - 1:
            A = (1 / 2) * A
        else:
            A = (1 / 3) * A
        d, q = np.linalg.eigh(eta_ * (1 / 2) * (A + A.T) - S[i])
        d = np.diag(d)
        theta[i] = ( 1 / (2 * eta_)) * q * (d + np.sqrt(d ** 2 + 4 * eta_ * np.eye(A.shape[0]))) * q.T

    return theta

def update_z_l1(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2):
    #update z_0
    for i in range(len(z0)):
        A = theta[i] + u0[i]
        for m in range(A.shape[0]):
            for n in range(m + 1, A.shape[1]):
                if abs(A[m, n]) <= alpha / rho:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m ,n] = A[m, n] - alpha / rho
                        A[n ,m] = A[n, m] - alpha / rho
                    else:
                        A[m, n] = A[m, n] + alpha / rho
                        A[n, m] = A[n, m] + alpha / rho
        z0[i] = A

    #update z_1, z_2
    eta = 2 * beta / rho
    for i in range(1, len(z1)):
        A = theta[i] - theta[i - 1] + u2[i] - u1[i - 1]
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                if abs(A[m, n]) <= eta:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m, n] = A[m, n] - eta
                        A[n, m] = A[n, m] - eta
                    else:
                        A[m, n] = A[m, n] + eta
                        A[n, m] = A[n, m] + eta
        z1[i - 1] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) - (1 / 2) * A
        z2[i] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) + (1 / 2) * A

    return z0, z1, z2

def update_z_l2(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2):
    #update z_0
    for i in range(len(z0)):
        A = theta[i] + u0[i]
        for m in range(A.shape[0]):
            for n in range(m + 1, A.shape[1]):
                if abs(A[m, n]) <= alpha / rho:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m ,n] = A[m, n] - alpha / rho
                        A[n ,m] = A[n, m] - alpha / rho
                    else:
                        A[m, n] = A[m, n] + alpha / rho
                        A[n, m] = A[n, m] + alpha / rho
        z0[i] = A

    #update z_1, z_2
    eta = 2 * beta / rho
    for i in range(1, len(z1)):
        A = theta[i] - theta[i - 1] + u2[i] - u1[i - 1]
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                A[m, n] = (1 / (1 + 2 * eta)) * A[m, n]
                A[n, m] = (1 / (1 + 2 * eta)) * A[n, m]
        z1[i - 1] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) - (1 / 2) * A
        z2[i] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) + (1 / 2) * A

    return z0, z1, z2

def update_u(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2):
    #update u_0
    for i in range(len(S)):
        u0[i] = u0[i] + theta[i] - z0[i]

    #update u_1, u_2
    for i in range(1, len(S)):
        u1[i - 1] = u1[i - 1] + theta[i - 1] - z1[i - 1]
        u2[i] = u2[i] + theta[i] - z2[i]

    return u0, u1, u2

def calc_list_norm(A):
    norm = 0
    for i in range(len(A)):
        norm_ = 0
        for m in range(A[i].shape[0]):
            for n in range(m, A[i].shape[1]):
                norm_ += A[i][m,n] ** 2
        norm += norm_
    norm = np.sqrt(norm)

    return norm

def check_convergence(rho, e_abs, e_rel, theta, z0, z0_pre, u0):
    #calc e_dual and e_dual
    p = len(theta) * ((theta[0].shape[0] ** 2 - theta[0].shape[0]) / 2 + theta[0].shape[0])
    e_pri = np.sqrt(p) * e_abs + e_rel * max(calc_list_norm(theta), calc_list_norm(z0)) + .0001
    e_dual = np.sqrt(p) * e_abs + e_rel * calc_list_norm(u0) + .0001

    #calc res_pri
    res_pri = 0
    for i in range(len(theta)):
        A = theta[i] - z0[i]
        norm = 0
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                norm += A[m,n] ** 2
        res_pri += norm
    res_pri = np.sqrt(res_pri)

    #calc res_dual
    res_dual = 0
    for i in range(len(z0)):
        A = rho * (z0[i] - z0_pre[i])
        norm = 0
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                norm += A[m,n] ** 2
        res_dual += norm
    res_dual = np.sqrt(res_pri)

    return (res_pri < e_pri) and (res_dual < e_dual)

def admm(X, S, alpha, beta, penalty_type, slice_size, rho, max_iters, e_abs, e_rel):
    #set penalty_type
    if penalty_type == "L1":
        update_z = update_z_l1
    elif penalty_type == "L2":
        update_z = update_z_l2

    #initialize theta, z, u
    theta = initialize_theta(S)
    z0, z1, z2 = initialize_z(S)
    u0, u1, u2 = initialize_u(S)

    #start ADMM
    iters = 0
    stop = False
    while iters < max_iters:
        z_pre = z0
        theta = update_theta(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2)
        z0, z1, z2 = update_z(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2)
        u0, u1, u2 = update_u(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2)
        iters = iters + 1
        if check_convergence(rho, e_abs, e_rel, theta, z0, z_pre, u0) == True:
            break
    if iters == max_iters:
        print("max iters", iters)

    return theta
