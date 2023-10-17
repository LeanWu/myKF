import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import poliastro.constants.general as constant
import dynamics

# 卡尔曼滤波

# 卡尔曼预测步
def KF_predict(x, P, F, Q, B=0, u=0):
    x_new = np.dot(F, x) + np.dot(B, u).flatten()
    P_new = np.dot(np.dot(F, P), F.T) + Q
    return x_new, P_new

# 卡尔曼更新步
def KF_update(z, x, P, H, R):
    y = z - np.dot(H, x)
    K = np.dot(P, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P, H.T))+R)))
    x_next = x + np.dot(K, y)
    size = len(x)
    P_next = np.dot(np.eye(size) - np.dot(K, H), np.dot(P, (np.eye(size) - np.dot(K, H)).T)) + np.dot(np.dot(K, R), K.T)
    return x_next, P_next

# 执行卡尔曼滤波
def KF_run(x0, P0, F, H, Q, R, zs, B=0, u=0):
    xs, cov = [], []
    x, P = x0, P0
    for z in zs:
        if np.linalg.norm(z-zs[0])==0:
            x_next, P_next = KF_update(z, x, P, H, R)
        else:
            x_new, P_new = KF_predict(x, P, F, Q, B, u)
            x_next, P_next = KF_update(z, x_new, P_new, H, R)
        x, P = x_next, P_next
        # print(x)
        # print(P)
        xs.append(x)
        cov.append(P)
    xs, cov = np.array(xs), np.array(cov)
    return xs, cov


# 无迹卡尔曼滤波

# 动力学过程方程
def F_UKF(x, dt):
    mu = constant.GM_earth
    y = dynamics.mypropagation(x, dt, mu.value, dt)
    return y[-1,:]

# 观测方程
def H_UKF(y):
    H = np.array([[1., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.]])
    z = np.dot(H, y)
    return z

# 权重矩阵
def weights_UKF(alpha, beta, n, kappa):
    lambda_ = alpha**2 * (n + kappa) - n
    Wc = np.full(2*n + 1, 1. / (2*(n + lambda_)))
    Wm = np.full(2*n + 1, 1. / (2*(n + lambda_)))
    Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)
    return Wm, Wc

# sigma点
def sigmas_UKF(alpha, beta, kappa, X, P):
    n = len(X)
    lambda_ = alpha**2 * (n + kappa) - n
    sigmas = np.zeros((2*n+1, n))
    U = scipy.linalg.cholesky((n+lambda_)*P) # sqrt
    sigmas[0] = X
    for k in range (n):
        sigmas[k+1] = X + U[k]
        sigmas[n+k+1] = X - U[k]
    return sigmas

# 无迹转换
def UT(sigmas, Wm, Wc):
    x = np.dot(Wm, sigmas)
    kmax, n = sigmas.shape
    P = np.zeros((n, n))
    for k in range(kmax):
        y = sigmas[k] - x
        P += Wc[k] * np.outer(y, y)
    # eigenvalues = np.linalg.eigvals(P)
    # print(eigenvalues)
    return x, P

# 预测步
def UKF_predict(x, P, Q, Wm, Wc, alpha, beta, kappa, dt):
    n = len(x)
    sigmas = sigmas_UKF(alpha, beta, kappa, x, P)    
    sigmas_f = np.zeros((2*n+1, n))
    for i in range(2*n+1):
        sigmas_f[i] = F_UKF(sigmas[i], dt)
    x_new, P_new = UT(sigmas_f, Wm, Wc)
    P_new += Q
    return x_new, P_new, sigmas_f

# 更新步
def UKF_update(z, x, P, R, sigmas_f, Wm, Wc):
    nz = len(z)
    nx = len(x)
    sigmas_num = 2*nx+1
    sigmas_h = np.zeros((sigmas_num, nz))
    for i in range(sigmas_num):
        sigmas_h[i] = H_UKF(sigmas_f[i])
    
    zp, Pz = UT(sigmas_h, Wm, Wc)
    Pz += R

    Pxz = np.zeros((nx, nz))
    for i in range(sigmas_num):
        Pxz += Wc[i] * np.outer(sigmas_f[i] - x, sigmas_h[i] - zp)
    K = np.dot(Pxz, np.linalg.inv(Pz))

    x_next = x + np.dot(K, z - zp)
    P_next = P - np.dot(K, Pz).dot(K.T)
    # eigenvalues = np.linalg.eigvals(P_next)
    # print(eigenvalues)

    return x_next, P_next

# 执行无迹卡尔曼滤波
def UKF_run(x0, P0, Q, R, zs, alpha, beta, kappa, dt):
    xs, cov = [], []
    x, P = x0, P0
    n = len(x)
    Wm, Wc = weights_UKF(alpha, beta, n, kappa)
    for z in zs:
        if np.linalg.norm(z-zs[0])==0:
            sigmas_f = sigmas_UKF(alpha, beta, kappa, x, P)
            x_next, P_next = UKF_update(z, x, P, R, sigmas_f, Wm, Wc)
        else:
            x_new, P_new, sigmas_f = UKF_predict(x, P, Q, Wm, Wc, alpha, beta, kappa, dt)
            x_next, P_next = UKF_update(z, x_new, P_new, R, sigmas_f, Wm, Wc)
        x, P = x_next, P_next
        # eigenvalues = np.linalg.eigvals(P)
        # print(eigenvalues)
        # print(x)
        # print(P)
        xs.append(x)
        cov.append(P)
    xs, cov = np.array(xs), np.array(cov)
    return xs, cov