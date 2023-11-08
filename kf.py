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

# 应用动力学过程方程
def apply_F_UKF(F_UKF, x, dt):
    return F_UKF(x, dt)

# 应用观测方程
def apply_H_UKF(H_UKF, y):
    return H_UKF(y)

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
    return x, P

# 预测步
def UKF_predict(x, P, Q, Wm, Wc, alpha, beta, kappa, dt, F_UKF):
    n = len(x)
    sigmas = sigmas_UKF(alpha, beta, kappa, x, P)    
    sigmas_f = np.zeros((2*n+1, n))
    for i in range(2*n+1):
        sigmas_f[i] = apply_F_UKF(F_UKF, sigmas[i], dt)
    x_new, P_new = UT(sigmas_f, Wm, Wc)
    P_new += Q
    return x_new, P_new, sigmas_f

# 更新步
def UKF_update(z, x, P, R, sigmas_f, Wm, Wc, H_UKF):
    nz = len(z)
    nx = len(x)
    sigmas_num = 2*nx+1
    sigmas_h = np.zeros((sigmas_num, nz))
    for i in range(sigmas_num):
        sigmas_h[i] = apply_H_UKF(H_UKF, sigmas_f[i])
    
    zp, Pz = UT(sigmas_h, Wm, Wc)
    Pz += R

    Pxz = np.zeros((nx, nz))
    for i in range(sigmas_num):
        Pxz += Wc[i] * np.outer(sigmas_f[i] - x, sigmas_h[i] - zp)
    K = np.dot(Pxz, np.linalg.inv(Pz))

    x_next = x + np.dot(K, z - zp)
    P_next = P - np.dot(K, Pz).dot(K.T)

    return x_next, P_next

# 执行无迹卡尔曼滤波
def UKF_run(x0, P0, Q, R, zs, alpha, beta, kappa, dt, F_UKF, H_UKF):
    xs, cov = [], []
    x, P = x0, P0
    n = len(x)
    Wm, Wc = weights_UKF(alpha, beta, n, kappa)
    for z in zs:
        if np.linalg.norm(z-zs[0])==0:
            sigmas_f = sigmas_UKF(alpha, beta, kappa, x, P)
            x_next, P_next = UKF_update(z, x, P, R, sigmas_f, Wm, Wc, H_UKF)
        else:
            x_new, P_new, sigmas_f = UKF_predict(x, P, Q, Wm, Wc, alpha, beta, kappa, dt, F_UKF)
            x_next, P_next = UKF_update(z, x_new, P_new, R, sigmas_f, Wm, Wc, H_UKF)
        x, P = x_next, P_next
        xs.append(x)
        cov.append(P)
    xs, cov = np.array(xs), np.array(cov)
    return xs, cov


# 容积卡尔曼滤波
def ksi_points(n):
    ksi = np.concatenate((np.sqrt(n)*np.eye(n),-np.sqrt(n)*np.eye(n)),axis=1)
    return ksi

# # 应用动力学过程方程
# def apply_F_CKF(F_CKF, x, dt):
#     return F_CKF(x, dt)

# # 应用观测方程
# def apply_H_CKF(H_CKF, y):
#     return H_CKF(y)

# # 容积点生成
# def points_CKF(x, P, ksi):
#     S = scipy.linalg.cholesky(P, lower=True)
#     n = len(x)
#     points = np.zeros((2*n,n))
#     for i in range(2*n):
#         points[i] = np.dot(S, ksi[:,i]) + x
#     return points

# # 容积转换
# def CT(points, W):
#     n = points.shape[1]    
#     x = np.dot(W, points)
#     P = np.zeros((n,n))
#     for i in range(2*n):
#         y = points[i] - x
#         P += W[i] * np.outer(y,y)
#     return x, P

# # 预测步
# def CKF_predict(x, P, Q, W, ksi, dt, F_CKF):
#     n = len(x)
#     points = points_CKF(x, P, ksi)
#     points_f = np.zeros((2*n, n))
#     for i in range(2*n):
#         points_f[i] = apply_F_CKF(F_CKF, points[i], dt)
#     x_new, P_new = CT(points_f, W)
#     P_new += Q
#     return x_new, P_new

# # 更新步
# def CKF_update(z, x, P, R, W, ksi, H_CKF):
#     nz = len(z)
#     nx = len(x)
#     points_num = 2*nx
#     points = points_CKF(x, P, ksi)
#     points_h = np.zeros((points_num, nz))
#     for i in range(points_num):
#         points_h[i] = apply_H_UKF(H_CKF, points[i])
    
#     zp, Pz =  CT(points_h, W)
#     Pz += R

#     Pxz = np.zeros((nx, nz))
#     for i in range(points_num):
#         Pxz += W[i] * np.outer(points[i] - x, points_h[i] - zp)
#     K = np.dot(Pxz, np.linalg.inv(Pz))

#     x_next = x + np.dot(K, z - zp)
#     P_next = P - np.dot(K, Pz).dot(K.T)
#     return x_next, P_next

# # 执行容积卡尔曼滤波
# def CKF_run(x0, P0, Q, R, zs, dt, F_CKF, H_CKF):
#     xs, cov = [], []
#     x, P = x0, P0
#     n = len(x)
#     ksi = ksi_points(n)
#     W = np.ones(2*n)/2/n
#     for z in zs:
#         if np.linalg.norm(z-zs[0])==0:
#             x_next, P_next = CKF_update(z, x, P, R, W, ksi, H_CKF)
#         else:
#             x_new, P_new = CKF_predict(x, P, Q, W, ksi, dt, F_CKF)
#             x_next, P_next = CKF_update(z, x_new, P_new, R, W, ksi, H_CKF)
#         x, P = x_next, P_next
#         xs.append(x)
#         cov.append(P)
#     xs, cov = np.array(xs), np.array(cov)
#     return xs, cov

# 平方根容积卡尔曼滤波

# 应用动力学过程方程
def apply_F_SRCKF(F_SRCKF, x, dt):
    return F_SRCKF(x, dt)

# 应用观测方程
def apply_H_SRCKF(H_SRCKF, y):
    return H_SRCKF(y)

# 容积点生成
def points_SRCKF(x, S, ksi):
    n = len(x)
    points = np.zeros((2*n,n))
    for i in range(2*n):
        points[i] = np.dot(S, ksi[:,i]) + x
    return points

# 平方根容积转换
def SRCT(points, W, noiseMatrix):
    n = points.shape[1]    
    x = np.dot(W,points)
    P = np.zeros((n,n))
    x_residual = (points - x)/np.sqrt(2*n)
    noiseMatrix = noiseMatrix + 1e-12*np.eye(len(noiseMatrix),len(noiseMatrix))
    M = np.concatenate((x_residual.T, scipy.linalg.cholesky(noiseMatrix, lower=True)), axis=1)
    Q, R = np.linalg.qr(M.T)
    S = R.T
    return x, S

# 预测步
def SRCKF_predict(x, S, Q, W, ksi, dt, F_SRCKF):
    n = len(x)
    points = points_SRCKF(x, S, ksi)
    points_f = np.zeros((2*n, n))
    for i in range(2*n):
        points_f[i] = apply_F_SRCKF(F_SRCKF, points[i], dt)
    x_new, S_new = SRCT(points_f, W, Q)
    return x_new, S_new

# 更新步
def SRCKF_update(z, x, S, R, W, ksi, H_SRCKF):
    nz = len(z)
    nx = len(x)
    points_num = 2*nx
    points = points_SRCKF(x, S, ksi)
    points_h = np.zeros((points_num, nz))
    for i in range(points_num):
        points_h[i] = apply_H_UKF(H_SRCKF, points[i])
    
    zp, Sz =  SRCT(points_h, W, R)

    Pz = np.dot(Sz, Sz.T)
    x_residual = (points - x)/np.sqrt(2*nx)
    z_residual = (points_h - zp)/np.sqrt(2*nx)
    Pxz = np.dot(x_residual.T,z_residual)
    K = np.dot(Pxz, np.linalg.inv(Pz))

    x_next = x + np.dot(K, z - zp)
    M = np.concatenate((x_residual.T - np.dot(K,z_residual.T), np.dot(K,scipy.linalg.cholesky(R, lower=True))), axis=1)
    Q, R = np.linalg.qr(M.T)
    S_next = R.T
    return x_next, S_next

# 执行平方根容积卡尔曼滤波
def SRCKF_run(x0, P0, Q, R, zs, dt, F_SRCKF, H_SRCKF):
    xs, cov = [], []
    x, P = x0, P0
    S = scipy.linalg.cholesky(P, lower=True)
    n = len(x)
    ksi = ksi_points(n)
    W = np.ones(2*n)/2/n
    for z in zs:
        if np.linalg.norm(z-zs[0])==0:
            x_next, S_next = SRCKF_update(z, x, S, R, W, ksi, H_SRCKF)
        else:
            x_new, S_new = SRCKF_predict(x, S, Q, W, ksi, dt, F_SRCKF)
            x_next, S_next = SRCKF_update(z, x_new, S_new, R, W, ksi, H_SRCKF)
        x, S = x_next, S_next
        P = np.dot(S,S.T)
        xs.append(x)
        cov.append(P)
    xs, cov = np.array(xs), np.array(cov)
    return xs, cov


# 观测站SRCKF
# 应用观测方程
def apply_H_SRCKF_station(H_SRCKF_station, y, t, stationPos0):
    return H_SRCKF_station(y, t, stationPos0)
# 更新步
def SRCKF_update_station(z, x, S, R, W, ksi, H_SRCKF_station, t, stationPos0):
    nz = len(z)
    nx = len(x)
    points_num = 2*nx
    points = points_SRCKF(x, S, ksi)
    points_h = np.zeros((points_num, nz))
    for i in range(points_num):
        points_h[i] = apply_H_SRCKF_station(H_SRCKF_station, points[i], t, stationPos0)
    
    zp, Sz =  SRCT(points_h, W, R)

    Pz = np.dot(Sz, Sz.T)
    x_residual = (points - x)/np.sqrt(2*nx)
    z_residual = (points_h - zp)/np.sqrt(2*nx)
    Pxz = np.dot(x_residual.T,z_residual)
    K = np.dot(Pxz, np.linalg.inv(Pz))

    x_next = x + np.dot(K, z - zp)
    M = np.concatenate((x_residual.T - np.dot(K,z_residual.T), np.dot(K,scipy.linalg.cholesky(R, lower=True))), axis=1)
    Q, R = np.linalg.qr(M.T)
    S_next = R.T
    return x_next, S_next
# 执行平方根容积卡尔曼滤波
def SRCKF_run_station(x0, P0, Q, R, zs, dt, F_SRCKF, H_SRCKF_station, stationPos0):
    xs, cov = [], []
    x, P = x0, P0
    S = scipy.linalg.cholesky(P, lower=True)
    n = len(x)
    ksi = ksi_points(n)
    W = np.ones(2*n)/2/n
    t = 0
    for z in zs:
        if np.linalg.norm(z-zs[0])==0:
            x_next, S_next = SRCKF_update_station(z, x, S, R, W, ksi, H_SRCKF_station, t, stationPos0)
        else:
            x_new, S_new = SRCKF_predict(x, S, Q, W, ksi, dt, F_SRCKF)
            x_next, S_next = SRCKF_update_station(z, x_new, S_new, R, W, ksi, H_SRCKF_station, t, stationPos0)
        x, S = x_next, S_next
        t += dt
        P = np.dot(S,S.T)
        xs.append(x)
        cov.append(P)
    xs, cov = np.array(xs), np.array(cov)
    return xs, cov

# RTS-smoother
def rts_smoother(Xs, Ps, F_function, Q_noise, dt):
    num, n = Xs.shape
    ksi = ksi_points(n)
    W = np.ones(2*n)/2/n
    
    K = np.zeros((num,n,n))
    x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()

    for i in range(num-2,-1,-1):
        # 预测步
        x_now = x[i]
        P_now = P[i]
        S_now = scipy.linalg.cholesky(P_now, lower=True)
        points = points_SRCKF(x_now, S_now, ksi)
        points_f = np.zeros((2*n, n))
        for j in range(2*n):
            points_f[j] = apply_F_SRCKF(F_function, points[j], dt)
        x_new, S_new = SRCT(points_f, W, Q_noise)

        # 预测步修正当前步
        Pp[i] = np.dot(S_new,S_new.T) 
        # points_num = 2*n
        # points = points_SRCKF(x_now, S_now, ksi)
        # points_h = np.zeros((points_num, n))
        # for j in range(points_num):
        #     points_h[j] = apply_F_SRCKF(F_function, points[j], dt)
        
        # zp, Sz =  SRCT(points_h, W, Q_noise)

        Pz = np.dot(S_new, S_new.T)
        x_residual = (points - x_now)/np.sqrt(2*n)
        z_residual = (points_f - x_new)/np.sqrt(2*n)
        Pxz = np.dot(x_residual.T,z_residual)
        K[i] = np.dot(Pxz, np.linalg.inv(Pz))

        x[i] += np.dot(K[i], x[i+1]-x_new)
        P[i] += np.dot(K[i], np.dot(P[i+1]-Pp[i], K[i].T))

    return x, P