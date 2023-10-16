import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

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