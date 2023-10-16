import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import kf

# 生成测试数据
def compute_data(z_var, count=1, dt=1.):
    "returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var) 
    xs, zs = [], []
    for _ in range(count):
        v = vel 
        x += v*dt        
        xs.append(x)
        zs.append(x + randn() * z_std)
    return np.array(xs), np.array(zs)

z_var = 0.1
count = 50
dt = 1
xt, zs = compute_data(z_var, count)
t = np.linspace(0, len(xt)*dt, len(xt))

F = np.array([[1., dt],
              [0., 1.]])
H = np.array([[1., 0.]])
Q = np.array([[0.0025, 0.005],
              [0.005, 0.01]])
R = np.diag([10.])
x0 = np.array([0., 0.]).T
P0 = np.diag([500., 49.])

xs, cov = kf.KF_run(x0, P0, F, H, Q, R, zs)

plt.scatter(t, zs, facecolors='none', edgecolors='black')
plt.plot(t, xt, color='black', linestyle='dashed')
plt.plot(t, xs[:,0], color='blue')
plt.show()