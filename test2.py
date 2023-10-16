import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import kf

# 生成测试数据
def compute_data(x0, y0, vel, theta_deg, noise, count=1, dt=1.):
    "returns track, measurements"    
    g = -9.8
    theta = math.radians(theta_deg)
    vx = math.cos(theta) * vel
    vy = math.sin(theta) * vel
    xs, zs = [[],[]], [[],[]]
    for i in range(count):
        t = i * dt
        x = vx * t + x0
        y = 0.5 * g * t **2 + vy * t + y0  
        xs[0].append(x)
        xs[1].append(y)
        zs[0].append(x + randn() * noise[0])
        zs[1].append(y + randn() * noise[1])
    return np.array(xs).T, np.array(zs).T

# 过程噪声矩阵(一阶)
def Q_discrete_white_noise(v, var):
    return np.dot(v, np.dot(var, v.T))

noise = np.array([1., 1.])
x0, y0 = 0, 15
vel = 100
theta_deg = 60
count = 70
dt = 0.25
xt, zs = compute_data(x0, y0, vel, theta_deg, noise, count, dt)

F = np.array([[1., dt, 0., 0.], # x = x0 + dx*dt
            [0., 1., 0., 0.], # dx = dx0
            [0., 0., 1., dt], # y = y0 + dy*dt
            [0., 0., 0., 1.]]) # dy = dy0
H = np.array([[1., 0., 0., 0.],
            [0., 0., 1., 0.]])
v = np.array([[0.5 * dt ** 2], [dt]])
var = 1.
q = Q_discrete_white_noise(v, var)
Q = np.block([[q, np.zeros_like(q)],
              [np.zeros_like(q), q]])
R = np.diag([0.2, 0.2])
# x0 = np.array([x0, math.cos(math.radians(theta_deg)) * vel, y0, math.sin(math.radians(theta_deg)) * vel] ).T
x0 = np.array([0., 0., 0., 0.]).T
P0 = np.diag([100., 25., 100., 25.])
B = np.array([0., 0., 0., dt]).T
u = -9.8
xs, cov = kf.KF_run(x0, P0, F, H, Q, R, zs, B, u)

plt.plot(xs[:,0], xs[:,2], color='blue', label='Kalman filter')
plt.scatter(zs[:,0], zs[:,1], facecolors='none', edgecolors='red', label='Measurements')
plt.plot(xt[:,0], xt[:,1], color='black', linestyle='dashed', label='Real track')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')
# plt.xlim(-10, 900)
# plt.ylim(0, 500)
plt.show()

t = np.linspace(0, dt * (count-1), count)
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(t, cov[:,0,0])
axs[0,0].set_title(r'$\sigma^2_x$')
axs[0,1].plot(t, cov[:,1,1])
axs[0,1].set_title(r'$\sigma^2_\dot{x}$')
axs[1,0].plot(t, cov[:,2,2])
axs[1,0].set_title(r'$\sigma^2_y$')
axs[1,1].plot(t, cov[:,3,3])
axs[1,1].set_title(r'$\sigma^2_\dot{y}$')
plt.tight_layout()
plt.show()