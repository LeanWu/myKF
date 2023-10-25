import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import poliastro.constants.general as constant

import dynamics
import kf

# 定轨测试UKF

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

# 生成测试数据
def compute_data(rv0, noise, count=1, dt=1.):
    "returns track, measurements" 
    mu = constant.GM_earth.value   
    xs, zs = [[],[]], [[],[]]
    xs = np.zeros((count,6))
    zs = np.zeros((count,3))
    xs = dynamics.mypropagation(rv0, (count-1)*dt, mu, dt)
    for i in range(count): 
        for j in range(3):
            zs[i,j] = xs[i,j] + randn() * noise
    return np.array(xs), np.array(zs)

# 卫星初始状态
Re=constant.R_earth.to(u.m)
a = Re+1000 * u.km
ecc = 0.0 * u.one
inc = 30 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu = 0 * u.deg
time=Time('2020-01-01 00:00:00',format='iso', scale='utc')

orb=Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, time)
r0=orb.r.to(u.m)
v0=orb.v.to(u.m/u.second)
rv0=np.array([r0.value[0],r0.value[1],r0.value[2],v0.value[0],v0.value[1],v0.value[2]])
np.set_printoptions(precision=2)
print('rv0:',rv0)
print('Orbit Period:',orb.period.to(u.hour))

# rv0=np.array([0.6, 0.5, -0.6, 0.7, -0.5, 0.3])

# 测试数据
noise=1e3
count = 200
dt = 1.
xs,zs = compute_data(rv0, noise, count, dt)
t = np.linspace(0, (count-1)*dt, count)
# debug = 1

# 轨道展示
test_plot=0
if test_plot==1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:,0], xs[:,1], xs[:,2], label="Track")
    ax.scatter(zs[:,0], zs[:,1], zs[:,2], c='red', s=5, facecolors='none', label='Measurement')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,xs[:,0])
    ax.scatter(t, zs[:,0], facecolors='none', c='red', s=8)
    plt.show()

# 滤波输入
x0 = np.array([zs[0,0], zs[0,1], zs[0,2], (zs[1,0]-zs[0,0])/dt, (zs[1,1]-zs[0,1])/dt, (zs[1,2]-zs[0,2])/dt])
P0 = np.diag([noise**2, noise**2, noise**2, noise**2, noise**2, noise**2])
v = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt]])
var = 1.
Q = np.dot(v, np.dot(var, v.T))
# Q = np.diag([noise, noise, noise, noise, noise, noise])
r = noise**2
R = np.diag([r, r, r])

# 无迹参数
alpha = 1
beta = 2
n = len(x0)
kappa = 3 - n


# 执行UKF
xs_ukf, cov_ukf = kf.UKF_run(x0, P0, Q, R, zs, alpha, beta, kappa, dt, F_UKF, H_UKF)
# 轨道展示
test_plot=1
if test_plot==1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs_1 = xs_ukf/Re.value
    zs_1 = zs/Re.value
    ax.plot(xs_1[:,0], xs_1[:,1], xs_1[:,2], label="UKF")
    ax.scatter(zs_1[:,0], zs_1[:,1], zs_1[:,2], c='red', s=1, facecolors='none', label='Measurement')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(t, cov_ukf[:,0,0]/1e6)
    axs[0,0].set_title(r'$\sigma^2_x$')
    axs[0,1].plot(t, cov_ukf[:,1,1]/1e6)
    axs[0,1].set_title(r'$\sigma^2_y$')
    axs[0,2].plot(t, cov_ukf[:,2,2]/1e6)
    axs[0,2].set_title(r'$\sigma^2_z$')
    axs[1,0].plot(t, cov_ukf[:,3,3]/1e6)
    axs[1,0].set_title(r'$\sigma^2_\dot{x}$')
    axs[1,1].plot(t, cov_ukf[:,4,4]/1e6)
    axs[1,1].set_title(r'$\sigma^2_\dot{y}$')
    axs[1,2].plot(t, cov_ukf[:,5,5]/1e6)
    axs[1,2].set_title(r'$\sigma^2_\dot{z}$')
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,(xs_ukf[:,0]-xs[:,0])/1e3,label="UKF")
    ax.plot(t,(zs[:,0]-xs[:,0])/1e3,label='Measurement')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,(xs_ukf[:,3]-xs[:,3])/1e3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.show()

debug = 1