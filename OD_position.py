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

# 观测变量为三个位置的小推力定轨

# 动力学过程方程
def F_1(x, dt):
    mu = constant.GM_earth
    y = dynamics.mypropagation(x, dt, mu.value, dt)
    return y[-1,:]

def F_2(x, dt):
    alpha = 1e-3
    mu = constant.GM_earth
    y = dynamics.mypropagation_a(x, dt, mu.value, dt, alpha)
    return y[-1,:]

def F_3(x, dt):
    alpha = 1e-3
    mu = constant.GM_earth
    y = dynamics.mypropagation_jerk(x, dt, mu.value, dt, alpha)
    return y[-1,:]

# 观测方程
def H_1(y):
    H = np.array([[1., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.]])
    z = np.dot(H, y)
    return z

def H_2(y):
    H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 0., 0., 0.]])
    z = np.dot(H, y)
    return z

def H_3(y):
    H = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    z = np.dot(H, y)
    return z

# 读取数据
data1=np.loadtxt('.\data\point_observe_data_1.txt')
data2=np.loadtxt('.\data\point_observe_para_1.txt')
t = data1[:,0]
xs = data1[:,1:7]
zs = data1[:,7:10]
noise = data2[0]
a_test = data2[1]
count1 = int(data2[2])
count2 = int(data2[3])
dt = data2[4]

# 滤波输入
x0 = np.array([zs[0,0], zs[0,1], zs[0,2], (zs[1,0]-zs[0,0])/dt, (zs[1,1]-zs[0,1])/dt, (zs[1,2]-zs[0,2])/dt, 0, 0, 0, 0, 0, 0])
P0 = np.diag([noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2])
v = np.array([[dt ** 3 / 6], [dt ** 3 / 6], [dt ** 3 / 6], [0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt], [1], [1], [1]])
var = 1e-6
Q = np.dot(v, np.dot(var, v.T))
r = noise**2
R = np.diag([r, r, r])
# 无迹参数
alpha = 1
beta = 2
n = len(x0)
kappa = 3 - n
# 执行滤波
xs_ukf, cov_ukf = kf.UKF_run(x0, P0, Q, R, zs, alpha, beta, kappa, dt, F_3, H_3)
xs_ckf, cov_ckf = kf.SRCKF_run(x0, P0, Q, R, zs, dt, F_3, H_3)
# xs_ukf = xs_ckf

# 轨道展示
test_plot=1
if test_plot==1:
    color1='lightgreen'
    color2='orange'
    label1='UKF'
    label2='SRCKF'

    r1=(xs_ukf[:,0]-xs[:,0])/1e3
    r2=(xs_ckf[:,0]-xs[:,0])/1e3
    v1=(xs_ukf[:,3]-xs[:,3])/1e3
    v2=(xs_ckf[:,3]-xs[:,3])/1e3
    a_ukf = np.zeros(count1+count2-1)
    a_ckf = np.zeros(count1+count2-1)
    for i in range(count1+count2-1):
        a_ukf[i]=np.linalg.norm(xs_ukf[i,6:9])
        a_ckf[i]=np.linalg.norm(xs_ckf[i,6:9])
    a1=a_ukf-a_test
    a2=a_ckf-a_test

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,r1,label=label1,c=color1)
    ax.plot(t,r2,label=label2,c=color2)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,v1,label=label1,c=color1)
    ax.plot(t,v2,label=label2,c=color2)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.plot(t,a1,label=label1,c=color1)
    ax.plot(t,a2,label=label2,c=color2)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of a(m/s^2)')
    plt.legend()
    plt.show()

debug = 1