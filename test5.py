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

# 小推力定轨测试(SRCFK)

# 生成测试数据
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

def compute_data_thrust(rv0, noise, a, count=1, dt=1.):
    "returns track, measurements" 
    mu = constant.GM_earth.value   
    xs, zs = [[],[]], [[],[]]
    xs = np.zeros((count,6))
    zs = np.zeros((count,3))
    xs = dynamics.mypropagation_thrust(rv0, (count-1)*dt, mu, dt, a)
    for i in range(count): 
        for j in range(3):
            zs[i,j] = xs[i,j] + randn() * noise
    return np.array(xs), np.array(zs)

# 卫星初始状态
Re=constant.R_earth.to(u.m)
a = Re + 1000 * u.km
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


# 测试数据
noise = 1e3
a_test = 1
count1 = 1
count2 = 4000
dt = 1
xs1,zs1 = compute_data(rv0, noise, count1, dt)
rv1 = xs1[-1,:]
xs2,zs2 = compute_data_thrust(rv1, noise, a_test, count2, dt)
xs = np.concatenate((xs1[0:-1,:], xs2), axis=0)
zs = np.concatenate((zs1[0:-1,:], zs2), axis=0)
t = np.linspace(0, (count1+count2-2)*dt, count1+count2-1)
# debug = 1

# 轨道展示
test_plot=0
if test_plot==1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:,0], xs[:,1], xs[:,2], label="Track")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax_test = np.zeros(count2)
    # for i in range(count2):
    #     vnorm = np.linalg.norm(xs[i,3:6])
    #     ax_test[i] = a_test*xs[i,5]/vnorm
    # ax.plot(t,ax_test)
    # plt.show()

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

# 滤波输入
x0 = np.array([zs[0,0], zs[0,1], zs[0,2], (zs[1,0]-zs[0,0])/dt, (zs[1,1]-zs[0,1])/dt, (zs[1,2]-zs[0,2])/dt])
P0 = np.diag([noise**2, noise**2, noise**2, noise**2, noise**2, noise**2])
v = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt]])
var = 1
Q = np.dot(v, np.dot(var, v.T))
r = noise**2
R = np.diag([r, r, r])
# 无迹参数
alpha = 1
beta = 2
n = len(x0)
kappa = 3 - n
# 执行CKF
xs_ckf1, cov_ckf1 = kf.SRCKF_run(x0, P0, Q, R, zs, dt, F_1, H_1)

# 滤波输入
x0 = np.array([zs[0,0], zs[0,1], zs[0,2], (zs[1,0]-zs[0,0])/dt, (zs[1,1]-zs[0,1])/dt, (zs[1,2]-zs[0,2])/dt, 0, 0, 0])
P0 = np.diag([noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2, noise**2])
v = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt], [1], [1], [1]])
var = 1e-3
Q = np.dot(v, np.dot(var, v.T))
r = noise**2
R = np.diag([r, r, r])
# 无迹参数
alpha = 1
beta = 2
n = len(x0)
kappa = 3 - n
# 执行CKF
xs_ckf2, cov_ckf2 = kf.SRCKF_run(x0, P0, Q, R, zs, dt, F_2, H_2)

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
# 执行CKF
xs_ckf3, cov_ckf3 = kf.SRCKF_run(x0, P0, Q, R, zs, dt, F_3, H_3)

# 轨道展示
test_plot=1
if test_plot==1:
    color1='cornflowerblue'
    color2='lightgreen'
    color3='orange'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,(xs_ckf1[:,0]-xs[:,0])/1e3,label="SRCKF",c=color1)
    ax.plot(t,(xs_ckf2[:,0]-xs[:,0])/1e3,label='Singer-SRCKF',c=color2)
    ax.plot(t,(xs_ckf3[:,0]-xs[:,0])/1e3,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t[300:-1],(xs_ckf2[300:-1,0]-xs[300:-1,0])/1e3,label='Singer-SRCKF',c=color2)
    ax.plot(t[300:-1],(xs_ckf3[300:-1,0]-xs[300:-1,0])/1e3,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,(xs_ckf1[:,3]-xs[:,3])/1e3,label="SRCKF",c=color1)
    ax.plot(t,(xs_ckf2[:,3]-xs[:,3])/1e3,label='Singer-SRCKF',c=color2)
    ax.plot(t,(xs_ckf3[:,3]-xs[:,3])/1e3,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.legend()
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t[300:-1],(xs_ckf2[300:-1,3]-xs[300:-1,3])/1e3,label='Singer-SRCKF',c=color2)
    ax.plot(t[300:-1],(xs_ckf3[300:-1,3]-xs[300:-1,3])/1e3,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    a_ckf2 = np.zeros(count1+count2-1)
    a_ckf3 = np.zeros(count1+count2-1)
    for i in range(count1+count2-1):
        a_ckf2[i]=np.linalg.norm(xs_ckf2[i,6:9])
        a_ckf3[i]=np.linalg.norm(xs_ckf3[i,6:9])
    ax.plot(t,a_ckf2-a_test,label='Singer-SRCKF',c=color2)
    ax.plot(t,a_ckf3-a_test,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of a(m/s^2)')
    plt.legend()
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t[300:-1],a_ckf2[300:-1]-a_test,label='Singer-SRCKF',c=color2)
    ax.plot(t[300:-1],a_ckf3[300:-1]-a_test,label='Jerk-SRCKF',c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of a(m/s^2)')
    plt.legend()
    plt.show()

debug = 1