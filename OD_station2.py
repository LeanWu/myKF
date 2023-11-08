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
import observe

# 观测变量为测角测距的小推力定轨

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
def H_station(y, t, stationPos0):
    omega = 2*np.pi/(23*3400+56*60+4)
    stationPos = np.zeros(2)
    stationPos[1] = stationPos0[1]
    stationPos[0] = (stationPos0[0]+omega*t) % (2*np.pi)
    obs = observe.get_observation(stationPos, y[0:3])
    return obs

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
data1=np.loadtxt('.\data\station_observe_data_1.txt')
data2=np.loadtxt('.\data\station_observe_para_1.txt')
t = data1[:,0]
xs = data1[:,1:7]
zs = data1[:,7:10]
# noise_rho,noise_angle,stationPos0[0],stationPos0[1],a_test,count1,count2,dt
noise_rho = data2[0]
noise_angle = data2[1]
stationPos0 = np.array([data2[2], data2[3]])
a_test = data2[4]
count1 = int(data2[5])
count2 = int(data2[6])
dt = data2[7]

# 初值计算
num=len(t)
omega = 2*np.pi/(23*3400+56*60+4)
r_observe = np.zeros([num,3])
for i in range(num):
    stationPos = np.array([stationPos0[0]+i*dt*omega,stationPos0[1]])
    r_observe[i]=observe.get_r(stationPos, zs[i,:])

# 滤波输入
x0 = np.array([r_observe[0,0], r_observe[0,1], r_observe[0,2], (r_observe[1,0]-r_observe[0,0])/dt, (r_observe[1,1]-r_observe[0,1])/dt, (r_observe[1,2]-r_observe[0,2])/dt])
P0 = np.diag([1e3,1e3,1e3,10,10,10])
v = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt]])
var = 1
Q = np.dot(v, np.dot(var, v.T))
R = np.diag([1e3**2, 1e3**2, 1e3**2])
# 执行CKF
xs_ckf1, cov_ckf1 = kf.SRCKF_run(x0, P0, Q, R, r_observe, dt, F_1, H_1)

# 滤波输入
x0 = np.array([r_observe[0,0], r_observe[0,1], r_observe[0,2], (r_observe[1,0]-r_observe[0,0])/dt, (r_observe[1,1]-r_observe[0,1])/dt, (r_observe[1,2]-r_observe[0,2])/dt, 0, 0, 0, 0, 0, 0])
P0 = np.diag([1e3,1e3,1e3,10,10,10,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2])
v = np.array([[dt ** 3 / 6], [dt ** 3 / 6], [dt ** 3 / 6], [0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt], [1], [1], [1]])
var = 1
Q = np.dot(v, np.dot(var, v.T))
R = np.diag([1e3**2, 1e3**2, 1e3**2])
# 执行滤波
xs_ckf3, cov_ckf3 = kf.SRCKF_run(x0, P0, Q, R, r_observe, dt, F_3, H_3)

# 保存数据
t=t.reshape(-1,1)
data = np.hstack((t,xs,r_observe,xs_ckf1,xs_ckf3))
np.savetxt('.\data\calculte_result2.txt',(data))

# 轨道展示
test_plot=1
if test_plot==1:
    color1='orange'
    color3='lightgreen'
    label1='SRCKF'
    label3='Jerk-SRCKF'

    r1=(xs_ckf1[:,0]-xs[:,0])/1e3
    v1=(xs_ckf1[:,3]-xs[:,3])/1e3
    r3=(xs_ckf3[:,0]-xs[:,0])/1e3
    v3=(xs_ckf3[:,3]-xs[:,3])/1e3
    a_ckf = np.zeros(count1+count2-1)
    for i in range(count1+count2-1):
        a_ckf[i]=np.linalg.norm(xs_ckf1[i,6:9])
    a2=a_ckf-a_test

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,r1,label=label1,c=color1)
    ax.plot(t,r3,label=label3,c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,v1,label=label1,c=color1)
    ax.plot(t,v3,label=label3,c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.legend()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(t,a2,label=label2,c=color2)
    # ax.set_xlabel('t(s)')
    # ax.set_ylabel('error of a(m/s^2)')
    # plt.legend()
    # plt.show()

debug = 1