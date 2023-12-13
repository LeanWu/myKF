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

# 观测变量为多个测站的测角测距的小推力定轨

# 动力学过程方程
# def F_1(x, dt):
#     mu = constant.GM_earth
#     y = dynamics.mypropagation(x, dt, mu.value, -1)
#     return y[-1,:]

# def F_2(x, dt):
#     alpha = 1e-3
#     mu = constant.GM_earth
#     y = dynamics.mypropagation_a(x, dt, mu.value, -1, alpha)
#     return y[-1,:]

def F_3(x, dt):
    alpha = 1e-3
    mu = constant.GM_earth
    y = dynamics.mypropagation_jerk(x, dt, mu.value, -1, alpha)
    return y[-1,:]

# 观测方程
def H_station(y, t, stationPos0):
    omega = 2*np.pi/(23*3400+56*60+4)
    stationPos = np.zeros(2)
    stationPos[1] = stationPos0[1]
    stationPos[0] = (stationPos0[0]+omega*t) % (2*np.pi)
    obs = observe.get_observation(stationPos, y[0:3])
    return obs

# 读取数据
data1=np.loadtxt('.\data\station_observe_data_6.txt')
data2=np.loadtxt('.\data\station_observe_para_6.txt')
t = data1[:,0]
xs = data1[:,1:7]
zs1 = data1[:,7:10]
zs2 = data1[:,10:13]
zs3 = data1[:,13:16]
# noise_rho,noise_angle,stationPos0[0],stationPos0[1],a_test,count1,count2,dt
noise_rho = data2[0]
noise_angle = data2[1]
stationPos10 = np.array([data2[2], data2[3]])
stationPos20 = np.array([data2[4], data2[5]])
stationPos30 = np.array([data2[6], data2[7]])
a_test = data2[8]
count1 = int(data2[9])
count2 = int(data2[10])
dt = data2[11]

zs=np.zeros([3,len(zs1),3])
zs[0,:,:]=zs1
zs[1,:,:]=zs2
zs[2,:,:]=zs3
stationPos0 = np.array([stationPos10, stationPos20, stationPos30])

# 初值计算
r0 = observe.get_r(stationPos10, zs1[0,:])
omega = 2*np.pi/(23*3400+56*60+4)
stationPos1 = np.array([stationPos10[0]+dt*omega,stationPos10[1]])
r1 = observe.get_r(stationPos1, zs1[1,:])

num=len(t)
r_observe = np.zeros([num,3])
for i in range(num):
    stationPos = np.array([stationPos10[0]+i*dt*omega,stationPos10[1]])
    r_observe[i]=observe.get_r(stationPos, zs1[i,:])

# 滤波输入
x0 = np.array([r0[0], r0[1], r0[2], (r1[0]-r0[0])/dt, (r1[1]-r0[1])/dt, (r1[2]-r0[2])/dt, 0, 0, 0, 0, 0, 0])
P0 = np.diag([1e3,1e3,1e3,10,10,10,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2])
v = np.array([[dt ** 3 / 6], [dt ** 3 / 6], [dt ** 3 / 6], [0.5 * dt ** 2], [0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt], [dt], [1], [1], [1]])
var = 1e-6
Q = np.dot(v, np.dot(var, v.T))
R = np.diag([noise_rho**2, noise_angle**2, noise_angle**2])
# 执行滤波
xs_ckf1, cov_ckf1 = kf.SRCKF_run_multistation(x0, P0, Q, R, zs[0:1,:,:], dt, F_3, H_station, stationPos0[0:1,:])
xs_ckf2, cov_ckf2 = kf.SRCKF_run_multistation(x0, P0, Q, R, zs[0:2,:,:], dt, F_3, H_station, stationPos0[0:2,:])
xs_ckf3, cov_ckf3 = kf.SRCKF_run_multistation(x0, P0, Q, R, zs, dt, F_3, H_station, stationPos0)


# 保存数据
t=t.reshape(-1,1)
data = np.hstack((t,xs,r_observe,xs_ckf1,xs_ckf2,xs_ckf3))
np.savetxt('.\data\calculte_result_6.txt',(data))
print('finish')

# 轨道展示
test_plot=0
if test_plot==1:
    color3='lightgreen'
    label3='Jerk-SRCKF'

    r3=(xs_ckf1[:,0]-xs[:,0])/1e3
    v3=(xs_ckf1[:,3]-xs[:,3])/1e3
    a_ckf = np.zeros(count1+count2-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,r3,label=label3,c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of x(km)')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,v3,label=label3,c=color3)
    ax.set_xlabel('t(s)')
    ax.set_ylabel('error of vx(km/s)')
    plt.legend()
    plt.show()

debug = 1