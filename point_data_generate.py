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
noise = 1
a_test = 1e-5
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

# 保存数据
t=t.reshape(-1,1)
data1 = np.hstack((t,xs,zs))
np.savetxt('point_observe_data.txt',(data1))
data2 = np.vstack((noise,a_test,count1,count2,dt))
np.savetxt('point_observe_para.txt',(data2))

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