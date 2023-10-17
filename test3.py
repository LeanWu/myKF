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

# 生成测试数据
def compute_data(rv0, noise, count=1, dt=1.):
    "returns track, measurements"    
    xs, zs = [[],[]], [[],[]]
    xs = np.zeros((count,6))
    zs = np.zeros((count,3))
    xs = dynamics.mypropagation(rv0, (count-1)*dt, constant.GM_earth.value, dt)
    for i in range(count): 
        for j in range(3):
            zs[i,j] = xs[i,j] + randn() * noise
    return np.array(xs), np.array(zs)

# 卫星初始状态
Re=constant.R_earth.to(u.m)
a = Re+20000 * u.km
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
print('rv0:',rv0)
print('Orbit Period:',orb.period.to(u.hour))

# 测试数据
noise=1e5
count = 72
dt = 600.
xs,zs = compute_data(rv0, noise, count, dt)
t = np.linspace(0, (count-1)*dt, count)
debug = 1

# 轨道展示
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