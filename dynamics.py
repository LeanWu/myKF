import numpy as np
import math
from scipy.integrate import odeint

# 动力学方程
def dynamics(rv,t,mu):
    drdv=np.zeros(6)
    r_norm=np.linalg.norm(rv[:3])
    C=-mu/r_norm**3
    for i in range(6):
        if i<3:
            drdv[i]=rv[i+3]
        else:
            drdv[i]=C*rv[i-3]
    return drdv


# 积分求解
def mypropagation(rv0,dt,mu,t_step):
    if t_step==-1:
        num=2
    else:
        num=int(dt/t_step+1)    
    t=np.linspace(0,dt,num)
    new_dynamics=lambda rv,t:dynamics(rv,t,mu)
    rv=odeint(new_dynamics,rv0,t)
    return rv

# test：right rv:[ 0.86960342 -0.16077931 -0.05226604 -0.26656077 -0.69818329  0.70599091]
# rv0=np.array([0.6, 0.5, -0.6, 0.7, -0.5, 0.3])
# rv=mypropagation(rv0,1,1,1)
# print(rv)

# 旋转矩阵
def xRotationMatrix(angle):
    return np.array([ [1, 0, 0],
                      [0, math.cos(angle), -math.sin(angle)],
                      [0, math.sin(angle), math.cos(angle)]], dtype='float')

def yRotationMatrix(angle):
    return np.array([ [math.cos(angle), 0, math.sin(angle)],
                      [0, 1, 0],
                      [-math.sin(angle), 0, math.cos(angle)]], dtype='float')

def zRotationMatrix(angle):
    return np.array([ [math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]], dtype='float')
