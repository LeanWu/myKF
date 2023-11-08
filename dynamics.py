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
# rv0=np.array([0.86960342, -0.16077931, -0.05226604, -0.26656077, -0.69818329,  0.70599091])
# rv=mypropagation(rv0,-1,1,-1)
# print(rv)

# 加推力后的动力学方程(恒加速度)
def dynamics_thrust(rv,t,mu,a):
    drdv=np.zeros(6)
    r_norm=np.linalg.norm(rv[:3])
    v_norm=np.linalg.norm(rv[3:6])
    C=-mu/r_norm**3
    for i in range(6):
        if i<3:
            drdv[i]=rv[i+3]
        else:
            drdv[i]=C*rv[i-3]+a*rv[i]/v_norm
    return drdv

# 推力积分求解(恒加速度)
def mypropagation_thrust(rv0,dt,mu,t_step,a):
    if t_step==-1:
        num=2
    else:
        num=int(dt/t_step+1)    
    t=np.linspace(0,dt,num)
    new_dynamics_thrust=lambda rv,t:dynamics_thrust(rv,t,mu,a)
    rv=odeint(new_dynamics_thrust,rv0,t)
    return rv

# 状态含加速度的动力学方程
def dynamics_a(rva,t,mu,alpha):
    drdvda=np.zeros(9)
    r_norm=np.linalg.norm(rva[:3])
    v_norm=np.linalg.norm(rva[3:6])
    C=-mu/r_norm**3
    for i in range(9):
        if i<3:
            drdvda[i]=rva[i+3]
        elif i<6:
            drdvda[i]=C*rva[i-3]+rva[i+3]
        else:
            drdvda[i]=-alpha*rva[i]
    return drdvda

# 积分求解
def mypropagation_a(rva0,dt,mu,t_step,alpha):
    if t_step==-1:
        num=2
    else:
        num=int(dt/t_step+1)    
    t=np.linspace(0,dt,num)
    new_dynamics=lambda rva,t:dynamics_a(rva,t,mu,alpha)
    rva=odeint(new_dynamics,rva0,t)
    return rva

# 状态含jerk的动力学方程
def dynamics_jerk(rvaj,t,mu,alpha):
    drdvdadj=np.zeros(12)
    r_norm=np.linalg.norm(rvaj[:3])
    v_norm=np.linalg.norm(rvaj[3:6])
    C=-mu/r_norm**3
    for i in range(12):
        if i<3:
            drdvdadj[i]=rvaj[i+3]
        elif i<6:
            drdvdadj[i]=C*rvaj[i-3]+rvaj[i+3]
        elif i<9:
            drdvdadj[i]=rvaj[i+3]
        else:
            drdvdadj[i]=-alpha*rvaj[i]            
    return drdvdadj

# 积分求解
def mypropagation_jerk(rvaj0,dt,mu,t_step,alpha):
    if t_step==-1:
        num=2
    else:
        num=int(dt/t_step+1)    
    t=np.linspace(0,dt,num)
    new_dynamics=lambda rvaj,t:dynamics_jerk(rvaj,t,mu,alpha)
    rvaj=odeint(new_dynamics,rvaj0,t)
    return rvaj