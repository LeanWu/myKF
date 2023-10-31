import numpy as np
import poliastro.constants.general as constant

# 旋转矩阵
def xRotationMatrix(angle):
    return np.array([ [1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]], dtype='float')

def yRotationMatrix(angle):
    return np.array([ [np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], dtype='float')

def zRotationMatrix(angle):
    return np.array([ [np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], dtype='float')

# 根据基站位置和相对位置计算测角测距
def get_observation(stationPos, r):
    longitude = stationPos[0] #经度
    latitude = stationPos[1] #纬度
    r_station = np.zeros(3)
    r_station[0] = constant.R_earth.value * np.cos(latitude) * np.cos(longitude)
    r_station[1] = constant.R_earth.value * np.cos(latitude) * np.sin(longitude)
    r_station[2] = constant.R_earth.value * np.sin(latitude)
    r_local=np.dot(yRotationMatrix(latitude-np.pi/2),np.dot(zRotationMatrix(-longitude),r-r_station))
    for i in range(3):
        if abs(r_local[i])<1e-12:
            r_local[i]=0
    rho = np.linalg.norm(r)
    E = np.arctan(r_local[2] / np.sqrt(r_local[0]**2 + r_local[1]**2))
    A = np.arctan(r_local[1]/r_local[0])
    if r_local[0]<0:
        A = A + np.pi
    if (r_local[0]==0) & (r_local[1]==0):
        A = 0
    obs=np.array([rho,E,A])
    return obs

# stationPos=np.array([0,np.pi/2])
# r=np.array([1,0,0])
# obs=get_observation(stationPos, r)
# print(obs[0])
# print(obs[1]*180/np.pi)
# print(obs[2]*180/np.pi)