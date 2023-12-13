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

# 根据基站位置和绝对位置计算测角测距
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
    rho = np.linalg.norm(r_local)
    E = np.arctan(r_local[2] / np.sqrt(r_local[0]**2 + r_local[1]**2))
    A = np.arctan(r_local[1]/r_local[0])
    if r_local[0]<0:
        A = A + np.pi
    if (r_local[0]==0) & (r_local[1]==0):
        A = 0
    A = A % (2*np.pi)
    obs=np.array([rho,E,A])
    return obs

def get_observation_R(stationPos, r, R_earth):
    longitude = stationPos[0] #经度
    latitude = stationPos[1] #纬度
    r_station = np.zeros(3)
    r_station[0] = R_earth * np.cos(latitude) * np.cos(longitude)
    r_station[1] = R_earth * np.cos(latitude) * np.sin(longitude)
    r_station[2] = R_earth * np.sin(latitude)
    r_local=np.dot(yRotationMatrix(latitude-np.pi/2),np.dot(zRotationMatrix(-longitude),r-r_station))
    for i in range(3):
        if abs(r_local[i])<1e-12:
            r_local[i]=0
    rho = np.linalg.norm(r_local)
    E = np.arctan(r_local[2] / np.sqrt(r_local[0]**2 + r_local[1]**2))
    A = np.arctan(r_local[1]/r_local[0])
    if r_local[0]<0:
        A = A + np.pi
    if (r_local[0]==0) & (r_local[1]==0):
        A = 0
    A = A % (2*np.pi)
    obs=np.array([rho,E,A])
    return obs

# stationPos=np.array([0,np.pi/2])
# r=np.array([1,0,0])
# obs=get_observation(stationPos, r)
# print(obs[0])
# print(obs[1]*180/np.pi)
# print(obs[2]*180/np.pi)

# 根据基站位置和测角测距计算绝对位置
def get_r(stationPos, obs):
    longitude = stationPos[0] #经度
    latitude = stationPos[1] #纬度
    r_station = np.zeros(3)
    r_station[0] = constant.R_earth.value * np.cos(latitude) * np.cos(longitude)
    r_station[1] = constant.R_earth.value * np.cos(latitude) * np.sin(longitude)
    r_station[2] = constant.R_earth.value * np.sin(latitude)
    r_relative = np.zeros(3)
    r_relative[0] = obs[0] * np.cos(obs[1]) * np.cos(obs[2])
    r_relative[1] = obs[0] * np.cos(obs[1]) * np.sin(obs[2])
    r_relative[2] = obs[0] * np.sin(obs[1])
    r_abs=np.dot(zRotationMatrix(longitude),np.dot(yRotationMatrix(-latitude+np.pi/2),r_relative))
    r = r_station + r_abs
    return r

def get_r_R(stationPos, obs, R_earth):
    longitude = stationPos[0] #经度
    latitude = stationPos[1] #纬度
    r_station = np.zeros(3)
    r_station[0] = R_earth * np.cos(latitude) * np.cos(longitude)
    r_station[1] = R_earth * np.cos(latitude) * np.sin(longitude)
    r_station[2] = R_earth * np.sin(latitude)
    r_relative = np.zeros(3)
    r_relative[0] = obs[0] * np.cos(obs[1]) * np.cos(obs[2])
    r_relative[1] = obs[0] * np.cos(obs[1]) * np.sin(obs[2])
    r_relative[2] = obs[0] * np.sin(obs[1])
    r_abs=np.dot(zRotationMatrix(longitude),np.dot(yRotationMatrix(-latitude+np.pi/2),r_relative))
    r = r_station + r_abs
    return r

# stationPos=np.array([0,np.pi/2])
# r=np.array([1,0,0])
# obs=get_observation(stationPos, r)
# print(obs[0])
# print(obs[1]*180/np.pi)
# print(obs[2]*180/np.pi)

# r1 = get_r(stationPos, obs)
# print(r1[0])
# print(r1[1])
# print(r1[2])

# stationPos = np.array([34*np.pi/180,108*np.pi/180])
# r=np.array([7378136.6,0,0])
# obs=get_observation(stationPos, r)
# zs=np.array([1.091921271677847207e7,-8.416030589276650709e-1,-5.994339756866378099e-1])
# print(obs[0]-zs[0])
# print((obs[1]-zs[1])*180/np.pi)
# print((obs[2]-zs[2])*180/np.pi)

# r1 = get_r(stationPos, obs)
# r2 = get_r(stationPos, zs)
# print(r1)
# print(r2)
# debug=1