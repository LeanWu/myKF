import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data=np.loadtxt('.\data\calculte_result_6.txt')
a_test=1e-2

t = data[:,0]
xs = data[:,1:7]
r_observe = data[:,7:10]
xs_ckf1 = data[:,10:22]
xs_ckf2 = data[:,22:34]
xs_ckf3 = data[:,34:46]


color1='orange'
color2='lightgreen'
color3='violet'
color_track='cornflowerblue'
label1='stationNum=1'
label2='stationNum=2'
label3='stationNum=3'
r0=(r_observe[:,1]-xs[:,1])/1e3
r1=(xs_ckf1[:,0]-xs[:,0])/1e3
v1=(xs_ckf1[:,3]-xs[:,3])/1e3
r2=(xs_ckf2[:,0]-xs[:,0])/1e3
v2=(xs_ckf2[:,3]-xs[:,3])/1e3
r3=(xs_ckf3[:,0]-xs[:,0])/1e3
v3=(xs_ckf3[:,3]-xs[:,3])/1e3

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,r0,c='black',label='Measurement')
ax.plot(t,r1,label=label1,c=color1)
ax.plot(t,r2,label=label2,c=color2)
ax.plot(t,r3,label=label3,c=color3)
ax.set_xlabel('t(s)')
ax.set_ylabel('error of x(km)')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,v1,label=label1,c=color1)
ax.plot(t,v2,label=label2,c=color2)
ax.plot(t,v3,label=label3,c=color3)
ax.set_xlabel('t(s)')
ax.set_ylabel('error of vx(km/s)')
plt.legend()
plt.show()


a_x=np.zeros(len(t))
for i in range(len(t)):
    v_norm=np.linalg.norm(xs[i,3:6])
    a_x[i]=a_test*xs[i,3]/v_norm

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,xs_ckf1[:,6],label=label1,c=color1)
ax.plot(t,xs_ckf2[:,6],label=label2,c=color2)
ax.plot(t,xs_ckf3[:,6],label=label3,c=color3)
ax.plot(t,a_x,label='Track',c=color_track)
ax.set_xlabel('t(s)')
ax.set_ylabel('ax(m/s^2)')
# ax.set_ylim(-20,20)
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t[1000:-1],xs_ckf1[1000:-1,6],label=label1,c=color1)
ax.plot(t[1000:-1],xs_ckf2[1000:-1,6],label=label2,c=color2)
ax.plot(t[1000:-1],xs_ckf3[1000:-1,6],label=label3,c=color3)
ax.plot(t[1000:-1],a_x[1000:-1],label='Track',c=color_track)
ax.set_xlabel('t(s)')
ax.set_ylabel('ax(m/s^2)')
plt.legend()
plt.show()