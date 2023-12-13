import numpy as np
import matplotlib.pyplot as plt

# 读取数据
# data = np.hstack((t,xs,r_observe,xs_ckf1,xs_ckf3))
data=np.loadtxt('.\data\calculte_result_0.txt')
a_test=1e-2

t = data[:,0]
xs = data[:,1:7]
r_observe = data[:,7:10]
xs_ckf = data[:,10:22]
xs_low = data[:,22:34]


color1='orange'
color2='lightgreen'
color_track='cornflowerblue'

label1='Jerk-SRCKF'
label2='Low-pass'
r0=(r_observe[:,1]-xs[:,1])/1e3
r3=(xs_ckf[:,0]-xs[:,0])/1e3
v3=(xs_ckf[:,3]-xs[:,3])/1e3
# r1=(xs_rts[:,0]-xs[:,0])/1e3
# v1=(xs_rts[:,3]-xs[:,3])/1e3


# color2='hotpink'
# label2='FLS'
# xs_fls = data[:,34:46]
# r2=(xs_fls[:,0]-xs[:,0])/1e3
# v2=(xs_fls[:,3]-xs[:,3])/1e3

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,r0,c='black',label='Measurement')
ax.plot(t,r3,label=label1,c=color1)
# ax.plot(t,r1,label=label1,c=color1)
# ax.plot(t,r2,label=label2,c=color2)
ax.set_xlabel('t(s)')
ax.set_ylabel('error of x(km)')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,v3,label=label1,c=color1)
# ax.plot(t,v1,label=label1,c=color1)
# ax.plot(t,v2,label=label2,c=color2)
ax.set_xlabel('t(s)')
ax.set_ylabel('error of vx(km/s)')
plt.legend()
plt.show()


a_x=np.zeros(len(t))
for i in range(len(t)):
    v_norm=np.linalg.norm(xs[i,3:6])
    a_x[i]=a_test*xs[i,3]/v_norm
    # if i<4000:
    #     a_x[i]=0

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,xs_ckf[:,6],label=label1,c=color1)
ax.plot(t,xs_low[:,6],label=label2,c=color2)
# ax.plot(t,xs_fls[:,6],label=label2,c=color2)
ax.plot(t,a_x,label='Track',c=color_track)
ax.set_xlabel('t(s)')
ax.set_ylabel('ax(m/s^2)')
# ax.set_ylim(-20,20)
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t[3000:-1],xs_ckf[3000:-1,6],label=label1,c=color1)
ax.plot(t[3000:-1],xs_low[3000:-1,6],label=label2,c=color2)
# ax.plot(t[1000:-1],xs_fls[1000:-1,6],label=label2,c=color2)
ax.plot(t[3000:-1],a_x[3000:-1],label='Track',c=color_track)
ax.set_xlabel('t(s)')
ax.set_ylabel('ax(m/s^2)')
plt.legend()
plt.show()