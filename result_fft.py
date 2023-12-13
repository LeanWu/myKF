import numpy as np
import scipy
import matplotlib.pyplot as plt

# 读取数据
# data = np.hstack((t,xs,r_observe,xs_ckf1,xs_ckf3))
data=np.loadtxt('.\data\calculte_result_2.txt')
a_test=1e-2

t = data[:,0]
xs = data[:,1:7]
r_observe = data[:,7:10]
xs_ckf = data[:,10:22]
xs_rts = data[:,22:34]

x1 = xs_rts[1000:-1,6]
n = 3  # 滤波器阶数
cut_off = 2*1e-3  # 截止频率
b, a = scipy.signal.butter(n, cut_off, 'low')
x1_filtered = scipy.signal.filtfilt(b, a, x1)

x2 = xs_rts[:,6]
n = 3  # 滤波器阶数
cut_off = 2*1e-3  # 截止频率
b, a = scipy.signal.butter(n, cut_off, 'low')
x2_filtered = scipy.signal.filtfilt(b, a, x2)

a_x=np.zeros(len(t))
for i in range(len(t)):
    v_norm=np.linalg.norm(xs[i,3:6])
    a_x[i]=a_test*xs[i,3]/v_norm

# 绘制结果
plt.figure()
plt.plot(t, x2, label='Before')
plt.plot(t, x2_filtered, label='After2')
plt.plot(t, a_x, label='Track')
plt.legend()
plt.show()

# 绘制结果
plt.figure()
plt.plot(t[1000:-1], x1, label='Before')
plt.plot(t[1000:-1], x1_filtered, label='After')
# plt.plot(t[1000:-1], x2_filtered[1000:-1], label='After2')
plt.plot(t[1000:-1], a_x[1000:-1], label='Track')
plt.xlabel('t(s)')
plt.ylabel('ax(m/s^2)')
plt.legend()
plt.show()

# # 绘制结果
# plt.figure()
# plt.plot(t[1000:-1], x[1000:-1], label='Before')
# plt.plot(t[1000:-1], x_filtered[1000:-1], label='After')
# plt.plot(t[1000:-1], a_x[1000:-1], label='Track')
# plt.legend()
# plt.show()

debug = 1