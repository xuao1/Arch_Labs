import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

x = [128, 256, 512, 1024, 2048]
CPU0 = [4.83, 39.48, 344.29, 4192, 91657]
CPU1 = [0.99, 8.65, 72.96, 922.9, 17343]
CPU2 = [1.28, 9.18, 78.30, 743.2, 7496]
CPU2_1 = [3.37, 27.44, 208.6, 1717, 13412]

plt.plot(x, CPU0, label='基础矩阵乘法')
plt.plot(x, CPU1, label='AVX矩阵乘法')
plt.plot(x, CPU2, label='AVX分块矩阵乘法')
plt.plot(x, CPU2_1, label='AVX分块矩阵乘法（对 B 转置）')

plt.xlabel('n')
plt.ylabel('Time/ms')
plt.title('矩阵乘法执行时间')
plt.legend()

plt.show()

