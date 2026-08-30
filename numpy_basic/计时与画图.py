import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）

##计时
A=np.random.randn(500,500)
b=np.random.randn(500)


_=np.linalg.solve(A,b)
_=np.linalg.inv(A)@b

N=20

t0=time.perf_counter()
for _ in range(N):
    x1=np.linalg.inv(A)@b
t1=time.perf_counter()
t_inv=(t1-t0)/N

t0=time.perf_counter()
for _ in range(N):
    x2=np.linalg.solve(A,b)
t1=time.perf_counter()
t_solve=(t1-t0)/N

rel_err = np.linalg.norm(x1 - x2) / np.linalg.norm(x2)
print(f'相对误差: {rel_err:.2e}')
print(f'inv @ b 平均耗时: {t_inv*1e3:.2f} ms')
print(f'solve    平均耗时: {t_solve*1e3:.2f} ms')

##对比不同规模的SVD时间
n_list=[100,200,500,1000,2000]
times=[]
for n in n_list:
    A=np.random.randn(n,n)
    t0=time.perf_counter()
    np.linalg.svd(A)
    times.append(time.perf_counter()-t0)
    
plt.figure(figsize=(6,4))  #创建画布尺寸
plt.loglog(n_list,times,'o-',linewidth=2,label='SVD')
plt.loglog(n_list,[times[0]*(n/n_list[0])**3 for n in n_list],'k--',label='O(n^3)参考线')
#`times[0]` 是**最小规模 n_list [0] 时的实际耗时
#为什么用 loglog 才能看出 "平行"：对数 - 对数坐标下，幂函数 y = c・n³ 变成一条斜率为 3 的直线，
#所以 "O (n³) 参考线" 在图上是一条直线，而任何幂律数据 y = c・nᵏ 也是一条直线（斜率 k）。直线斜率的对比远比曲线直观。
plt.xlabel('矩阵规模n',fontsize=12)
plt.ylabel('时间(秒)',fontsize=12)
plt.title('SVD运行时间随规模变化',fontsize=14)
plt.legend(fontsize=11)
#- 自动读取前面所有曲线的 `label`（'SVD' 和 'O (n^3) 参考线 '），在图上生成图例框。
#*没有 `label` 的线不会出现在图例里— 这就是第 2、3 行必须写 `label=` 的原因。
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('svd_timing.png',dpi=150)
plt.show()





















