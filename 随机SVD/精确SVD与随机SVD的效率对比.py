import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）

def randomized_svd(A,k,p=10,q=0):
  
   m,n=A.shape
   l=k+p
   omega=np.random.randn(n,l)
   Y=A@omega
   for _ in range(q):  
       Y=A@(A.T@Y)     
       Y, _ = np.linalg.qr(Y)
   Q,_=np.linalg.qr(Y)
   B=Q.T@A
   U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
   U=Q@U_tilde
   return U[:,:k],S[:k],Vt[:k,:]


def run_expermient3():
    np.random.seed(42)
    n_list=[500,1000,2000,5000,10000]
    t_exact,t_random=[],[]
    
    for n in n_list:
        A=np.random.randn(n,n)
        k=50
        
        t0=time.perf_counter()
        np.linalg.svd(A,full_matrices=False)
        t_exact.append(time.perf_counter()-t0)
        
        t1=time.perf_counter()
        randomized_svd(A,k=k,p=10,q=1)
        t_random.append(time.perf_counter()-t1)
        
        print(f"n={n:>6}: 精确SVD {t_exact[-1]:.3f}s, 随机SVD {t_random[-1]:.4f}s")
 
    plt.figure(figsize=(7, 5))
    plt.loglog(n_list, t_exact, 'o-', label='精确SVD', linewidth=2)
    plt.loglog(n_list, t_random, 's-', label='随机SVD (k=50)', linewidth=2)
    plt.xlabel('矩阵规模 n', fontsize=12)
    plt.ylabel('运行时间 (秒)', fontsize=12)
    plt.title('随机SVD vs 精确SVD 运行时间', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp3_timing.png', dpi=150)
    plt.show()

run_expermient3()

























