import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）


#基础随机SVD
def randomized_svd(A,k,p=10,q=0):
   # A:m×n矩阵
   # k:目标秩
   # p:过采样数，通常取k的1/3到1倍
   # q:幂迭代次数,提高精度用
   m,n=A.shape#m，n很大，我们只想要前k个主奇异值。牺牲一点点精度
   l=k+p
   omega=np.random.randn(n,l)
   Y=A@omega
   for _ in range(q):  #
       Y=A@(A.T@Y)     #AA.T的奇异值是A的奇异值平方。大奇异值随着幂次增大而不断增大，小的则被削弱了
       Y, _ = np.linalg.qr(Y)#我们发现如果直接用Y迭代数值量级会爆炸，数值溢出，所以我们最好每一步幂迭代后做归一化处理
   Q,_=np.linalg.qr(Y)
   B=Q.T@A#把A投影到低维子空间m*n到l*n
   U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
   U=Q@U_tilde
   return U[:,:k],S[:k],Vt[:k,:]

#过采样数p对精度的影响
def run_experiment1():
   
    np.random.seed(42)
    m,n,rank=2000,2000,20
    U_true=np.linalg.qr(np.random.randn(m,rank))[0]
    V_true=np.linalg.qr(np.random.randn(n,rank))[0]
    S_true=np.logspace(1,0,rank)
    A=U_true @ np.diag(S_true) @ V_true.T 
    
    p_list=[0,1,2,5,10,20,40]
    errors = []
    
    for p in p_list:
        # 多次实验取平均，减少随机性影响
        errs = []
        for trial in range(20):
            U,S,Vt=randomized_svd(A,k=rank,p=p,q=0)
            A_approx=U @ np.diag(S) @ Vt
            err=np.linalg.norm(A-A_approx)/np.linalg.norm(A)
            errs.append(err)
        errors.append(np.mean(errs))
        print(f"p={p:>3}:平均相对误差 = {np.mean(errs):.2e}")
    

    plt.figure(figsize=(7, 5))
    plt.semilogy(p_list, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('过采样数 p', fontsize=12)
    plt.ylabel('相对误差', fontsize=12)
    plt.title('过采样对随机化SVD精度的影响（k=20）', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_oversampling.png', dpi=150)
    plt.show()

    print("\n结论：当 p=0 时误差较大，随着p增大误差迅速下降，"
          "p超过5-10后误差不再明显改善。")
    print("这验证了过采样对随机算法稳定性的重要性。")

run_experiment1()

