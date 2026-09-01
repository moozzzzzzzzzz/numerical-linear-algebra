import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）


#基础随机SVD
def randomized_svd(A,k,p=10,q=0):
  
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




def run_experiment2():
    
    np.random.seed(42)
    m, n, rank = 2000, 2000, 20
    U_true = np.linalg.qr(np.random.randn(m, rank))[0]
    V_true = np.linalg.qr(np.random.randn(n, rank))[0]
    
    S_true = np.logspace(0, -6, rank)  
    A_fast = U_true @ np.diag(S_true) @ V_true.T
    
   
    S_slow = np.linspace(1, 0.8, rank)
    A_slow = U_true @ np.diag(S_slow) @ V_true.T
    
    q_list = [0, 1, 2, 3, 5, 10]
    err_fast, err_slow = [], []
    
    for q in q_list:
        U, S, Vt = randomized_svd(A_fast, k=rank, p=10, q=q)
        ef=np.linalg.norm(A_fast - U@np.diag(S)@Vt) / np.linalg.norm(A_fast)
        err_fast.append(ef)
        
        U, S, Vt = randomized_svd(A_slow, k=rank, p=10, q=q)
        es=np.linalg.norm(A_slow - U@np.diag(S)@Vt) / np.linalg.norm(A_slow)
        err_slow.append(es)
    
    print(f"{'q':<4} {'快速衰减相对误差':<18} {'慢速衰减相对误差':<18}")
    print("-"*50)
    for q,ef,es in zip(q_list, err_fast, err_slow):
        print(f"{q:<4} {ef:<18.3e} {es:<18.3e}")
    
    plt.figure(figsize=(7, 5))
    plt.semilogy(q_list, err_fast, 'o-', label='奇异值快速衰减', linewidth=2)
    plt.semilogy(q_list, err_slow, 's-', label='奇异值缓慢衰减', linewidth=2)
    plt.xlabel('幂迭代次数 q', fontsize=12)
    plt.ylabel('相对误差', fontsize=12)
    plt.title('幂迭代对精度的影响', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp2_poweriteration.png', dpi=150)
    plt.show()
    
    

run_experiment2()