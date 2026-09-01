import numpy as np
import time


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


#测试该随机SVD
np.random.seed(42)
m,n,rank=1000,800,10
U_true=np.linalg.qr(np.random.randn(m,rank))[0]
V_true=np.linalg.qr(np.random.randn(n,rank))[0]
S_true=np.logspace(1,0,rank)#奇异值从10到1
A_clean=U_true @ np.diag(S_true)@V_true.T 
A=A_clean+1e-8*np.random.randn(m,n)

U,S,Vt=randomized_svd(A,k=10,p=10,q=2)

A_approx=U@np.diag(S)@Vt
rel_err=np.linalg.norm(A-A_approx)/np.linalg.norm(A)
print(f"重建相对误差：{rel_err:.2e}")



U_exact, S_exact, Vt_exact = np.linalg.svd(A)
print("精确SVD前10个奇异值:", np.round(S_exact[:10], 4))
print("随机SVD前10个奇异值:", np.round(S, 4))


U_trunc, S_trunc, Vt_trunc = np.linalg.svd(A, full_matrices=False)
A_trunc = U_trunc[:, :10] @ np.diag(S_trunc[:10]) @ Vt_trunc[:10, :]
print(f"精确截断SVD误差: {np.linalg.norm(A - A_trunc)/np.linalg.norm(A):.2e}")
print(f"随机SVD误差:     {rel_err:.2e}")





























