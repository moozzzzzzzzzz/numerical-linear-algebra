##经典Gram-Schmidt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）


def classical_gram_schmidt(A):
    m,n=A.shape
    Q=np.zeros((m,n))
    R=np.zeros((n,n))
    
    for j in range(n):
        v=A[:,j].copy()
        for i in range(j):
            R[i,j]=Q[:,i]@A[:,j]
            v=v-R[i,j]*Q[:,i]
        R[j,j]=np.linalg.norm(v)
        Q[:,j]=v/R[j,j]
    return Q,R

A=np.random.randn(10,6)
Q,R=classical_gram_schmidt(A)
print('A-QR的误差：',np.linalg.norm(A-Q@R))
print('Q.T@Q-I的误差：',np.linalg.norm(Q.T@Q-np.eye(6)))
print('R的下三角部分绝对值和：',np.sum(np.abs(np.tril(R,-1))))

##修正Gram-Schmidt算法
def modified_gram_schmidt(A):
    m,n=A.shape
    Q=np.zeros((m,n))
    R=np.zeros((n,n))
    for j in range(n):
        v=A[:,j].copy()
        for i in range(n):
            R[i,j]=Q[:,i]@v
            v=v-R[i,j]*Q[:,i]
        R[j,j]=np.linalg.norm(v)
        Q[:,j]=v/R[j,j]
    return Q,R


##实验：CGS和MGS方法对条件数过大的病态矩阵的优良对比
def construct_ill_conditioned_matrix(m,n,kappa):
    U,_=np.linalg.qr(np.random.randn(m,n))#Q赋给了U，得到一个列正交矩阵
    V,_=np.linalg.qr(np.random.randn(n,n))#得到n*n的正交矩阵V
    S=np.logspace(0,np.log10(kappa),n)#np.logspace(a, b, n)：生成从10ᵃ到10ᵇ、在对数尺度上均匀分布的n个数
    return U @ np.diag(S)@V.T 

kappa_list=[1e1,1e3,1e5,1e7,1e9,1e11,1e13,1e15]
err_cgs=[]
err_mgs=[]

for kappa in kappa_list:
    A=construct_ill_conditioned_matrix(100, 50, kappa)
    Q_cgs,_=classical_gram_schmidt(A)
    Q_mgs,_=modified_gram_schmidt(A)
    err_cgs.append(np.linalg.norm(Q_cgs.T@Q_cgs-np.eye(50)))
    err_mgs.append(np.linalg.norm(Q_mgs.T@Q_mgs-np.eye(50)))
    

plt.figure(figsize=(7,5))
plt.loglog(kappa_list,err_cgs,'o-',label='Classical GS',linewidth=2)
plt.loglog(kappa_list,err_mgs,'s-',label='Modified GS',linewidth=2)
plt.loglog(kappa_list,[k*1e-16 for k in kappa_list],'k--',label='O(κ·ε)参考线',alpha=0.5)
plt.xlabel('条件数κ',fontsize=12)
plt.ylabel('正交性误差$||Q^T Q - I||$',fontsize=12)
plt.title('经典vs修正Gram-Schmidt的数值稳定性',fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gs_stability.png', dpi=150)
plt.show()