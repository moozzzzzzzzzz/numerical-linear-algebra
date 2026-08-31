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



##Householder QR
def householder_qr(A):
    m,n=A.shape
    R=A.copy()
    Q=np.eye(m)
    for k in range(min(m,n)):
        x=R[k:,k].copy()
        alpha=-np.sign(x[0])*np.linalg.norm(x)
        v=x.copy()
        v[0]=v[0]-alpha
        v=v/np.linalg.norm(v)
        R[k:,k:]=R[k:,k:]-2*np.outer(v,v@R[k:,k:])
        Q[:,k:]=Q[:,k:]-2*Q[:,k:]@np.outer(v,v)
    return Q,R

A = np.random.randn(100, 60)
Q, R = householder_qr(A)

print("A - QR 误差:", np.linalg.norm(A - Q @ R))
print("Q.T @ Q - I 误差:", np.linalg.norm(Q.T @ Q - np.eye(100)))
print("R下三角绝对值和:", np.sum(np.abs(np.tril(R, -1))))

# 和 numpy 的结果对比
Q_np, R_np = np.linalg.qr(A)
print("和numpy的Q差:", np.linalg.norm(np.abs(Q[:,:60]) - np.abs(Q_np)))  # 符号可能不同，取绝对值


kappa = 1e14
A = construct_ill_conditioned_matrix(200, 100, kappa)

Q1, R1 = classical_gram_schmidt(A)
Q2, R2 = modified_gram_schmidt(A)
Q3, R3 = householder_qr(A)
n = A.shape[1]
Q3_thin = Q3[:, :n]
R3_thin = R3[:n, :]

print(f"条件数 κ = {kappa:.0e}")
print(f"{'方法':<20} {'正交性误差':<15} {'重建误差':<15}")
print("-" * 50)
print(f"{'Classical GS':<20} {np.linalg.norm(Q1.T@Q1 - np.eye(100)):<15.2e} {np.linalg.norm(A - Q1@R1):<15.2e}")
print(f"{'Modified GS':<20} {np.linalg.norm(Q2.T@Q2 - np.eye(100)):<15.2e} {np.linalg.norm(A - Q2@R2):<15.2e}")
print(f"{'Householder':<20} {np.linalg.norm(Q3_thin.T@Q3_thin - np.eye(n)):<15.2e} {np.linalg.norm(A - Q3_thin@R3_thin):<15.2e}")















