import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False             # 负号正常显示（否则-号也变方块）



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

