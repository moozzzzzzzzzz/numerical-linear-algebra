import numpy as np

A=np.random.randn(200,100)

##SVD
U,S,Vt=np.linalg.svd(A,full_matrices=False)
print('U的形状：',U.shape)
print('Vt的形状',Vt.shape)
print('S的形状：',S.shape)

A_reconstructed=U @ np.diag(S) @Vt
print('SVD重建误差：',np.linalg.norm(A-A_reconstructed)/np.linalg.norm(A))

r=10 ##取前r个值做低秩近似
A_r=U[:,:r] @ np.diag(S[:r]) @Vt[:r,:]
print(f"前{r}个奇异值的近似误差:",np.linalg.norm(A-A_r)/np.linalg.norm(A))

##QR分解
Q,R=np.linalg.qr(A)
print('Q形状：',Q.shape)
print('R形状：',R.shape)
print('Q.T @ Q-I的误差：',np.linalg.norm(Q.T@Q-np.eye(100))) #Q列正交
#print('R.T @ R-I的误差：',np.linalg.norm(R.T@R-np.eye(100)))

##特征值分解
A_sym=A @A.T
eigvals,eigvecs=np.linalg.eigh(A_sym)#对称矩阵eigh，返回（特征值数组，特征向量矩阵）
print("特征值（升序）：",eigvals[:5])#前五个最小的
print("特征值（降序）:",eigvals[-5:])#前五个最大的

#S_from_eig=np.sqrt(eigvals[::-1])
#计算机做浮点运算的时候可能会把0算成微小的负数，出现了微小噪声，我们应该修改将开放前所有的负数变为零
S_from_eig = np.sqrt(np.maximum(eigvals[::-1], 0))
print("特征值算出的奇异值：",S_from_eig[:5])
print('SVD的奇异值：',S[:5])


##Cholesky分解
A_pd=A@A.T+10*np.eye(200) #保持正定
L=np.linalg.cholesky(A_pd)
print('Cholesky重建误差：',np.linalg.norm(A_pd-L@L.T)/np.linalg.norm(A_pd))




















