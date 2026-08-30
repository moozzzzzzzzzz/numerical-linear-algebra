import time
import numpy as np

A=np.random.randn(100,100)
B=np.random.randn(100,100)
x=np.random.randn(100)
b=np.random.randn(100)

##乘法
C=A@B              ##矩阵乘法
y=A@x              
C_elementwise=A*B  ##逐元素相乘

##转置
At=A.T             ##转置
Ac=A.conj().T      ##共轭转置

##求逆&&解线性系统
Ainv=np.linalg.inv(A)       ##求逆
x_sol=np.linalg.solve(A,b)  ##解Ax=b

print("残差||Ax-b||：",np.linalg.norm(A@x_sol-b))

##范数
nf=np.linalg.norm(A)           ##默认Frobenius范数
n2=np.linalg.norm(A,ord=2)     ##2-范数（最大奇异值
n1=np.linalg.norm(A,ord=1)     ##1-范数（最大列和
ninf=np.linalg.norm(A,ord=np.inf) #无穷范数（最大行和

##相对误差
rel_err=np.linalg.norm(A-B)/np.linalg.norm(A)
print('A与B的相对Frobenius误差：',rel_err)






























