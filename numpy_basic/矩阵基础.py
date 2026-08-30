# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:40:52 2026

@author: 10051
"""

import numpy as np
from scipy.linalg import toeplitz

##创建矩阵
A=np.array([[1,2,3],
           [4,5,6],
           [7,8,9]])
print("A=\n",A)
print('A的形状：',A.shape)

##特殊矩阵
Z=np.zeros((5,5))     #全零矩阵
O=np.ones((4,3))      #全一矩阵
I=np.eye(6)           #单位矩阵
D=np.diag([1,2,3,4])  #对角矩阵

##随机矩阵
G=np.random.randn(100,50) #每个元素服从标准正态分布的，100行50列的矩阵
U=np.random.rand(50,50)   #[0,1]均匀分布

##Toeplitz矩阵
T = toeplitz([1, 2, 3, 4, 5, 6, 7, 8])
print("Toeplitz矩阵左上角3x3:\n", T[:3, :3])

##构造低秩矩阵（已知奇异值，逆向构造一个奇异值已知的矩阵）
def low_rank_matrix(m,n,r): #m行数，n列数，r目标秩
    Q1=np.linalg.qr(np.random.randn(m,r))[0] #对随机矩阵做QR分解，并只取Q
    Q2=np.linalg.qr(np.random.randn(n,r))[0]
    s=np.linspace(10,1,r) #在10，1区间上均匀取r个点
    return Q1 @ np.diag(s) @ Q2.T
L=low_rank_matrix(200,200,10)
print("低秩矩阵的秩：",np.linalg.matrix_rank(L))
##构造出的矩阵大多数元素非零（稠密），方向随机（没有对称性等特殊结构），近似低秩（小奇异值接近零）

##测试：生成一个 1000×1000、秩为 5 的低秩矩阵，打印它的形状和秩
AA=low_rank_matrix(1000, 1000, 5)
print('AA的形状：',AA.shape)
print("AA的秩：",np.linalg.matrix_rank(AA))

##测试：生成一个第一行为[1, 0.5, 0.25, 0.125, ...]（共 10 个元素）的 Toeplitz 矩阵，打印左上角 4×4。
r=[1/2**k for k in range(10)]
TT=toeplitz(r,r)
print('TT左上角4*4的矩阵：\n',TT[:4,:4])












