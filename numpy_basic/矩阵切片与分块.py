import numpy as np

A=np.random.randn(100,100)

##基本切片
print(A[0:10,0:10])  #左上角10*10
print(A[:,0:5])      #前5列
print(A[:5,:])       #前5行
print(A[::2,::2])    #隔行隔列


##对角切块
n=A.shape[0]
mid=n//2
A11=A[:mid,:mid]
A12=A[:mid,mid:]
A21=A[mid:,:mid]
A22=A[mid:,mid:]
print(A11.shape,A22.shape,A12.shape,A21.shape)


##离对角块的数值秩
U12,S12,_=np.linalg.svd(A12)
print('A12的前10个奇异值：',S12[:10])
#奇异值衰减很快说明可以低秩近似

##子块赋值
B=np.zeros((100,100))
B[:50,:50]=np.random.randn(50,50)
B[np.diag_indices(100)]=1.0  #对角线设为1
























