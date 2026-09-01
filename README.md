# numerical-linear-algebra
本项目包含数值代数中几个核心算法的Python实现，以及随机化SVD的数值实验。

## 已实现内容
NumPy基础笔记：整理了学习NumPy时的常用操作和要点。
QR分解：实现了Gram-Schmidt和Householder的QR 分解，并验证了重构误差。
随机化SVD：基于随机化算法，支持设置目标秩和幂迭代次数。
数值实验：研究了幂迭代次数、过采样参数对随机化SVD精度的影响，并与标准SVD进行了精度和运行时间的对比。

### 第一阶段：NumPy 基础（`numpy_basic/`）

掌握数值计算的基本工具，为后续算法实现打基础。

| 文件 | 内容 |
|---|---|
| `矩阵基础.py` | 矩阵创建（零矩阵、单位矩阵、随机矩阵、Toeplitz 矩阵、低秩矩阵构造） |
| `矩阵运算.py` | 矩阵乘法、转置、求逆、线性方程组求解、各种范数、相对误差计算 |
| `矩阵分解.py` | SVD、QR、特征值分解、Cholesky 分解的实现与验证 |
| `矩阵切片与分块.py` | 矩阵索引、2×2 分块、离对角块低秩性观察（为 HSS 矩阵做铺垫） |
| `计时与画图.py` | 运行时间测量、matplotlib 绘图（对数坐标、双对数坐标） |

### 第二阶段：QR 分解（`QR分解/`）

理解正交化的两种思路，并对比其数值稳定性。

| 文件 | 内容 |
|---|---|
| `gram_schmidt.py` | 经典 Gram-Schmidt（CGS）与修正 Gram-Schmidt（MGS）的实现 |
| `householder_qr.py` | Householder 反射变换的 QR 分解实现（LAPACK 实际使用的方法） |
| `gs_stability.py` | 病态矩阵下 CGS vs MGS vs Householder 的正交性误差对比实验 |
| `CGS vs MGS.png` | 稳定性对比结果图：条件数增大时，经典 GS 正交性急剧恶化 |
| `svd_timing.png` | SVD 运行时间随矩阵规模变化图（验证 O(n³) 复杂度） |

**核心结论**：经典 Gram-Schmidt 在病态矩阵下数值不稳定，修正版有所改善，Householder 方法最稳定——这也是实际库（如 LAPACK / NumPy）选择 Householder 的原因。

### 第三阶段：随机化 SVD（根目录）

学习当前大规模矩阵低秩近似的主流方法，这是我最感兴趣的方向，也是夏建林老师研究中用到的核心技术之一。

| 文件 | 内容 |
|---|---|
| `randomized_svd.py` | 随机化 SVD 核心实现，支持过采样参数 `p` 和幂迭代次数 `q` |
| `精度随采样p的变化.py` | 实验1：过采样数 p 对重建精度的影响 |
| `exp1_oversampling.png` | 实验1结果图 |
| `精度随迭代次数q的变化.py` | 实验2：幂迭代次数 q 对精度的影响（对比奇异值快/慢衰减两种情形） |
| `exp2_poweriteration.png` | 实验2结果图 |
| `SVD.pdf` / `SVD.tex` | 用 LaTeX 撰写的随机化 SVD 学习笔记，包含算法推导、误差分析与实验总结 |
