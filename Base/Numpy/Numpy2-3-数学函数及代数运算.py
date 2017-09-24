
# coding: utf-8

# # 1. 数学函数

# ## 1.1 三角函数

# 1. numpy.sin(x)：三角正弦。
# - numpy.cos(x)：三角余弦。
# - numpy.tan(x)：三角正切。
# - numpy.arcsin(x)：三角反正弦。
# - numpy.arccos(x)：三角反余弦。
# - numpy.arctan(x)：三角反正切。
# - numpy.hypot(x1,x2)：直角三角形求斜边。
# - numpy.degrees(x)：弧度转换为度。
# - numpy.radians(x)：度转换为弧度。
# - numpy.deg2rad(x)：度转换为弧度。
# - numpy.rad2deg(x)：弧度转换为度。

# In[1]:

import numpy as np

np.rad2deg(np.pi)


# ## 1.2 双曲函数

# 1. numpy.sinh(x)：双曲正弦。
# - numpy.cosh(x)：双曲余弦。
# - numpy.tanh(x)：双曲正切。
# - numpy.arcsinh(x)：反双曲正弦。
# - numpy.arccosh(x)：反双曲余弦。
# - numpy.arctanh(x)：反双曲正切。

# ## 1.3 数值修约

# 1. numpy.around(a)：平均到给定的小数位数。
# - numpy.round_(a)：将数组舍入到给定的小数位数。
# - numpy.rint(x)：修约到最接近的整数。
# - numpy.fix(x, y)：向 0 舍入到最接近的整数。
# - numpy.floor(x)：返回输入的底部(标量 x 的底部是最大的整数 i)。
# - numpy.ceil(x)：返回输入的上限(标量 x 的底部是最小的整数 i).
# - numpy.trunc(x)：返回输入的截断值。

# In[2]:

a = np.array([1.21, 2.53, 3.86])
a


# In[3]:

np.around(a)


# In[8]:

np.round_(a)


# In[10]:

np.rint(a)


# In[11]:

np.fix(a)


# In[12]:

np.floor(a)


# In[13]:

np.ceil(a)


# In[14]:

np.trunc(a)


# ## 1.4 求和、求积、差分

# 1. numpy.prod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积。
# - numpy.sum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和。
# - numpy.nanprod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积, 将NaN视作1。
# - numpy.nansum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和, 将NaN视作0。
# - numpy.cumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积。
# - numpy.cumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和。
# - numpy.nancumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积, 将 NaN 视作 1。
# - numpy.nancumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和, 将 NaN 视作 0。
# - numpy.diff(a, n, axis)：计算沿指定轴的第 n 个离散差分。
# - numpy.ediff1d(ary, to_end, to_begin)：数组的连续元素之间的差异。
# - numpy.gradient(f)：返回 N 维数组的梯度。
# - numpy.cross(a, b, axisa, axisb, axisc, axis)：返回两个(数组）向量的叉积。
# - numpy.trapz(y, x, dx, axis)：使用复合梯形规则沿给定轴积分。

# ## 1.5 指数和对数

# 1. numpy.exp(x)：计算输入数组中所有元素的指数。
# - numpy.expm1(x)：对数组中的所有元素计算 exp(x） - 1.
# - numpy.exp2(x)：对于输入数组中的所有 p, 计算 2\*\*p。
# - numpy.log(x)：计算自然对数。
# - numpy.log10(x)：计算常用对数。
# - numpy.log2(x)：计算二进制对数。
# - numpy.log1p(x)：log(1 + x)。
# - numpy.logaddexp(x1, x2)：log2(2\*\*x1 + 2\*\*x2)。
# - numpy.logaddexp2(x1, x2)：log(exp(x1) + exp(x2))。

# ## 1.6 算术运算

# 1. numpy.add(x1, x2)：对应元素相加。
# - numpy.reciprocal(x)：求倒数 1/x。
# - numpy.negative(x)：求对应负数。
# - numpy.multiply(x1, x2)：求解乘法。
# - numpy.divide(x1, x2)：相除 x1/x2。
# - numpy.power(x1, x2)：类似于 x1^x2。
# - numpy.subtract(x1, x2)：减法。
# - numpy.fmod(x1, x2)：返回除法的元素余项。
# - numpy.mod(x1, x2)：返回余项。
# - numpy.modf(x1)：返回数组的小数和整数部分。
# - numpy.remainder(x1, x2)：返回除法余数。

# In[15]:

a1 = np.random.randint(0, 10, 5)
a2 = np.random.randint(0, 10, 5)

a1


# In[16]:

a2


# In[17]:

np.add(a1, a2)


# In[18]:

np.reciprocal(a1)


# In[19]:

np.negative(a1)


# In[20]:

np.multiply(a1, a2)


# In[21]:

np.divide(a1, a2)


# In[22]:

np.power(a1, a2)


# In[23]:

np.subtract(a1, a2)


# In[24]:

np.fmod(a1, a2)


# In[25]:

np.mod(a1, a2)


# In[27]:

np.modf(a1)


# In[28]:

np.remainder(a1, a2)


# ## 1.7 矩阵和向量积

# 1. numpy.dot(a,b)：求解两个数组的点积。
# - numpy.vdot(a,b)：求解两个向量的点积。
# - numpy.inner(a,b)：求解两个数组的内积。
# - numpy.outer(a,b)：求解两个向量的外积。
# - numpy.matmul(a,b)：求解两个数组的矩阵乘积。
# - numpy.tensordot(a,b)：求解张量点积。
# - numpy.kron(a,b)：计算 Kronecker 乘积。

# ## 1.8 其他

# 1. numpy.angle(z, deg)：返回复参数的角度。
# - numpy.real(val)：返回数组元素的实部。
# - numpy.imag(val)：返回数组元素的虚部。
# - numpy.conj(x)：按元素方式返回共轭复数。
# - numpy.convolve(a, v, mode)：返回线性卷积。
# - numpy.sqrt(x)：平方根。
# - numpy.cbrt(x)：立方根。
# - numpy.square(x)：平方。
# - numpy.absolute(x)：绝对值, 可求解复数。
# - numpy.fabs(x)：绝对值。
# - numpy.sign(x)：符号函数。
# - numpy.maximum(x1, x2)：最大值。
# - numpy.minimum(x1, x2)：最小值。
# - numpy.nan_to_num(x)：用 0 替换 NaN。
# - numpy.interp(x, xp, fp, left, right, period)：线性插值。

# # 2. 代数运算

# 1. numpy.linalg.cholesky(a)：Cholesky 分解。
# - numpy.linalg.qr(a ,mode)：计算矩阵的 QR 因式分解。
# - numpy.linalg.svd(a ,full_matrices,compute_uv)：奇异值分解。
# - numpy.linalg.eig(a)：计算正方形数组的特征值和右特征向量。
# - numpy.linalg.eigh(a, UPLO)：返回 Hermitian 或对称矩阵的特征值和特征向量。
# - numpy.linalg.eigvals(a)：计算矩阵的特征值。
# - numpy.linalg.eigvalsh(a, UPLO)：计算 Hermitian 或真实对称矩阵的特征值。
# - numpy.linalg.norm(x ,ord,axis,keepdims)：计算矩阵或向量范数。
# - numpy.linalg.cond(x ,p)：计算矩阵的条件数。
# - numpy.linalg.det(a)：计算数组的行列式。
# - numpy.linalg.matrix_rank(M ,tol)：使用奇异值分解方法返回秩。
# - numpy.linalg.slogdet(a)：计算数组的行列式的符号和自然对数。
# - numpy.trace(a ,offset,axis1,axis2,dtype,out)：沿数组的对角线返回总和。
# - numpy.linalg.solve(a,b)：求解线性矩阵方程或线性标量方程组。
# - numpy.linalg.tensorsolve(a,b ,axes)：为 x 解出张量方程a x = b
# - numpy.linalg.lstsq(a,b ,rcond)：将最小二乘解返回到线性矩阵方程。
# - numpy.linalg.inv(a)：计算逆矩阵。
# - numpy.linalg.pinv(a ,rcond)：计算矩阵的（Moore-Penrose）伪逆。
# - numpy.linalg.tensorinv(a ,ind)：计算N维数组的逆。
