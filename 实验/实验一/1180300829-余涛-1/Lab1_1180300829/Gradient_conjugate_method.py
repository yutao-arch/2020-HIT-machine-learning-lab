import numpy as np

'''
	将所有的样本sample_x进行转换，生成一个X矩阵 
	在CreateData中生成的均匀的sample_x的样本数据是一维向量，需要预处理成为矩阵X
	其中矩阵X的纬度为sample_num * (polynomial_order + 1)
'''
def CreateMatrixX(sample_x,sample_num,polynomial_order):
    X = np.zeros((sample_num, polynomial_order + 1))  #先建立一个矩阵，行数为sample_num，列数为polynomial_order + 1
    for i in range(sample_num):  #对矩阵每一行进行赋值
        every_row_i = np.ones(polynomial_order + 1) * sample_x[i]  #先将每一行全部赋值为sample_x[i]
        poly_row = np.arange(0, polynomial_order+1)  #得到每一列的阶数
        every_row_i = np.power(every_row_i, poly_row)  #得到这一行所有sample_x[i]值的阶数值
        X[i] = every_row_i
    return X

'''
	误差函数E(w) 
	由已知得到误差函数的表达式为E(w) = 1/2 * (Xw - Y)^{T} . (Xw - Y) 
'''
def ErrorFunction(sample_x, sample_y,sample_num, w):
    X = CreateMatrixX(sample_x)   #将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))  #将sample_y变成一个竖着的一维向量
    temp = X.dot(w) - Y  #X矩阵与w相乘后减去Y
    ErrorFunction = 1/2 * np.dot(temp.T, temp)  #套用误差函数表达式即可
    return ErrorFunction

'''
    情况：加惩罚项，此时惩罚项系数为lamda
    对误差函数用共轭梯度法，循环迭代polynomial_order+1次，得到此时的w  
'''
def Gradient_conjugate_method(sample_x, sample_y, lamda, sample_num, polynomial_order):
    X = CreateMatrixX(sample_x,sample_num,polynomial_order)    #将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))  #将sample_y变成一个竖着的一维向量
    Q = np.dot(X.T, X) + lamda * np.eye(X.shape[1])  #w的系数Q为X^{T} * X + lamda * 一个主对角线全为1，其他元素全为0的标准矩阵
    w = np.zeros((polynomial_order + 1, 1)) #w为一个行数为polynomial_order + 1，列数为1的矩阵
    Gradient = np.dot(X.T, X).dot(w) - np.dot(X.T, Y) + lamda * w #误差函数对w的偏导数公式为X^{T} * X * w - X^{T} * Y + lamda * w，即梯度
    r = -Gradient #r为负梯度
    p = r #从p为负梯度开始
    for i in range(polynomial_order + 1):  #迭代polynomial_order + 1次
        a = (r.T.dot(r)) / (p.T.dot(Q).dot(p))  #a = r^{T} * r / p^{T} * Q * p
        r_prev = r #这个r
        w = w + a * p  #w = w + a * p
        r = r - (a * Q).dot(p)  #r = r + a * Q * p
        beita= (r.T.dot(r)) / (r_prev.T.dot(r_prev))  #beita = r^{T} * r / r_prev^{T} * r_prev
        p = r + beita * p #p = r + beita * p
    poly = np.poly1d(w[::-1].reshape(polynomial_order + 1))  #先将w从后往前，得到多项式系数，按照阶数从高到低排列
    return poly