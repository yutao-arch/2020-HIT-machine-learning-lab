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
def ErrorFunction(sample_x, sample_y,sample_num,polynomial_order, w):
    X = CreateMatrixX(sample_x,sample_num,polynomial_order)   #将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))  #将sample_y变成一个竖着的一维向量
    temp = X.dot(w) - Y  #X矩阵与w相乘后减去Y
    ErrorFunction = 1/2 * np.dot(temp.T, temp)  #套用误差函数表达式即可
    return ErrorFunction

'''
    情况：加惩罚项，此时惩罚项系数为lamda
    对误差函数使用梯度下降法，当误差函数收敛到期望的最小值时，得到此时的w
    参数中，最多循环轮次cycle_times，下降步长descending_step_size，迭代误差iteration_error
'''
def Gradient_descent_method(sample_x, sample_y, lamda, cycle_times, descending_step_size, iteration_error,sample_num,polynomial_order):
    w = np.ones((polynomial_order + 1, 1))   #生成一个行数为polynomial_order + 1，列数为1的矩阵，元素值全为1
    X = CreateMatrixX(sample_x,sample_num, polynomial_order)   #将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))  #将sample_y变成一个竖着的一维向量
    for i in range(cycle_times):  #迭代cycle_times次
        old_ErrorFunction = abs(ErrorFunction(sample_x, sample_y,sample_num,polynomial_order, w))  #求得旧的误差函数值的绝对值
        ErrorFunction_partial_deriv = X.T.dot(X).dot(w) - X.T.dot(Y) + lamda * w  #误差函数对w的偏导数公式为X^{T} * X * w - X^{T} * Y + lamda * w，即梯度
        w = w - descending_step_size * ErrorFunction_partial_deriv  #梯度下降法的迭代公式为w <--- w - descending_step_size * 误差函数对w的偏导数
        new_ErrorFunction = abs(ErrorFunction(sample_x, sample_y,sample_num,polynomial_order, w))  #求得新的误差函数值的绝对值
        if(new_ErrorFunction > old_ErrorFunction): #当新的误差函数值大于旧的误差函数值时，将步长变为原来的一半
            descending_step_size *=0.5
        if(abs(new_ErrorFunction - old_ErrorFunction) < iteration_error): #如果新的误差函数值与旧的误差函数值的差小于迭代误差则终止迭代
            break
    poly = np.poly1d(w[::-1].reshape(polynomial_order + 1)) #先将w从后往前，得到多项式系数，按照阶数从高到低排列
    return poly