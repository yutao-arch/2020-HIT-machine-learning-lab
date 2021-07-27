import numpy as np

'''
	将所有的样本sample_x进行转换，生成一个X矩阵 
	在CreateData中生成的均匀的sample_x的样本数据是一维向量，需要预处理成为矩阵X
	其中矩阵X的纬度为sample_num * (polynomial_order + 1)
'''
def CreateMatrixX(sample_x,sample_num, polynomial_order):
    X = np.zeros((sample_num, polynomial_order + 1))  #先建立一个矩阵，行数为sample_num，列数为polynomial_order + 1
    for i in range(sample_num):  #对矩阵每一行进行赋值
        every_row_i = np.ones(polynomial_order + 1) * sample_x[i]  #先将每一行全部赋值为sample_x[i]
        poly_row = np.arange(0, polynomial_order+1)  #得到每一列的阶数
        every_row_i = np.power(every_row_i, poly_row)  #得到这一行所有sample_x[i]值的阶数值
        X[i] = every_row_i
    return X

'''
    情况：加惩罚项，此时惩罚项系数为lamda
    令误差函数导数等于0，求此时的w，此时的w = (X^{T} * X + lamda)^{-1} * X^{T} * Y
'''
def Penalty_item_add_ErrorFunction(sample_x, sample_y, lamda,sample_num,polynomial_order):
    X = CreateMatrixX(sample_x,sample_num, polynomial_order)    #将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))   #将sample_y变成一个竖着的一维向量
    w = np.linalg.inv(np.dot(X.T, X) + lamda * np.eye(X.shape[1])).dot(X.T).dot(Y)  #套用公式w = (X^{T} * X + lamda)^{-1} * X^{T} * Y得到w
    poly = np.poly1d(w[::-1].reshape(polynomial_order + 1))  #先将w从后往前，得到多项式系数，按照阶数从高到低排列
    return poly