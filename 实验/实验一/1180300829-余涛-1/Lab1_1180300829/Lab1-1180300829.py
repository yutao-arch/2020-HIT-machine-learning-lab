import numpy as np
import matplotlib.pyplot as matplt

from Lab1_1180300829.Gradient_conjugate_method import Gradient_conjugate_method
from Lab1_1180300829.Gradient_descent_method import Gradient_descent_method
from Lab1_1180300829.Penalty_item_ErrorFunction import Penalty_item_ErrorFunction
from Lab1_1180300829.Penalty_item_add_ErrorFunction import Penalty_item_add_ErrorFunction

global sample_num  # 样本点的数量
global polynomial_order  # 多项式的阶数
sample_num = 50
polynomial_order = 9

'''
    生成sample_num个数据，并加入噪声
    sample_x为[0,0.9]上均匀的sample_num个点
    sample_y为sin(2πx)的值，再加上均值为mu，方差为sigma的高斯噪声
'''
def CreateData(mu, sigma):
    sample_x = np.arange(0, 1, 1 / sample_num)  # 创建[0,0.9]上均匀的sample_num个点
    gauss_noise = np.random.normal(mu, sigma, sample_num)  # 创建均值为mu，标准差为sigma的高斯噪声
    sample_y = np.sin(sample_x * 2 * np.pi) + gauss_noise  # 对于sample_x得到每个sin(2πx)的值，再加上均值为mu，标准差为sigma的高斯噪声
    return sample_x, sample_y


'''
	将所有的样本sample_x进行转换，生成一个X矩阵 
	在CreateData中生成的均匀的sample_x的样本数据是一维向量，需要预处理成为矩阵X
	其中矩阵X的纬度为sample_num * (polynomial_order + 1)
'''
def CreateMatrixX(sample_x):
    X = np.zeros((sample_num, polynomial_order + 1))  # 先建立一个矩阵，行数为sample_num，列数为polynomial_order + 1
    for i in range(sample_num):  # 对矩阵每一行进行赋值
        every_row_i = np.ones(polynomial_order + 1) * sample_x[i]  # 先将每一行全部赋值为sample_x[i]
        poly_row = np.arange(0, polynomial_order + 1)  # 得到每一列的阶数
        every_row_i = np.power(every_row_i, poly_row)  # 得到这一行所有sample_x[i]值的阶数值
        X[i] = every_row_i
    return X


'''
	误差函数E(w) 
	由已知得到误差函数的表达式为E(w) = 1/2 * (Xw - Y)^{T} . (Xw - Y) 
'''
def ErrorFunction(sample_x, sample_y, w):
    X = CreateMatrixX(sample_x)  # 将sample_x进行转换，生成一个X矩阵
    Y = sample_y.reshape((sample_num, 1))  # 将sample_y变成一个竖着的一维向量
    temp = X.dot(w) - Y  # X矩阵与w相乘后减去Y
    ErrorFunction = 1 / 2 * np.dot(temp.T, temp)  # 套用误差函数表达式即可
    return ErrorFunction


'''
	直接拟合数据，通过调用np.polyfit()拟合数据 
'''
def FittingData(sample_x, sample_y):
    w = np.polyfit(sample_x, sample_y, polynomial_order)  # 用polynomial_order次多项式拟合，
    poly = np.poly1d(w)  # 得到多项式系数，按照阶数从高到低排列
    return poly


'''
    得到一组测试数据与真实数据的均方误差(RMS)
'''
def Get_RMS(train_num, poly_fit):
    real_x = np.linspace(0, 1, train_num)  # 真实的[0,1]上train_num个均匀点
    real_y = np.sin(real_x * 2 * np.pi)  # 得到对应真实的sin2pix值
    fit_y = poly_fit(real_x)  # 得到拟合值
    the_loss = real_y - fit_y  # 得到拟合值和真实值的差
    the_RMS = np.sqrt((np.dot(the_loss, the_loss.T)) / train_num)  # 求解均方根，作为误差
    return the_RMS


'''
    得到最合适的惩罚项系数lamda
'''
def Get_best_lamda():
    train_num = 100;  # 设置测试数据容量为100
    sample_x, sample_y = CreateData(0, 0.5)  # 创建数据集
    the_degree = np.zeros(51)  # 惩罚项系数阶数集
    the_RMS = np.zeros(51)  # 惩罚项每个阶数对应的均方根集
    min = the_degree[0]
    min_num = 10000
    for i in range(0, 51):  # 对每个阶数进行均方根的求解
        the_degree[i] = -i  # 为阶数集赋值
        the_best_lamda = np.exp(the_degree[i])  # 得到每一个阶数对应的惩罚项系数
        poly = Penalty_item_add_ErrorFunction(sample_x, sample_y, the_best_lamda, sample_num, polynomial_order)  # 求解多项式
        the_RMS[i] = Get_RMS(train_num, poly)  # 对该多项式求解均方根
        if (min_num > the_RMS[i]):  # 得到最小的均方根对应的惩罚项系数阶数
            min = the_degree[i]
            min_num = the_RMS[i]
    plot = matplt.plot(the_degree, the_RMS, 'm', label='lamda line')
    matplt.xlabel('the poly of lamda')
    matplt.ylabel('RMS')
    matplt.legend(loc=1)
    matplt.title("the best poly of lamda is" + str(min))
    matplt.show()


'''
	进行图像的绘制
    sample_x：一维观测数据x
    sample_y：一维观测数据y
	poly_fit：拟合得到的多项式
    title：图像标题
'''
def Draw_Images(sample_x, sample_y, poly_fit, title):
    real_x = np.linspace(0, 0.9, 100)
    real_y = np.sin(real_x * 2 * np.pi)
    fit_y = poly_fit(real_x)
    plot1 = matplt.plot(sample_x, sample_y, 'p', label='the_data_by_train')  # 测试数据
    plot2 = matplt.plot(real_x, fit_y, 'm', label='fit_curve')  # 拟合曲线
    plot3 = matplt.plot(real_x, real_y, 'k', label='the_real_data')  # 真实曲线
    matplt.xlabel('x')
    matplt.ylabel('y')
    matplt.legend(loc=1)
    matplt.title(title)
    matplt.show()


Get_best_lamda()  # 得到最佳的惩罚项系数

sample_x, sample_y = CreateData(0, 0.1)  # 生成sample_num数据，并加入噪声

# 情况：用np.polyfit直接拟合
case1 = FittingData(sample_x, sample_y)
Draw_Images(sample_x, sample_y, case1,
            'np.polyfit result:' + 'sample_num =' + str(sample_num) + ',polynomial_order = ' + str(polynomial_order))
print(case1)

# 情况：不加惩罚项
case2 = Penalty_item_ErrorFunction(sample_x, sample_y, sample_num, polynomial_order)
Draw_Images(sample_x, sample_y, case2,
            'Penalty_item_ErrorFunction result:' + 'sample_num =' + str(sample_num) + ',polynomial_order = ' + str(
                polynomial_order))
print(case2)

# 情况：加惩罚项
lamda1 = np.exp(-7)
case3 = Penalty_item_add_ErrorFunction(sample_x, sample_y, lamda1, sample_num, polynomial_order)
Draw_Images(sample_x, sample_y, case3,
            'Penalty_item_add_ErrorFunction result:' + 'sample_num =' + str(sample_num) + ',polynomial_order = ' + str(
                polynomial_order))
print(case3)

# 情况：加惩罚项,对误差函数使用梯度下降法
lamda2 = np.exp(-7)
descending_step_size = 0.05
cycle_times = 100000
iteration_error1 = 1e-5
case4 = Gradient_descent_method(sample_x, sample_y, lamda2, cycle_times, descending_step_size, iteration_error1,
                                sample_num, polynomial_order)
Draw_Images(sample_x, sample_y, case4,
            'Gradient_descent_method result:' + 'sample_num =' + str(sample_num) + ',polynomial_order = ' + str(
                polynomial_order))
print(case4)

# 情况：加惩罚项,对误差函数使用共轭梯度法
lamda3 = np.exp(-7)
case5 = Gradient_conjugate_method(sample_x, sample_y, lamda3, sample_num, polynomial_order)
Draw_Images(sample_x, sample_y, case5,
            'Gradient_conjugate_method result:' + 'sample_num =' + str(sample_num) + ',polynomial_order = ' + str(
                polynomial_order))
print(case5)
