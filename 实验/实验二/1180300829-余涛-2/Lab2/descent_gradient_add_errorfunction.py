import numpy as np

'''
sigmoid函数a=1/(1+exp(-b)
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
自己生成数据
根据是否满足贝叶斯假设，在范围[0,1]上生成sample_number个二维点集，并将其均匀分为两类
并添加一个N(0,1)的高斯噪声
'''
def create_datas(sample_number, satisfy_naive):
    half = np.ceil(sample_number / 2).astype(np.int32)  # sample_number的一半
    variance = 0.2  # 随机变量方差
    covariance_xy = 0.01  # 两个维度的协方差
    point_mean1 = [-0.7, -0.3]  # 类别1的均值
    point_mean2 = [0.7, 0.3]  # 类别2的均值
    train_point_X = np.zeros((sample_number, 2))  # 二维的点集
    classification_Y = np.zeros(sample_number)  # 点集的分类
    if satisfy_naive:  # 满足朴素贝叶斯假设
        train_point_X[:half, :] = np.random.multivariate_normal(point_mean1, [[variance, 0], [0, variance]],
                                                                size=half)  # 生成half个类别1的1*2数组，每个数组含有两个维度
        train_point_X[half:, :] = np.random.multivariate_normal(point_mean2, [[variance, 0], [0, variance]],
                                                                size=sample_number - half)  # 生成half个类别2的1*2数组，每个数组含有两个维度
        classification_Y[:half] = 0  # 将前half个数组标记为类别1
        classification_Y[half:] = 1  # 将后half个数组标记为类别2
    else:  # 不满足朴素贝叶斯假设
        train_point_X[:half, :] = np.random.multivariate_normal(point_mean1,
                                                                [[variance, covariance_xy], [covariance_xy, variance]],
                                                                size=half)  # 生成half个类别1的1*2数组，每个数组含有两个维度
        train_point_X[half:, :] = np.random.multivariate_normal(point_mean2,
                                                                [[variance, covariance_xy], [covariance_xy, variance]],
                                                                size=sample_number - half)  # 生成half个类别2的1*2数组，每个数组含有两个维度
        classification_Y[:half] = 0  # 将前half个数组标记为类别1
        classification_Y[half:] = 1  # 将后half个数组标记为类别2
    return train_point_X, classification_Y  # 返回生成的所有点及类别



'''
根据公式得到点集的极大条件似然得到极大条件似然l(W)
'''
def maximum_conditional_likelihood(train_point_X, classification_Y, w):
    total_points = np.size(train_point_X, axis=0)  # 得到train_point_X的行数，即点集个数
    predict = np.zeros((total_points, 1))
    for i in range(total_points):
        predict[i] = w.dot(train_point_X[i].T)  # 极大条件似然中的exp中的部分，即求和wi * Xl
    t = 0
    for i in range(total_points):
        t += np.log(1 + np.exp(predict[i]))  # 极大条件似然中ln的部分，即ln ( 1 + exp ( 求和wi * Xi ) )
    MCLE = classification_Y.dot(predict) - t  # 得到极大条件似然l(w)
    return MCLE


'''
加惩罚项的梯度下降法
对于train_point_X和classification_Y对参数w做梯度下降，对损失函数使用梯度下降法，当误差函数收敛到期望的最小值时，得到此时的w并返回w
参数中：
迭代最大次数为cycle_times
下降步长descending_step_size
迭代误差iteration_error
数据点集维度dimension
惩罚项参数lamda
'''
def descent_gradient_add_errorfunction(train_point_X, classification_Y, cycle_times, descending_step_size,
                                       iteration_error, dimension, lamda):
    total_points = np.size(train_point_X, axis=0)  # 得到train_point_X的行数，即点集个数
    w = np.ones((1, dimension + 1))  # 生成系数矩阵w，一个列数为dimension + 1，行数为1的矩阵，元素值全为1
    cycle_times_list = np.zeros(cycle_times)  # 迭代次数统计
    loss_list = np.zeros(cycle_times)  # 迭代次数对应的损失函数值统计
    for i in range(cycle_times):
        old_loss = - 1 / total_points * maximum_conditional_likelihood(train_point_X, classification_Y, w)  # 原先损失函数的值
        t = np.zeros((total_points, 1))
        for j in range(total_points):
            t[j] = w.dot(train_point_X[j].T)  # 极大条件似然中的exp中的部分，即求和wi * Xl
        gradient = - 1 / total_points * (classification_Y - sigmoid(t.T)).dot(train_point_X)
        w = w - descending_step_size * lamda * w - descending_step_size * gradient  # 梯度下降加惩罚项的迭代公式
        new_loss = - 1 / total_points * maximum_conditional_likelihood(train_point_X, classification_Y, w)  # 新的损失函数的值
        cycle_times_list[i] = i  # 储存迭代次数
        loss_list[i] = new_loss  # 储存每次迭代对应的损失函数值
        if abs(new_loss - old_loss) < iteration_error:  # 如果新的误差函数值与旧的误差函数值的差小于迭代误差则终止迭代
            cycle_times_list = cycle_times_list[:i + 1]
            loss_list = loss_list[:i + 1]
            print('迭代次数=', i, '\n对应的损失函数值=', new_loss, '\n对应的系数w=', w, '\n对应的梯度=', gradient)
            break
        if new_loss > old_loss:  # 当新的误差函数值大于旧的误差函数值时，将步长变为原来的一半
            descending_step_size *= 0.5
    return w, cycle_times_list, loss_list