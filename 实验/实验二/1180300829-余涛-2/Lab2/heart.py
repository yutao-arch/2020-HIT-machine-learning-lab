import numpy as np

from Lab2.descent_gradient_add_errorfunction import descent_gradient_add_errorfunction
from Lab2.draw import draw_picture_loss

'''
读入heart.csv文件，对数据进行拆分，拆分成训练集合测试集
'''
def heart_getdata():
    all_data = np.loadtxt(open('./heart.csv'), delimiter=",", skiprows=0)  # 读取文件中的所有数据
    np.random.shuffle(all_data)  # 将原数据集打乱，分成训练集和测试集
    test_rate = 0.2  # 测试集所占比例
    all_data_size = np.size(all_data, axis=0)  # 总数据集数据数量
    train_data_X = all_data[:int(test_rate * all_data_size), :]  # 训练集数据
    test_data_x = all_data[int(test_rate * all_data_size):, :]  # 测试集数据
    dimension = np.size(all_data, axis=1) - 1  # 数据集样本维度
    # 消除exp溢出，防止数据太大导致exp溢出
    for i in range(dimension):  # 对于样本的所有维度
        d_length = max(train_data_X[:, i]) - min(train_data_X[:, i])  # 计算最大值和最小值之间的极差
        for j in range(np.size(train_data_X, axis=0)):  # 对于每一维度的所有数
            train_data_X[j, i] = (max(train_data_X[:, i]) - train_data_X[j, i]) / d_length  # 将其化为[0,1]之间的值，防止exp溢出
    train_point_X = train_data_X[:, 0:dimension]  # 将所有数据集赋给train_point_X
    train_classification_Y = train_data_X[:, dimension:dimension + 1]  # 为train_classification_Y赋值为0/1
    train_size = np.size(train_point_X, axis=0)  # 训练集数据总数
    train_classification_Y = train_classification_Y.reshape(train_size)  # 将矩阵转化为行向量
    test_point_X = test_data_x[:, 0:dimension]  # 将所有数据集赋给test_point_X
    test_classification_Y = test_data_x[:, dimension:dimension + 1]  # 为test_classification_Y赋值为0/1
    test_size = np.size(test_point_X, axis=0)  # 测试集数据总数
    test_classification_Y = test_classification_Y.reshape(test_size)  # 将矩阵转化为行向量
    return train_point_X, train_classification_Y, test_point_X, test_classification_Y


'''
使用heart.csv上的数据进行试验
参数中lamda为惩罚项系数，cycle_times为梯度下降迭代最大次数,descending_step_size为梯度下降下降步长,iteration_error为梯度下降迭代误差
'''
def heart_exp(lamda, cycle_times, descending_step_size, iteration_error):
    train_point_X, train_classification_Y, test_point_X, test_classification_Y = heart_getdata()
    train_size = np.size(train_point_X, axis=0)  # 训练集样本数量
    test_size = np.size(test_point_X, axis=0)  # 测试集样本数量
    dimension = np.size(train_point_X, axis=1)  # 样本维度
    # 构造训练集样本矩阵
    train_all = np.ones((train_size, dimension + 1))  # 创建行为train_size，列为样本维度+1的矩阵train_all
    for i in range(dimension):  # 依次将训练集样本的每一个维度放入train_all的下一个维度
        train_all[:, i + 1] = train_point_X[:, i]
    w, cycle_times_list, loss_list = descent_gradient_add_errorfunction(train_all, train_classification_Y, cycle_times,
                                                                        descending_step_size, iteration_error,
                                                                        dimension, lamda)
    w = w.reshape(-1)  # 得到的w是一个一行dimension + 1列的矩阵,需要先将w改成行向量
    function_coefficient = - (w / w[dimension])[0:dimension]  # w整体除y的系数然后移项得到决策面系数
    draw_picture_loss(cycle_times_list, loss_list)
    # 计算测试集准确率
    label = np.ones(test_size)
    hit_count = 0
    test_all = np.ones((test_size, dimension + 1))  # 创建行为train_size，列为样本维度+1的矩阵test_all
    for i in range(dimension):  # 依次将测试集样本的每一个维度放入train_all的下一个维度
        test_all[:, i + 1] = test_point_X[:, i]
    for i in range(test_size):  # 对每种预测给label进行赋值
        if np.dot(w, test_all[i].T) >= 0:
            label[i] = 1
        else:
            label[i] = 0
    for i in range(test_size):
        if label[i] == test_classification_Y[i]:  # 如果预测的结果与真实结果相同计数加一
            hit_count += 1
    hit_rate = hit_count / test_size
    print('数据的测试集的准确率为：', hit_rate)
