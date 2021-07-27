import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Lab2.descent_gradient_add_errorfunction import descent_gradient_add_errorfunction
from Lab2.draw import draw_picture_loss

'''
Skin_NonSkin数据集是三个维度的，所以可以画出三维的图像。
'''
def Skin_NonSkin_draw_picture(train_point_X, classification_Y, function_coefficient):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('3D of the regression curve')
    ax.scatter(train_point_X[:, 0], train_point_X[:, 1], train_point_X[:, 2], c=classification_Y, cmap=plt.cm.Spectral)
    real_x = np.arange(np.min(train_point_X[:, 0]), np.max(train_point_X[:, 0]), 1)
    real_y = np.arange(np.min(train_point_X[:, 1]), np.max(train_point_X[:, 1]), 1)
    real_X, real_Y = np.meshgrid(real_x, real_y)
    real_z = function_coefficient[0] + function_coefficient[1] * real_X + function_coefficient[2] * real_Y
    ax.plot_surface(real_x, real_y, real_z, rstride=1, cstride=1)
    ax.set_zlim(np.min(real_z) - 10, np.max(real_z) + 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


'''
读入Skin_NonSkin.csv文件，对数据进行拆分，拆分成训练集合测试集
由于原文件中数据量巨大，所以对数据集以50步长取部分数据作为数据集
'''
def Skin_NonSkin_getdata():
    all_data = np.loadtxt(open('./Skin_NonSkin.csv'), delimiter=",", skiprows=0)  # 读取文件中的所有数据
    np.random.shuffle(all_data)  # 将原数据集打乱，分成训练集和测试集
    test_rate = 0.2  # 测试集所占比例
    all_data_size = np.size(all_data, axis=0)  # 总数据集数据数量
    train_data_X = all_data[:int(test_rate * all_data_size), :]  # 训练集数据
    test_data_x = all_data[int(test_rate * all_data_size):, :]  # 测试集数据
    dimension = np.size(all_data, axis=1) - 1  # 训练集样本维度
    step = 50  # 由于Skin_NonSkin的数据集太大，所以采用步长为50的方式取数据
    train_point_X = train_data_X[:, 0:dimension]  # 将所有数据集赋给train_point_X
    train_point_X = train_point_X[::step]  # 以step为间隔取数据
    train_point_X = train_point_X - 100  # 对样本点进行坐标平移，方便在3D图中显示
    train_classification_Y = train_data_X[:, dimension:dimension + 1] - 1  # 因为数据集的分类是1/2,需要减1变成0/1
    train_classification_Y = train_classification_Y[::step]  # 以step为间隔取数据
    train_size = np.size(train_point_X, axis=0)  # 训练集数据总数
    train_classification_Y = train_classification_Y.reshape(train_size)  # 将矩阵转化为行向量
    test_point_X = test_data_x[:, 0:dimension]  # 将所有数据集赋给test_point_X
    test_point_X = test_point_X[::step] - 100  # 对样本点进行坐标平移，方便在3D图中显示
    test_classification_Y = test_data_x[:, dimension:dimension + 1] - 1  # 因为数据集的分类是1/2,需要减1变成0/1
    test_classification_Y = test_classification_Y[::step]  # 以step为间隔取数据
    test_size = np.size(test_point_X, axis=0)  # 测试集数据总数
    test_classification_Y = test_classification_Y.reshape(test_size)  # 将矩阵转化为行向量
    return train_point_X, train_classification_Y, test_point_X, test_classification_Y


'''
使用Skin_NonSkin.csv上的数据进行试验
参数中lamda为惩罚项系数，cycle_times为梯度下降迭代最大次数,descending_step_size为梯度下降下降步长,iteration_error为梯度下降迭代误差
'''
def Skin_NonSkin_experiment(lamda, cycle_times, descending_step_size, iteration_error):
    train_point_X, train_classification_Y, test_point_X, test_classification_Y = Skin_NonSkin_getdata()  # 得到Skin_NonSkin.csv上的训练集样本和测试集样本
    train_size = np.size(train_point_X, axis=0)  # 训练集样本数量
    test_size = np.size(test_point_X, axis=0)  # 测试集样本数量
    dimension = np.size(train_point_X, axis=1)  # 样本维度
    train_all = np.ones((train_size, dimension + 1))  # 创建行为train_size，列为样本维度+1的矩阵train_all
    for i in range(dimension):  # 依次将训练集样本的每一个维度放入train_all的下一个维度
        train_all[:, i + 1] = train_point_X[:, i]
    w, cycle_times_list, loss_list = descent_gradient_add_errorfunction(train_all, train_classification_Y, cycle_times,
                                                                        descending_step_size, iteration_error,
                                                                        dimension, lamda)
    w = w.reshape(-1)  # 得到的w是一个一行dimension + 1列的矩阵,需要先将w改成行向量
    function_coefficient = - (w / w[dimension])[0:dimension]  # w整体除y的系数然后移项得到决策面系数
    Skin_NonSkin_draw_picture(train_point_X, train_classification_Y, function_coefficient)
    draw_picture_loss(cycle_times_list, loss_list)
    # 计算测试集的准确率
    label = np.ones(test_size)
    hit_count = 0
    test_all = np.ones((test_size, dimension + 1))  # 创建行为train_size，列为样本维度+1的矩阵test_all
    for i in range(dimension):  # 依次将测试集样本的每一个维度放入train_all的下一个维度
        test_all[:, i + 1] = test_point_X[:, i]
    for i in range(test_size):  # 对每种预测给label进行赋值
        if w.dot(test_all[i].T) >= 0:
            label[i] = 1
        else:
            label[i] = 0
    for i in range(test_size):
        if label[i] == test_classification_Y[i]:  # 如果预测的结果与真实结果相同计数加一
            hit_count += 1
    hit_rate = hit_count / test_size
    print('数据的测试集的准确率为：', hit_rate)
