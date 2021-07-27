import numpy as np
import matplotlib.pyplot as plt

'''
对于train_point_X中，含有二维数据，分别对应横坐标x和纵坐标y
对于classification_Y,为train_point_X中两种数据的点对应的类别
根据train_point_X和classification_Y画出二维坐标下的点图，然后画出分界判别函数boundary_check_function
'''
def draw_picture(train_point_X, classification_Y, boundary_check_function):
    if boundary_check_function:  # 绘制分界判定函数boundary_check_function
        d_length = max(train_point_X[:, 0]) - min(train_point_X[:, 0])  # 找到横坐标x的极差
        real_x = np.linspace(min(train_point_X[:, 0]), min(train_point_X[:, 0]) + d_length, 50)  # 在x的范围内均匀产生50个点
        real_y = boundary_check_function(real_x)  # 对real_x每个点调用boundary_check_function求解对应的real_y
        plt.plot(real_x, real_y, 'r', label='boundary_check_function')  # 绘制图像
    plot = plt.scatter(train_point_X[:, 0], train_point_X[:, 1], s=30, c=classification_Y, marker='o',
                       cmap=plt.cm.Spectral)  # 绘制两种点集
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title('The regression curve')
    plt.show()


'''
根据迭代次数cycle_times_list和对应的误差loss_list画出损失函数图像
参数中，cycle_times_list为迭代次数表，loss_list为对应的误差表
'''
def draw_picture_loss(cycle_times_list, loss_list):
    plt.plot(cycle_times_list, loss_list, 'r', label='loss_function')
    plt.xlabel('cycle_times')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.title('the loss_funciton')
    plt.show()
