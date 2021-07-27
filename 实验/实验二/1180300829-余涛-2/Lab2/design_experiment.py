import numpy as np

from Lab2.descent_gradient_add_errorfunction import create_datas, descent_gradient_add_errorfunction
from Lab2.draw import draw_picture, draw_picture_loss

'''
自定义二维点集进行试验并绘制图像
参数中：sample_number为点集中点的个数，lamda为惩罚项系数（可以为0，此时没有惩罚项），satisfy_naive为是否满足朴素贝叶斯假设
cycle_times为梯度下降迭代最大次数,descending_step_size为梯度下降下降步长,iteration_error为梯度下降迭代误差
'''
def design_experiment(sample_number, lamda, satisfy_naive, cycle_times, descending_step_size, iteration_error):
    train_point_X, classification_Y = create_datas(sample_number, satisfy_naive)  # 生成sample_number个二维点集数据
    train_all = np.ones((sample_number, 3))  # 创建行为sample_number，列为3的矩阵train_all
    train_all[:, 1] = train_point_X[:, 0]  # 将生成点集的第一个维度放入train_all的第二列
    train_all[:, 2] = train_point_X[:, 1]  # 将生成点集的第二个维度放入train_all的第三列
    dimension = np.size(train_point_X, axis=1)
    w, cycle_times_list, loss_list = descent_gradient_add_errorfunction(train_all, classification_Y, cycle_times,
                                                                        descending_step_size, iteration_error,
                                                                        dimension, lamda)
    w = w.reshape(-1)  # 得到的w是一个一行三列的矩阵,需要先将w改成行向量
    function_coefficient = -(w / w[2])[0:2]  # w整体除y的系数然后移项得到决策面系数
    boundary_check_function = np.poly1d(function_coefficient[::-1])  # 将function_coefficient从后往前倒序然后调用poly1d得到多项式函数
    print('分界判别函数为: y = ', boundary_check_function)
    draw_picture(train_point_X, classification_Y, boundary_check_function)
    draw_picture_loss(cycle_times_list, loss_list)
