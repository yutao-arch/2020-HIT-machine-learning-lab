import numpy as np
import matplotlib.pyplot as plt
import math


def draw_picture_by_create_PCA(dimension, data_before_PCA, x_after_PCA):
    """
    对执行PCA前后的数据集，在图像上显示
    :param dimension: 维度
    :param data_before_PCA: 原始数据集
    :param x_after_PCA: 执行PCA之后的数据集
    :return:
    """
    if dimension == 2:  # 对二维数据画图
        plt.scatter(data_before_PCA[:, 0], data_before_PCA[:, 1], facecolor="none", edgecolor="b",
                    label="data_before_PCA")
        plt.scatter(x_after_PCA[:, 0], x_after_PCA[:, 1], facecolor='r', label='data_after_PCA')
        plt.xlabel('x')
        plt.ylabel('y')
    elif dimension == 3:  # 对三维数据画图
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(data_before_PCA[:, 0], data_before_PCA[:, 1], data_before_PCA[:, 2], edgecolor="b",
                   label='data_before_PCA')
        ax.scatter(x_after_PCA[:, 0], x_after_PCA[:, 1], x_after_PCA[:, 2], facecolor='r', label='data_after_PCA')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        assert False
    plt.legend()
    plt.show()


def PSNR(image1, image2):
    """
    计算两章图像的峰值信噪比PSNR
    :param image1: 第一张图像
    :param image2: 第二张图像
    :return: 信噪比PSNR
    """
    mse = np.mean((image1 / 255. - image2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    max_pixel = 1  # 将最大像素设置为1
    return 20 * math.log10(max_pixel / math.sqrt(mse))  # 计算信噪比即可


def draw_picture_by_image(data_set, w, center_data, mu_x, x_num):
    """
    输出PCA后的图像并打印信噪比
    :param data_set: PCA前的数据集
    :param w: 特征向量矩阵
    :param center_data: 中心化后的数据
    :param mu_x: 降维前样本均值
    :param x_num: 数据个数
    """
    size = (50, 50)  # 由于较大的数据在求解特征值和特征向量时很慢，故统一压缩图像为size大小
    w = np.real(w)  # 当降维后的维度超过某个值，特征向量矩阵将出现复向量，对其保留实部
    x_after_PCA = np.dot(center_data, w)  # 计算降维后的数据
    refactoring_data = np.dot(x_after_PCA, w.T) + mu_x  # 重构降维后的数据
    plt.figure(figsize=size)
    for i in range(x_num):
        plt.subplot(3, 5, i + 1)
        plt.imshow(refactoring_data[i].reshape(size), cmap="gray")  # 预览该图像
    plt.show()
    print("PCA后的信噪比如下所示：")
    for i in range(x_num):  # 打印信噪比
        psnr = PSNR(data_set[i], refactoring_data[i])
        print('图像', i + 1, '的信噪比: ', psnr)
