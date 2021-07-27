import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def create_data_by_two_or_three_dimension(data_dimension, num):
    """
    生成三维或二维数据集
    :param data_dimension: 需要生成的维度
    :param num: 需要生成的数据集的数据量
    :return: 生成的数据集
    """
    if data_dimension == 2:  # 对二维数据定义均值和方差
        mean = [-2, 2]
        cov = [[1, 0], [0, 0.01]]
    elif data_dimension == 3:  # 对三维数据定义均值和方差
        mean = [1, 2, 3]
        cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        assert False
    data_set = []  # 定义数据集
    for index in range(num):  # 生成num个数据，加入数据集
        data_set.append(np.random.multivariate_normal(mean, cov).tolist())
    return np.array(data_set)


def PCA(data_set, k):
    """
    将数据集data_set用PCA从D维降至k维，data_set.shape = (N, D)
    :param data_set:原始数据集
    :param k:PCA后的维度
    :return:center_data，中心化后的数据，shape=(N, D)。eigenvector_matrix，特征向量矩阵，shape=(D, k)。data_mean，降维前样本均值，shape=(1, D)
    """
    rows, cols = data_set.shape  # 得到数据集的行和列
    data_mean = np.sum(data_set, 0) / rows  # 计算降维前样本均值
    center_data = data_set - data_mean  # 进行数据集的中心化操作
    covariance_matrix = np.dot(center_data.T, center_data)  # 计算协方差矩阵X.T · X
    eigenvalue, feature_vectors = np.linalg.eig(covariance_matrix)  # 对协方差矩阵(D,D)进行特征值分解，分别求得特征值和特征向量
    eigenvalue_sorted = np.argsort(eigenvalue)  # 将所有特征值排序
    eigenvector_matrix = feature_vectors[:, eigenvalue_sorted[:-(k + 1):-1]]  # 取出前k个最大的特征值对应的特征向量组成特征向量矩阵
    return center_data, eigenvector_matrix, data_mean


def read_from_file(file_path):
    """
    从文件中中读取面部图像数据并压缩
    :param file_path: 文件路径
    :return: 返回解析面部图像得到的数据集
    """
    size = (50, 50)  # 由于较大的数据在求解特征值和特征向量时很慢，故统一压缩图像为size大小
    i = 1
    file_list = os.listdir(file_path)  # 读取该路径下所有图像的列表，放入file_list
    data_set = []  # 定义数据集
    plt.figure(figsize=size)
    for file in file_list:  # 对于file_list中所有图像
        path = os.path.join(file_path, file)  # 连接文件路径，得到每个图像的路径
        plt.subplot(3, 5, i)
        with open(path) as f:
            image = cv2.imread(path)  # 读取这张图像
            image = cv2.resize(image, size)  # 将图像压缩至size大小
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像通过三通道转换为灰度图
            plt.imshow(image_gray, cmap="gray")  # 预览该图像
            h, w = image_gray.shape  # 得到该图像的维度
            image_col = image_gray.reshape(h * w)  # 对(h,w)的图像数据拉平
            data_set.append(image_col)  # 加入该图像数据给数据集中
        i += 1
    plt.show()
    return np.array(data_set)
