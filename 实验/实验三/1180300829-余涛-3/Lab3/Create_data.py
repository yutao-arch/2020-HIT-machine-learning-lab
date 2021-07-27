import numpy as np


'''
生成2维数据集
参数中：
sample_means表示k类数据的均值，以list的形式给出，如[[1, 2],[-1, -2], [0, 0]]
sample_number表示k类数据的数量，以list的形式给出，如[10, 20, 30]
category_K表示数据一共分为k类
'''
def create_data_two_dimensional(sample_means, sample_number, category_K):
    covariance = [[0.1, 0], [0, 0.1]]  # 协方差
    sample_data = []
    for index in range(category_K):
        for times in range(sample_number[index]):
            sample_data.append(np.random.multivariate_normal(
                [sample_means[index][0], sample_means[index][1]], covariance).tolist())
    return np.array(sample_data)