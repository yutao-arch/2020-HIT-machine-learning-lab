import numpy as np
import pandas as pd
import itertools


class read_from_csv(object):
    def __init__(self):
        self.all_data = pd.read_csv("./iris.csv")  # 读取文件数据集
        self.data_x = self.all_data.drop('class', axis=1)  # 删除class类别列作为数据集data_x
        self.data_y = self.all_data['class']  # 将class类别列作为分类数据集data_y
        self.classes = list(itertools.permutations(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 3))

    '''
    返回拆分得到的数据集data_x
    '''
    def get_data(self):
        return np.array(self.data_x, dtype=float)

    '''
    测试聚类的正确率
    '''
    def test_accuracy(self, y_label):
        number = len(self.data_y)
        counts = []
        for i in range(len(self.classes)):
            count = 0
            for j in range(number):
                if self.data_y[j] == self.classes[i][y_label[j]]:
                    count += 1
            counts.append(count)
        return np.max(counts) * 1.0 / number