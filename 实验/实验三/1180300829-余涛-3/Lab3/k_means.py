import numpy as np
import collections
import random


class KMeans(object):
    def __init__(self, data, k, error=1e-6):
        self.data = data  # 数据集
        self.k = k  # k-means的k值
        self.error = error  # 判断两个浮点数是否相等的误差值
        self.data_rows, self.data_columns = data.shape
        self.mu_datas = self.initial_cluster_center_point_by_maxlength()  # 初始簇中心点集合,即初始均值向量集合
        self.sample_assignments = [-1] * self.data_rows

    @staticmethod
    def distance_by_euclidean(x1, x2):
        return np.linalg.norm(x1 - x2)

    '''
    选择彼此距离尽可能远的K个点作为初始簇中心点集合{u1，u2,...,uk},储存了所有的均值向量点
    '''
    def initial_cluster_center_point_by_maxlength(self):
        mu_0 = np.random.randint(0, self.k) + 1  # 首先在k个点中随机选第一个初始点
        mu_collection = [self.data[mu_0]]  # 定义一个初始簇中心点集合（k集合），将选取的第一个点放入其中
        for m in range(self.k - 1):  # 除了第一个点还需要选取k-1个点，每次选择一个距离最大的点加入到k集合中
            all_length = []
            for i in range(self.data_rows):  # 对于样本集中的所有数据，求得其到k集合的距离
                all_length.append(np.sum([self.distance_by_euclidean(self.data[i], mu_collection[j])for j in range(len(mu_collection))]))
            mu_collection.append(self.data[np.argmax(all_length)])  # 取距离最大的点下标加入k集合
        print('初始均值向量集合为：\n', np.array(mu_collection))
        return np.array(mu_collection)

    '''
    k_means算法的实现，实现对样本的聚类
    '''
    def the_method_of_kmeans(self):
        number_of_times = 0  # 循环次数
        flag = 0  # 设置循环结束标签
        while True:
            c = collections.defaultdict(list)  # 初始化Ci=空，i=1,2,3,...,k，C为最终的初始均值向量集
            for i in range(self.data_rows):  # 对数据集中所有的点
                dij = [self.distance_by_euclidean(self.data[i], self.mu_datas[j]) for j in
                       range(self.k)]  # 对初始簇中心点集合中的所有均值向量点u1，u2,...,uk,求得两者的||xi-uj||
                lambda_j = np.argmin(dij)  # 求得数据集中最小的距离的簇的下标
                c[lambda_j].append(self.data[i].tolist())  # 将第i个点划分到C-lambda-j对应的簇中
                self.sample_assignments[i] = lambda_j
            new_mu = np.array([np.mean(c[i], axis=0).tolist() for i in range(self.k)])  # 求解每个簇的均值作为新的均值向量
            flag = 0  # 初始化标签为0
            for m in range(self.k):  # 对于所有的i属于1,2,3,...,k，需要所有新的ui都等于旧的ui,由于是浮点数，所以只需要两次的ui误差小于给定误差即可
                if self.distance_by_euclidean(self.mu_datas[m],
                                              new_mu[m]) > self.error:  # 若大于给定误差，则将新的ui赋值给旧的ui，并置flag为1
                    self.mu_datas[m] = new_mu[m]
                    flag = 1
            if flag == 0:  # 当没有新的ui产生后循环退出
                break
            print('kmeans得到的均值向量的循环次数为:', number_of_times)
            number_of_times += 1
            print('得到的均值向量集合为：\n', self.mu_datas)
        return self.mu_datas, c

    '''
    随机选择k个顶点作为初始簇中心点
    '''
    def k_means_by_random_selection(self):
        self.mu_datas = self.data[random.sample(range(self.data_rows), self.k)]  # 产生随机的K个点作为初始簇中心点集合{u1，u2,...,uk}
        return self.the_method_of_kmeans()

    '''
    先随机选择第一个簇中心点 再选择彼此距离最大的k个顶点作为初始簇中心点
    '''
    def k_means_by_maxlength(self):
        self.mu_datas = self.initial_cluster_center_point_by_maxlength()  # 选择彼此距离尽可能远的K个点作为初始簇中心点集合{u1，u2,...,uk},
        return self.the_method_of_kmeans()
