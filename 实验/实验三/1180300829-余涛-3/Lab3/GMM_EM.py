import numpy as np
from scipy.stats import multivariate_normal
import collections


class GMM_EM_model(object):

    def __init__(self, data, k, error=1e-12, iteration_times=1000):
        self.data = data  # 数据集
        self.k = k  # k-means的k值
        self.error = error  # 判断两个浮点数是否相等的误差值
        self.iteration_times = iteration_times  # 最大迭代次数
        self.data_rows, self.data_columns = self.data.shape
        self.alpha = np.ones(self.k) * (1.0 / self.k)
        self.mu_data, self.my_sigma = self.initial_params()  # 初始化初始簇中心点集合
        self.lamda = None  # 每个样本的簇标记
        self.c = collections.defaultdict(list)  # 簇划分
        self.last_alpha = self.alpha  # 保存更新后的混合系数
        self.last_mu = self.mu_data  # 保存更新后的均值向量系数
        self.last_sigma = self.my_sigma  # 保存更新后的协方差参数
        self.gamma = None  # 后验概率分布

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
                all_length.append(np.sum(
                    [self.distance_by_euclidean(self.data[i], mu_collection[j]) for j in range(len(mu_collection))]))
            mu_collection.append(self.data[np.argmax(all_length)])  # 取距离最大的点下标加入k集合
        print('初始均值向量集合为：\n', np.array(mu_collection))
        return np.array(mu_collection)

    '''
    初始化高斯混合分布的模型参数中的u和sigma
    进行准备工作，生成初始簇中心点集合{u1，u2,...,uk},储存了所有的均值向量点，和生成大小为k×k的对角矩阵sigma
    '''
    def initial_params(self):
        # mu = np.array(self.data[random.sample(range(self.data_rows), self.k)])
        # 随机选择k个点作为初始点 极易陷入局部最小值
        mu_collection = self.initial_cluster_center_point_by_maxlength()  # 选择彼此距离尽可能远的K个点作为初始簇中心点集合
        sigma = collections.defaultdict(list)  # 定义一个集合sigma
        for i in range(self.k):  # 构建k个对角线元素为0.1的，对角矩阵作为sigma的值，是协方差矩阵
            sigma[i] = np.eye(self.data_columns, dtype=float) * 0.1
        return mu_collection, sigma

    '''
    EM算法
    '''
    def EM_algorithm(self):
        temp_likelihoods = np.zeros((self.data_rows, self.k))  # 生成行为data_rows，列为k的矩阵，作为后验概率公式的分母的一部分
        for i in range(self.k):  # 对每一列
            temp_likelihoods[:, i] = multivariate_normal.pdf(self.data, self.mu_data[i],
                                                             self.my_sigma[i])  # 得到所有数据在mu_data[i]取值点附近的可能性
            # print("sadasdasdad",likelihoods[:, i])
        # 求期望E，即EM算法的E步
        the_weighted_likelihoods = temp_likelihoods * self.alpha  # 还未求和的后验概率的分母
        sum_likelihoods = np.expand_dims(np.sum(the_weighted_likelihoods, axis=1), axis=1)  # 对后验概率的分母求和
        print('似然值为：', np.log(np.prod(sum_likelihoods)))  # 输出似然值
        self.gamma = the_weighted_likelihoods / sum_likelihoods  # 根据公式，得到后验概率的值，即所有gamma的值
        # print('sadadawd',self.lamda)
        self.lamda = self.gamma.argmax(axis=1)  # 求得最大的gamma所在的簇标记lamda，即得到了每个样本的簇标记
        for i in range(self.data_rows):  # 根据簇标记将所有的数据划分簇
            self.c[self.lamda[i]].append(self.data[i].tolist())
        # 最大化M，即EM算法的M步
        for i in range(self.k):
            gamma = np.expand_dims(self.gamma[:, i], axis=1)  # 提取每一列,作为列向量
            self.mu_data[i] = (gamma * self.data).sum(axis=0) / gamma.sum()  # 根据公式，更新新的均值向量参数
            self.my_sigma[i] = (self.data - self.mu_data[i]).T.dot(
                (self.data - self.mu_data[i]) * gamma) / gamma.sum()  # 根据公式，更新新的协方差参数
            self.alpha = self.gamma.sum(axis=0) / self.data_rows  # 更新新的混合系数

    '''
    GMM算法，实现对样本的聚类
    '''
    def GMM_algorithm(self):
        print("GMM算法：\n")
        for i in range(self.iteration_times):  # 进行迭代以得到最终的均值向量参数，协方差参数和混合系数
            print('第', i+1, "次迭代：")
            self.EM_algorithm()
            loss = np.linalg.norm(self.last_alpha - self.alpha) \
                   + np.linalg.norm(self.last_mu - self.mu_data) \
                   + np.sum([np.linalg.norm(self.last_sigma[i] - self.my_sigma[i]) for i in range(self.k)])  # 两次迭代的误差
            if loss > self.error:  # 如果新产生的均值向量参数，协方差参数和混合系数与原均值向量参数，协方差参数和混合系数的误差大于给定误差范围，则进行更新
                self.last_sigma = self.my_sigma
                self.last_mu = self.mu_data
                self.last_alpha = self.alpha
            else:  # 迭代终止条件：新产生的均值向量参数，协方差参数和混合系数与原均值向量参数，协方差参数和混合系数误差几乎可以忽略
                break
        self.EM_algorithm()  # 由于之只更新了均值向量参数，协方差参数和混合系数，需要再运行一次EM来得到新的似然值
        return self.mu_data, self.c
