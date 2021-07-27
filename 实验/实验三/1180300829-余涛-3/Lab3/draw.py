import numpy as np
import matplotlib.pyplot as plt

'''
画随机选择均值向量的kmeans图像
'''
def draw_k_means_by_random_selection(k, mu_random, c_random):
    for i in range(k):
        plt.scatter(np.array(c_random[i])[:, 0], np.array(c_random[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(mu_random[:, 0], mu_random[:, 1], facecolor="none", edgecolor="r", label="center")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title("k_means_by_random_selection")
    plt.show()


'''
画彼此距离尽可能远的K个点的均值向量的kmeans图像
'''
def draw_k_means_by_maxlength(k, mu_maxlength, c_maxlength):
    for i in range(k):
        plt.scatter(np.array(c_maxlength[i])[:, 0], np.array(c_maxlength[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(mu_maxlength[:, 0], mu_maxlength[:, 1], facecolor="none", edgecolor="r", label="center")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title("k_means_by_maxlength")
    plt.show()


'''
画彼此距离尽可能远的K个点的均值向量的GMM图像
'''
def draw_GMM(k, mu_gmm, c_gmm):
    for i in range(k):
        plt.scatter(np.array(c_gmm[i])[:, 0], np.array(c_gmm[i])[:, 1], marker="x", label=str(i + 1))
    plt.scatter(mu_gmm[:, 0], mu_gmm[:, 1], facecolor="none", edgecolor="r", label="center")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title("GMM_by_maxlength")
    plt.show()