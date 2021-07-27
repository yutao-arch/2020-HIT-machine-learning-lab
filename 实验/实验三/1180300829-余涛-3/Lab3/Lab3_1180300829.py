from Lab3 import k_means, GMM_EM, read_from_document
from Lab3.Create_data import create_data_two_dimensional
from Lab3.draw import draw_k_means_by_random_selection, draw_k_means_by_maxlength, draw_GMM

k = 3
means = [[2, 4], [0, -4], [-2, 2]]
number = [100, 100, 100]
data = create_data_two_dimensional(means, number, k)


kmeans = k_means.KMeans(data, k)
mu_random, c_random = kmeans.k_means_by_random_selection()
mu_maxlength, c_maxlength = kmeans.k_means_by_maxlength()


draw_k_means_by_random_selection(k, mu_random, c_random)

draw_k_means_by_maxlength(k, mu_maxlength, c_maxlength)

GMM = GMM_EM.GMM_EM_model(data, k)
mu_GMM, c_GMM = GMM.GMM_algorithm()

draw_GMM(k, mu_GMM, c_GMM)


iris = read_from_document.read_from_csv()
iris_data = iris.get_data()
GMM_iris = GMM_EM.GMM_EM_model(iris_data, k)
mu_iris, c_iris = GMM_iris.GMM_algorithm()


kmeans_iris = k_means.KMeans(iris_data, 3)
kmeans_mu_iris, kmeans_c_iris = kmeans_iris.k_means_by_maxlength()

print('选取最远初始均值向量GMM的准确率为：', iris.test_accuracy(GMM_iris.lamda))
print('选取最远初始均值向量的k-means的准确率为：', iris.test_accuracy(kmeans_iris.sample_assignments))
