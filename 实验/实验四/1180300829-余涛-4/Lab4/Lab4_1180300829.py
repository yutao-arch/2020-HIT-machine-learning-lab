from Lab4.all_operation import create_data_by_two_or_three_dimension, read_from_file, PCA
from Lab4.draw import draw_picture_by_image, draw_picture_by_create_PCA

# 用于生成数据的测试
dimension = 3
data_num = 50
x = create_data_by_two_or_three_dimension(dimension, data_num)
center_data, w, mu_x = PCA(x, dimension - 1)
x_after_PCA = (x - mu_x).dot(w).dot(w.T) + mu_x
print("中心化后的数据集为为:\n", center_data)
print("特征向量矩阵为:\n", w)
print("降维前样本均值为:\n", mu_x)
draw_picture_by_create_PCA(dimension, x, x_after_PCA)

# 用人脸图像进行测试
x = read_from_file('face_collection')
x_num, x_dimension = x.shape  # 数据个数x_num和维度x_dimension
center_data, w, mu_x = PCA(x, 1)  # PCA降维
print("中心化后的数据集为为:\n", center_data)
print("特征向量矩阵为:\n", w)
print("降维前样本均值为:\n", mu_x)
draw_picture_by_image(x, w, center_data, mu_x, x_num)
