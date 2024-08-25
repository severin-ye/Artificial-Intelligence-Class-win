from sklearn import datasets  # 导入 sklearn 的数据集模块
from sklearn import svm  # 导入 sklearn 的支持向量机模块

d = datasets.load_iris()  # 加载 iris 数据集--名为 d

# 创建 SVM 分类器--名为 s
s = svm.SVC(gamma=0.1, C=10)  # gamma 和 C 是 SVM 的超参数 --用来指定复杂度和惩罚系数

s.fit(d.data, d.target)  # 使用数据集的特征和目标进行模型训练

# 创建新的测试数据
new_d = [[6.4, 3.2, 6.0, 2.5], [7.1, 3.1, 4.7, 1.35]]  # 两个新的样本

# 用训练好的模型进行预测
res = s.predict(new_d) # predict() 方法用来预测新的数据

# 输出预测结果
print("新的2个样本的类别是:", res)  # 打印预测结果
