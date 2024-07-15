from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

# 加载手写数字数据集
digit = datasets.load_digits()

# 将数据集随机分割为训练集和测试集，训练集占60%，测试集占40%
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)

# 创建SVM分类器
s = svm.SVC(gamma=0.001)

# 训练模型
s.fit(x_train, y_train)

# 在测试集上进行预测
res = s.predict(x_test)

# 计算混淆矩阵
conf = np.zeros((10, 10)) # 10*10的全零矩阵->混淆矩阵
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1 # 预测值为res[i]，真实值为y_test[i]->混淆矩阵中的计数+1

# 打印混淆矩阵
print(conf)

# 计算并打印准确率
no_correct = 0
for i in range(10):
    no_correct += conf[i][i] 
accuracy = no_correct / len(res)
print("测试集上的准确率是", accuracy * 100, "%")
