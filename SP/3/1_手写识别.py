from sklearn import datasets
from sklearn import svm

# 加载手写数字数据集
digit = datasets.load_digits()

# 创建SVM分类器
s = svm.SVC(gamma=0.1, C=10)

# 训练模型
s.fit(digit.data, digit.target)

# 选择前面的三个样本作为新的测试集
new_d = [digit.data[0], digit.data[1], digit.data[2]]
res = s.predict(new_d)

# 打印预测结果
print("预测值:", res)
print("真实值:", digit.target[0], digit.target[1], digit.target[2])

# 使用训练集作为测试集进行预测并计算准确率
res = s.predict(digit.data) # 预测
##############################################
correct = [i for i in range(len(res)) if res[i] == digit.target[i]] # 生成正确的索引的列表
# correct = []
# for i in range(len(res)):
#     if res[i] == digit.target[i]:
#         correct.append(i)
##############################################
accuracy = len(correct) / len(res)
print("使用像素特征时的准确率:", accuracy * 100, "%")
