from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

# 加载手写数字数据集
digits = datasets.load_digits()

# 创建SVM分类器
model = svm.SVC(gamma=0.001)

# 进行5折交叉验证
scores = cross_val_score(model, digits.data, digits.target, cv=5)

print("每次交叉验证的准确率: ", scores)
print("平均准确率: ", scores.mean())
print("准确率的标准差: ", scores.std())
