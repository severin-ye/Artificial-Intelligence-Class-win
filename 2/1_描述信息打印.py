from sklearn import datasets # 导入数据集模块--datasets
d = datasets.load_iris() # 加载数据集--鸢尾花数据集
print(d.DESCR) # 输出数据集的描述信息
