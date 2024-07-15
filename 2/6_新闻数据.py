from sklearn.datasets import fetch_20newsgroups

# 读取 20newsgroups 数据集的训练集
news = fetch_20newsgroups(subset='train')

# 打印第0个样本的内容
print("******\n", news.data[0], "\n******")

# 打印第0个样本的类别
print("这文档的类别是 <", news.target_names[news.target[0]], ">。")
