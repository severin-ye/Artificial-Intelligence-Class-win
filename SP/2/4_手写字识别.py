from sklearn import datasets
import matplotlib.pyplot as plt # 导入绘图库

# 加载手写数字数据集
digit = datasets.load_digits()

# 设置绘图区域大小
plt.figure(figsize=(5, 5))

# 显示第0个样本的图像
plt.imshow(digit.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

# 显示图像
plt.show()

# 输出第0个样本的像素值
print("像素值：")
print(digit.data[0])

# 输出第0个样本的标签
print("这数字是", digit.target[0], "。")
