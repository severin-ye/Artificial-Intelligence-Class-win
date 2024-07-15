from sklearn import datasets
import matplotlib.pyplot as plt
import os
import shutil
import time

# 打印当前时间
print("程序开始时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 删除现有数据集
lfw_dir = os.path.join(os.getenv('USERPROFILE'), 'scikit_learn_data', 'lfw_home')
if os.path.exists(lfw_dir):
    shutil.rmtree(lfw_dir)

# 打印删除数据集结束时间
print("删除数据集结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 重新下载数据集
lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 下载结束时间
print("数据集下载结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 设置绘图区域大小
plt.figure(figsize=(20, 5))

# 显示前8张面部图像
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(lfw.images[i], cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])

# 显示图像
plt.show()

# 打印当下时间
print("程序结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
