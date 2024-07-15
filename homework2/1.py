import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
import os
from sklearn.model_selection import train_test_split
import random

# 手动加载 CIFAR-10 数据集
def load_cifar10_batch(batch_id):
    with open(f'./cifar-10-python/cifar-10-batches-py/data_batch_{batch_id}', 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def load_cifar10_test():
    with open('./cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

x_train, y_train = [], []
for i in range(1, 6):
    features, labels = load_cifar10_batch(i)
    x_train.append(features)
    y_train.append(labels)

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

x_test, y_test = load_cifar10_test()

# 数据预处理
x_train = x_train.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

x_test = x_test.astype(np.float32) / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')

# 划分验证集
x_val, _, y_val, _ = train_test_split(x_test, y_test, test_size=0.6, random_state=1)

input_shape = x_train.shape[1:]

# 确保两种模型使用相同的超参数
g_epoch = 70
g_batch = 64

# 重置随机种子函数以确保可重复性
def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

reset_random_seeds() # 调用重置随机种子函数

print("reduced train/val size:", len(x_train), len(x_val), "input shape:", input_shape)

# 定义基线模型
cnn = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(1000, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 编译基线模型
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
cnn.summary()

# 训练基线模型
hist = cnn.fit(x_train, y_train, batch_size=g_batch, epochs=g_epoch, validation_data=(x_val, y_val), verbose=1)

# 保存基线模型
cnn.save('baseline_model.h5')

# 评估基线模型
g_org_res = cnn.evaluate(x_test, y_test, verbose=0)
print("Baseline 정확률은", g_org_res[1] * 100)

# 重置随机种子以确保可重复性
reset_random_seeds()

# 加载ResNet50模型，使用预训练权重
transfermodel = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# 定义迁移学习模型架构
model = Sequential([
    transfermodel,     # 使用预训练的ResNet50模型
    Flatten(),         # 展平层
    Dense(1000, activation='relu'),  # 全连接层
    Dense(10, activation='softmax')  # 输出层
])

# 编译迁移学习模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
model.summary()

# 训练迁移学习模型
hist = model.fit(x_train, y_train, batch_size=g_batch, epochs=g_epoch, validation_data=(x_val, y_val), verbose=1)

# 保存迁移学习模型
model.save('transfer_learning_model.h5')

# 评估迁移学习模型
yours = model.evaluate(x_test, y_test, verbose=0)
print("Baseline vs yours: ", g_org_res[1] * 100, yours[1] * 100)

# 加载并评估保存的基线模型
loaded_baseline_model = load_model('baseline_model.h5')
loaded_baseline_accuracy = loaded_baseline_model.evaluate(x_test, y_test, verbose=0)
print("Loaded Baseline 정확률은", loaded_baseline_accuracy[1] * 100)

# 加载并评估保存的迁移学习模型
loaded_transfer_model = load_model('transfer_learning_model.h5')
loaded_transfer_accuracy = loaded_transfer_model.evaluate(x_test, y_test, verbose=0)
print("Loaded Transfer Learning 정확률은", loaded_transfer_accuracy[1] * 100)
