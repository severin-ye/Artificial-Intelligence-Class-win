import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import random

# 手动加载 CIFAR-10 数据集的单个批次
def load_cifar10_batch(batch_id):
    with open(f'./cifar-10-python/cifar-10-batches-py/data_batch_{batch_id}', 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

# 加载 CIFAR-10 数据集的测试集
def load_cifar10_test():
    with open('./cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 加载和预处理数据
def load_and_preprocess_data():
    x_train, y_train = [], []
    for i in range(1, 6):
        features, labels = load_cifar10_batch(i)
        x_train.append(features)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test, y_test = load_cifar10_test()

    x_train = x_train.astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_test = x_test.astype(np.float32) / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_val, _, y_val, _ = train_test_split(x_test, y_test, test_size=0.6, random_state=1)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# 构建和训练模型
def build_and_train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, model_name):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
    model.summary()
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 30))
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_val, y_val), verbose=1, callbacks=[lr_scheduler])

    model.save(f'{model_name}.h5')
    print(f"{model_name} saved as '{model_name}.h5'")
    return model

# 基线模型
def build_baseline_model(input_shape):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
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

# 迁移学习模型
def build_transfer_learning_model(input_shape):
    transfermodel = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    return Sequential([
        transfermodel,
        Flatten(),
        Dense(1000, activation='relu'),
        Dense(10, activation='softmax')
    ])

# 主程序
if __name__ == "__main__":
    reset_random_seeds()
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()
    
    print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
    print("reduced train/val size:", len(x_train), len(x_val), "input shape:", x_train.shape[1:])
    
    epochs = 70
    batch_size = 64
    
    # 基线模型
    baseline_model = build_baseline_model(x_train.shape[1:])
    baseline_model = build_and_train_model(baseline_model, x_train, y_train, x_val, y_val, epochs, batch_size, 'baseline_model')
    baseline_accuracy = baseline_model.evaluate(x_test, y_test, verbose=0)
    print("Baseline model accuracy: ", baseline_accuracy[1] * 100)
    
    reset_random_seeds()
    
    # 迁移学习模型
    transfer_learning_model = build_transfer_learning_model(x_train.shape[1:])
    transfer_learning_model = build_and_train_model(transfer_learning_model, x_train, y_train, x_val, y_val, epochs, batch_size, 'transfer_learning_model')
    transfer_learning_accuracy = transfer_learning_model.evaluate(x_test, y_test, verbose=0)
    print("Transfer learning model accuracy: ", transfer_learning_accuracy[1] * 100)
    
    # 加载和评估保存的模型
    loaded_baseline_model = load_model('baseline_model.h5')
    loaded_baseline_accuracy = loaded_baseline_model.evaluate(x_test, y_test, verbose=0)
    print("Loaded Baseline model accuracy: ", loaded_baseline_accuracy[1] * 100)
    
    loaded_transfer_model = load_model('transfer_learning_model.h5')
    loaded_transfer_accuracy = loaded_transfer_model.evaluate(x_test, y_test, verbose=0)
    print("Loaded Transfer Learning model accuracy: ", loaded_transfer_accuracy[1] * 100)
