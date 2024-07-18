import os
import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as K
from PIL import Image
import time

# 设置 Keras 后端为 TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# 超参数配置
config_1 = {
    'base_image_path': "./homework3/7.jpg",  # 原图像路径
    'style_reference_image_path': './homework3/7.jpeg',  # 风格参考图像路径
    'result_prefix': "generated_result",  # 结果前缀
    'total_variation_weight': 1e-6,  # 总变差权重
    'style_weight': 1e-6,  # 风格损失权重
    'content_weight': 2.5e-8,  # 内容损失权重
    'img_nrows': 400,  # 生成图像的高度
    'iterations': 4000,  # 迭代次数
    'learning_rate': 100.0,  # 初始学习率
    'decay_steps': 100,  # 学习率衰减步数
    'decay_rate': 0.96  # 学习率衰减率
}

# config_2 = {
#     'base_image_path': "./homework3/7.jpg",  # 原图像路径
#     'style_reference_image_path': './shuimo.jpeg',  # 风格参考图像路径
#     'result_prefix': "generated_result",  # 结果前缀
#     'total_variation_weight': 1e-6,  # 总变差权重
#     'style_weight': 1e-4,  # 风格损失权重
#     'content_weight': 1e-4,  # 内容损失权重
#     'img_nrows': 400,  # 生成图像的高度
#     'iterations': 4000,  # 迭代次数
#     'learning_rate': 0.01,  # 初始学习率
#     'decay_steps': 100,  # 学习率衰减步数
#     'decay_rate': 0.96  # 学习率衰减率
# }

config_3= {
    'base_image_path': "./homework3/7.jpg",  # 原图像路径
    'style_reference_image_path': './homework3/7.jpeg',  # 风格参考图像路径
    'result_prefix': "generated_result",  # 结果前缀
    'total_variation_weight': 1e-6,  # 总变差权重
    'style_weight': 5e-5,  # 风格损失权重
    'content_weight': 5e-5,  # 内容损失权重
    'img_nrows': 400,  # 生成图像的高度
    'iterations': 4000,  # 迭代次数
    'learning_rate': 50.0,  # 初始学习率
    'decay_steps': 100,  # 学习率衰减步数
    'decay_rate': 0.96  # 学习率衰减率
}


# 获取基础图像的尺寸，并设定生成图像的尺寸
width, height = load_img(config_3['base_image_path']).size  # 获取基础图像的宽度和高度
config_3['img_ncols'] = int(width * config_3['img_nrows'] / height)  # 计算生成图像的宽度，以保持比例一致

# 图像预处理函数，将图像转换为模型输入的格式
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(config_3['img_nrows'], config_3['img_ncols']))  # 加载并调整图像尺寸
    img = img_to_array(img)  # 将图像转换为数组
    img = np.expand_dims(img, axis=0)  # 增加一个维度，形成 (1, 高, 宽, 通道) 的形状
    img = vgg19.preprocess_input(img)  # 对图像进行预处理，使其符合 VGG19 的输入要求
    return tf.convert_to_tensor(img)  # 转换为 TensorFlow 张量

# 图像后处理函数，将模型输出的张量转换为可视化的图像
def deprocess_image(x):
    x = x.numpy()
    x = x.reshape((config_3['img_nrows'], config_3['img_ncols'], 3))  # 重新调整张量形状为 (高, 宽, 通道)
    x[:, :, 0] += 103.939  # 还原预处理时减去的均值
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # 将图像从 'BGR' 转换为 'RGB'
    x = np.clip(x, 0, 255).astype("uint8")  # 限制像素值在 0-255 之间，并转换为无符号整型
    return x

# 计算 Gram 矩阵，用于风格损失
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))  # 转置张量，使通道维度在前
    features = tf.reshape(x, (tf.shape(x)[0], -1))  # 将张量重新形状化为 (通道数, 宽*高)
    gram = tf.matmul(features, tf.transpose(features))  # 计算特征图的外积
    return gram

# 计算风格损失，使生成图像的风格接近风格参考图像
def style_loss(style, combination):
    S = gram_matrix(style)  # 风格图像的 Gram 矩阵
    C = gram_matrix(combination)  # 生成图像的 Gram 矩阵
    channels = 3  # 图像的通道数
    size = config_3['img_nrows'] * config_3['img_ncols']  # 图像的尺寸
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))  # 计算 L2 损失

# 计算内容损失，使生成图像的高层特征接近基础图像
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))  # 计算 L2 损失

# 计算总变差损失，保持生成图像的局部一致性
def total_variation_loss(x):
    a = tf.square(
        x[:, : config_3['img_nrows'] - 1, : config_3['img_ncols'] - 1, :] - x[:, 1:, : config_3['img_ncols'] - 1, :]
    )  # 计算相邻像素点在 x 方向的差值平方
    b = tf.square(
        x[:, : config_3['img_nrows'] - 1, : config_3['img_ncols'] - 1, :] - x[:, : config_3['img_nrows'] - 1, 1:, :]
    )  # 计算相邻像素点在 y 方向的差值平方
    return tf.reduce_sum(tf.pow(a + b, 1.25))  # 计算总变差损失

# 构建 VGG19 模型，加载预训练的 ImageNet 权重
model = vgg19.VGG19(weights="imagenet", include_top=False)  # 不包含顶层的全连接层

# 获取每一层的输出，并构建一个特征提取器模型
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])  # 创建层名字到输出张量的字典
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)  # 创建特征提取模型

# 定义用于计算风格损失的层
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# 定义用于计算内容损失的层
content_layer_name = "block5_conv2"

# 计算总损失，包括内容损失、风格损失和总变差损失
def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )  # 将基础图像、风格参考图像和生成图像拼接在一起
    features = feature_extractor(input_tensor)  # 提取特征

    loss = tf.zeros(shape=())  # 初始化损失值

    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]  # 基础图像的特征
    combination_features = layer_features[2, :, :, :]  # 生成图像的特征
    loss = loss + config_3['content_weight'] * content_loss(
        base_image_features, combination_features
    )  # 计算内容损失并加到总损失中

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]  # 风格图像的特征
        combination_features = layer_features[2, :, :, :]  # 生成图像的特征
        sl = style_loss(style_reference_features, combination_features)  # 计算风格损失
        loss += (config_3['style_weight'] / len(style_layer_names)) * sl  # 平均风格损失并加到总损失中

    loss += config_3['total_variation_weight'] * total_variation_loss(combination_image)  # 加入总变差损失
    return loss

# 使用 tf.function 装饰器加速损失和梯度的计算
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)  # 计算总损失
    grads = tape.gradient(loss, combination_image)  # 计算生成图像相对于损失的梯度
    return loss, grads

# 设置优化器，使用带有指数衰减的 SGD 优化器
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config_3['learning_rate'], decay_steps=config_3['decay_steps'], decay_rate=config_3['decay_rate']
    )
)

# 预处理图像
base_image = preprocess_image(config_3['base_image_path'])  # 预处理基础图像
style_reference_image = preprocess_image(config_3['style_reference_image_path'])  # 预处理风格参考图像
combination_image = tf.Variable(preprocess_image(config_3['base_image_path']))  # 初始化生成图像

# 进行4000次迭代，每100次迭代保存一次图像
iterations = config_3['iterations']
start_time = time.time()  # 开始计时

# 提前停止计数器
loss_small_10_counter = 0
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )  # 计算损失和梯度
    
    # 提前停止条件
    if loss <= 10:
        loss_small_10_counter += 1
    if loss_small_10_counter >= 3:
        break

    # 应用梯度裁剪
    grads = tf.clip_by_value(grads, -1.0, 1.0)
    optimizer.apply_gradients([(grads, combination_image)])  # 应用梯度更新生成图像
    
    if i % 100 == 0:
        end_time = time.time()  # 结束计时
        print("Iteration %d: loss=%.2f, time=%.2fs" % (i, loss, end_time - start_time))  # 打印当前迭代次数、损失值和时间
        start_time = time.time()  # 重置开始时间
        img = deprocess_image(combination_image)  # 后处理生成的图像
        
        # 假设你想将图像保存到 'output_images' 文件夹中
        save_directory = 'output_images'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  # 如果文件夹不存在，则创建
        # 生成保存文件名，包含路径
        fname = os.path.join(save_directory, config_3['result_prefix'] + "_at_iteration_%d.png" % i)
        # 保存图像文件
        save_img(fname, img)
