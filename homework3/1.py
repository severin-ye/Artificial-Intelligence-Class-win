import os
import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as K
from PIL import Image

# 设置 Keras 后端为 TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# 加载图像路径
base_image_path = "./homework3/1.jpg"  # 原图像路径
style_reference_image_path = './homework3/2.jpg' # 风格参考图像路径
style_reference_image = Image.open(style_reference_image_path) # 加载风格参考图像
result_prefix = "generated_result"

# 定义不同损失组件的权重
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# 获取基础图像的尺寸，并设定生成图像的尺寸
width, height = load_img(base_image_path).size  # 获取基础图像的宽度和高度
img_nrows = 400  # 设定生成图像的高度
img_ncols = int(width * img_nrows / height)  # 计算生成图像的宽度，以保持比例一致

# 图像预处理函数，将图像转换为模型输入的格式
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))  # 加载并调整图像尺寸
    img = img_to_array(img)  # 将图像转换为数组
    img = np.expand_dims(img, axis=0)  # 增加一个维度，形成 (1, 高, 宽, 通道) 的形状
    img = vgg19.preprocess_input(img)  # 对图像进行预处理，使其符合 VGG19 的输入要求
    return tf.convert_to_tensor(img)  # 转换为 TensorFlow 张量

# 图像后处理函数，将模型输出的张量转换为可视化的图像
def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))  # 重新调整张量形状为 (高, 宽, 通道)
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
    size = img_nrows * img_ncols  # 图像的尺寸
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))  # 计算 L2 损失

# 计算内容损失，使生成图像的高层特征接近基础图像
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))  # 计算 L2 损失

# 计算总变差损失，保持生成图像的局部一致性
def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )  # 计算相邻像素点在 x 方向的差值平方
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
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
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )  # 计算内容损失并加到总损失中

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]  # 风格图像的特征
        combination_features = layer_features[2, :, :, :]  # 生成图像的特征
        sl = style_loss(style_reference_features, combination_features)  # 计算风格损失
        loss += (style_weight / len(style_layer_names)) * sl  # 平均风格损失并加到总损失中

    loss += total_variation_weight * total_variation_loss(combination_image)  # 加入总变差损失
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
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

# 预处理图像
base_image = preprocess_image(base_image_path)  # 预处理基础图像
style_reference_image = preprocess_image(style_reference_image_path)  # 预处理风格参考图像
combination_image = tf.Variable(preprocess_image(base_image_path))  # 初始化生成图像

# 进行4000次迭代，每100次迭代保存一次图像
iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )  # 计算损失和梯度
    optimizer.apply_gradients([(grads, combination_image)])  # 应用梯度更新生成图像
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))  # 打印当前迭代次数和损失值
        img = deprocess_image(combination_image.numpy())  # 后处理生成的图像
        fname = result_prefix + "_at_iteration_%d.png" % i  # 生成保存文件名
        save_img(fname, img)  # 保存图像文件
