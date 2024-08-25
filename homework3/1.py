import os
import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array, save_img
from keras import backend as K
from PIL import Image
import time

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Hyperparameter configuration
config_1 = {
    'base_image_path': "./homework3/7.jpg",  # Path to the base image
    'style_reference_image_path': './homework3/7.jpeg',  # Path to the style reference image
    'result_prefix': "generated_result",  # Prefix for the result
    'total_variation_weight': 1e-6,  # Total variation weight
    'style_weight': 1e-6,  # Style loss weight
    'content_weight': 2.5e-8,  # Content loss weight
    'img_nrows': 400,  # Height of the generated image
    'iterations': 4000,  # Number of iterations
    'learning_rate': 100.0,  # Initial learning rate
    'decay_steps': 100,  # Learning rate decay steps
    'decay_rate': 0.96  # Learning rate decay rate
}

config_3= {
    'base_image_path': "./homework3/7.jpg",  # Path to the base image
    'style_reference_image_path': './homework3/7.jpeg',  # Path to the style reference image
    'result_prefix': "generated_result",  # Prefix for the result
    'total_variation_weight': 1e-6,  # Total variation weight
    'style_weight': 5e-5,  # Style loss weight
    'content_weight': 5e-5,  # Content loss weight
    'img_nrows': 400,  # Height of the generated image
    'iterations': 4000,  # Number of iterations
    'learning_rate': 50.0,  # Initial learning rate
    'decay_steps': 100,  # Learning rate decay steps
    'decay_rate': 0.96  # Learning rate decay rate
}

# Get the dimensions of the base image and set the dimensions for the generated image
width, height = load_img(config_3['base_image_path']).size  # Get the width and height of the base image
config_3['img_ncols'] = int(width * config_3['img_nrows'] / height)  # Calculate the width of the generated image to maintain aspect ratio

# Image preprocessing function, converts the image to a format suitable for the model input
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(config_3['img_nrows'], config_3['img_ncols']))  # Load and resize the image
    img = img_to_array(img)  # Convert the image to an array
    img = np.expand_dims(img, axis=0)  # Add a dimension to create a shape of (1, height, width, channels)
    img = vgg19.preprocess_input(img)  # Preprocess the image to match VGG19 input requirements
    return tf.convert_to_tensor(img)  # Convert to TensorFlow tensor

# Image post-processing function, converts the model output tensor back to a visualizable image
def deprocess_image(x):
    x = x.numpy()
    x = x.reshape((config_3['img_nrows'], config_3['img_ncols'], 3))  # Reshape the tensor to (height, width, channels)
    x[:, :, 0] += 103.939  # Restore the mean subtracted during preprocessing
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert the image from 'BGR' to 'RGB'
    x = np.clip(x, 0, 255).astype("uint8")  # Clip pixel values to the range 0-255 and convert to unsigned integers
    return x

# Calculate the Gram matrix, used for style loss
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))  # Transpose the tensor so that the channel dimension is first
    features = tf.reshape(x, (tf.shape(x)[0], -1))  # Reshape the tensor to (channels, width*height)
    gram = tf.matmul(features, tf.transpose(features))  # Compute the outer product of the feature map
    return gram

# Calculate style loss, ensuring the generated image matches the style of the reference image
def style_loss(style, combination):
    S = gram_matrix(style)  # Gram matrix of the style image
    C = gram_matrix(combination)  # Gram matrix of the generated image
    channels = 3  # Number of channels in the image
    size = config_3['img_nrows'] * config_3['img_ncols']  # Size of the image
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))  # Compute L2 loss

# Calculate content loss, ensuring the generated image matches the high-level features of the base image
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))  # Compute L2 loss

# Calculate total variation loss, maintaining the local consistency of the generated image
def total_variation_loss(x):
    a = tf.square(
        x[:, : config_3['img_nrows'] - 1, : config_3['img_ncols'] - 1, :] - x[:, 1:, : config_3['img_ncols'] - 1, :]
    )  # Compute the squared difference between neighboring pixels in the x direction
    b = tf.square(
        x[:, : config_3['img_nrows'] - 1, : config_3['img_ncols'] - 1, :] - x[:, : config_3['img_nrows'] - 1, 1:, :]
    )  # Compute the squared difference between neighboring pixels in the y direction
    return tf.reduce_sum(tf.pow(a + b, 1.25))  # Compute total variation loss

# Build the VGG19 model, loading pretrained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)  # Exclude the top fully connected layers

# Extract the output of each layer and create a feature extractor model
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])  # Create a dictionary mapping layer names to outputs
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)  # Create the feature extractor model

# Define the layers used to calculate style loss
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# Define the layer used to calculate content loss
content_layer_name = "block5_conv2"

# Compute the total loss, including content loss, style loss, and total variation loss
def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )  # Concatenate the base image, style reference image, and generated image
    features = feature_extractor(input_tensor)  # Extract features

    loss = tf.zeros(shape=())  # Initialize the loss

    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]  # Features of the base image
    combination_features = layer_features[2, :, :, :]  # Features of the generated image
    loss = loss + config_3['content_weight'] * content_loss(
        base_image_features, combination_features
    )  # Compute content loss and add to total loss

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]  # Features of the style image
        combination_features = layer_features[2, :, :, :]  # Features of the generated image
        sl = style_loss(style_reference_features, combination_features)  # Compute style loss
        loss += (config_3['style_weight'] / len(style_layer_names)) * sl  # Average style loss and add to total loss

    loss += config_3['total_variation_weight'] * total_variation_loss(combination_image)  # Add total variation loss
    return loss

# Use the tf.function decorator to speed up the computation of loss and gradients
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)  # Compute total loss
    grads = tape.gradient(loss, combination_image)  # Compute gradients with respect to the generated image
    return loss, grads

# Set up the optimizer using SGD with exponential decay
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config_3['learning_rate'], decay_steps=config_3['decay_steps'], decay_rate=config_3['decay_rate']
    )
)

# Preprocess images
base_image = preprocess_image(config_3['base_image_path'])  # Preprocess the base image
style_reference_image = preprocess_image(config_3['style_reference_image_path'])  # Preprocess the style reference image
combination_image = tf.Variable(preprocess_image(config_3['base_image_path']))  # Initialize the generated image

# Run 4000 iterations, saving the image every 100 iterations
iterations = config_3['iterations']
start_time = time.time()  # Start timing

# Early stopping counter
loss_small_10_counter = 0
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )  # Compute loss and gradients
    
    # Early stopping condition
    if loss <= 10:
        loss_small_10_counter += 1
    if loss_small_10_counter >= 3:
        break

    # Apply gradient clipping
    grads = tf.clip_by_value(grads, -1.0, 1.0)
    optimizer.apply_gradients([(grads, combination_image)])  # Apply gradient updates to the generated image
    
    if i % 100 == 0:
        end_time = time.time()  # End timing
        print("Iteration %d: loss=%.2f, time=%.2fs" % (i, loss, end_time - start_time))  # Print current iteration, loss, and time
        start_time = time.time()  # Reset start time
        img = deprocess_image(combination_image)  # Post-process the generated image
        
        # Assume you want to save the image to the 'output_images' folder
        save_directory = 'output_images'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  # Create the folder if it doesn't exist
        # Generate the save file name with path
        fname = os.path.join(save_directory, config_3['result_prefix'] + "_at_iteration_%d.png" % i)
        # Save the image file
        save_img(fname, img)
