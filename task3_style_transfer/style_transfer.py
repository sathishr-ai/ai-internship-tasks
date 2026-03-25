# style_transfer.py
# Requirements: pip install tensorflow matplotlib numpy pillow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------------------
# Utility functions
# ---------------------------
def load_and_process_image(image_path, target_size=(512, 512)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(image_array, output_path):
    img = deprocess_image(image_array)
    plt.imsave(output_path, img)

def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    tensor = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(tensor, tensor, transpose_a=True)
    return gram / tf.cast(tf.shape(tensor)[0], tf.float32)

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return tf.reduce_mean(tf.square(S - G))

# ---------------------------
# Main style transfer function
# ---------------------------
def neural_style_transfer(content_path, style_path, output_path,
                          iterations=1000, content_weight=1e3, style_weight=1e-2):
    # Load images
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # Build VGG19 model with intermediate layers
    vgg = vgg19.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False

    # Layers for content and style
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    # Create models that output the activations of these layers
    content_outputs = vgg.get_layer(content_layer).output
    style_outputs = [vgg.get_layer(layer).output for layer in style_layers]
    model_outputs = [content_outputs] + style_outputs
    model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

    # Precompute content and style targets
    content_target = model(content_image)[0]
    style_targets = [model(style_image)[i] for i in range(1, len(style_layers)+1)]

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=2.0)

    # Training loop
    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(generated_image)
            c_loss = content_loss(content_target, outputs[0])
            s_loss = 0
            for j, target in enumerate(style_targets):
                s_loss += style_loss(target, outputs[j+1])
            total_loss = content_weight * c_loss + style_weight * s_loss

        grads = tape.gradient(total_loss, generated_image)
        opt.apply_gradients([(grads, generated_image)])

        # Clip pixel values to maintain valid range
        generated_image.assign(tf.clip_by_value(generated_image, -128.0, 128.0))

        if i % 100 == 0:
            print(f"Iteration {i}: total loss = {total_loss.numpy():.2f}")

    # Save final image
    save_image(generated_image.numpy(), output_path)
    print(f"Styled image saved to {output_path}")

if __name__ == "__main__":
    # Example usage – replace with your own image paths
    neural_style_transfer(
        content_path="content.jpg",
        style_path="style.jpg",
        output_path="stylized_output.jpg",
        iterations=500  # increase for better quality
    )