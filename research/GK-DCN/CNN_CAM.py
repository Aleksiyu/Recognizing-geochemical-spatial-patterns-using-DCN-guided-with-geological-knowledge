"""
Grad-CAM for CNN
Display the class activation patterns of the first ordinary convolutional layer
"""


import pandas as pd
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def check_data_files():
    required_files = [
        "train_data.npy",
        "train_labels.npy",
        "verify_data.npy",
        "verify_labels.npy",
        "Geochemical_array.npy"
    ]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Warning: Required data file not found:{file}.")
            return False
    return True

if not check_data_files():
    pass


# Load Data
train_data = np.load("train_data.npy")            # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")          # Shape: (M, 9, 9, 39)
verify_labels = np.load("verify_labels.npy")

window_size = 9
all_channel = 39 


os.makedirs('Grad-CAM_CNN', exist_ok=True)

# Model structure construction
model = Sequential(name="CustomModel")
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(window_size, window_size, all_channel), name='conv2d'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2d_1'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name='predictions'))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['acc'])

# Model Training
history = model.fit(
    train_data,
    train_labels,
    epochs=80,
    batch_size=64,
    shuffle=True,
    validation_data=(verify_data, verify_labels),
    verbose=2)
model.save('model_cnn.h5')


# ----------------------------------------------------
# Grad-CAM Implementation (Keras)
# ----------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    The core function for generating Grad-CAM heatmaps
    Args:
        img_array: Input image array, shape (1, H, W, C)
        model: Trained Keras model
        last_conv_layer_name: Name of the target convolutional layer
        pred_index: Target category index
    """

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute Gradient
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        # Forward propagation
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            target_class_output = preds[:, 0]
        else:
            target_class_output = preds[:, pred_index]

    # 3. Compute the gradient of the target class output with respect to the convolutional layer output
    grads = tape.gradient(target_class_output, last_conv_layer_output)

    # 4. Global Average Pooling Gradient (Calculating Weight alpha_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weighted Sum
    last_conv_layer_output = last_conv_layer_output[0] # shape (H, W, C)

    # (H, W, C) * (C,) ->  (H, W, C)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] 
    heatmap = tf.squeeze(heatmap) # shape (H, W)

    # 6. ReLU 
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()

def process_sample_gradcam(model, sample_data_with_batch, sample_name, last_conv_layer_name):
    # Process a single sample, generate Grad-CAM, and save it
    print(f"Processing sample:{sample_name}")
    input_tensor = tf.convert_to_tensor(sample_data_with_batch, dtype=tf.float32)

    # Generate Grad-CAM heatmap
    cam = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name)
    input_data_np = sample_data_with_batch[0] # (H, W, C)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Average of 39 channels
    combined_data = np.mean(input_data_np, axis=2)
    combined_min = combined_data.min()
    combined_max = combined_data.max()
    if combined_max > combined_min:
        combined_data_final = (combined_data - combined_min) / (combined_max - combined_min)
    else:
        combined_data_final = np.zeros_like(combined_data)

    ax1 = axes[0]
    im1 = ax1.imshow(combined_data_final, cmap='viridis')
    ax1.set_title(f'{sample_name} - Input Data Mean')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Normalized Value')

    ax2 = axes[1]
    im2 = ax2.imshow(cam, cmap='jet', interpolation='nearest') 
    ax2.set_title(f'{sample_name} - Keras Grad-CAM')
    ax2.set_xticks(np.arange(-.5, cam.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, cam.shape[0], 1), minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax2.tick_params(which='minor', size=0)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Contribution Score')

    plt.tight_layout()
    # Save result image
    OUTPUT_DIR = 'Grad-CAM_CNN'
    output_path = f'{OUTPUT_DIR}/{sample_name}_gradcam.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" Saved: {output_path}")


# Generate Grad-CAM for all known mining sites
model = tf.keras.models.load_model('model_cnn.h5')
# Determine the target convolutional layer
TARGET_CONV_LAYER_NAME = model.layers[0].name 

print("Start generating Grad-CAM for all known mining points (using center point coordinates)...")

try:
    all_array = np.load('Geochemical_array.npy')
    # Load label images (including mineral point information)
    label_img = cv.imread('./New_data/label/Au_deposits0.tif', cv.IMREAD_GRAYSCALE) 
    if label_img is None:
        label_img = cv.imread('Au_deposits0.tif', cv.IMREAD_GRAYSCALE) 
        if label_img is None:
             raise FileNotFoundError("Unable to read label image Au_deposits0.tif")
    print(f"Label image Au_deposits0.tif loaded successfully, shape:{label_img.shape}")
except (FileNotFoundError, Exception) as e:
    print(f"Error: Unable to load data file {e}")
    exit()

required_height = all_array.shape[0] - (window_size - 1)
required_width = all_array.shape[1] - (window_size - 1)

pad_h = (label_img.shape[0] - required_height) // 2
pad_w = (label_img.shape[1] - required_width) // 2

label_img_cropped = label_img[pad_h:pad_h+required_height, pad_w:pad_w+required_width]
print(f"Label image cropped size: {label_img_cropped.shape}")

mining_point_rows_cropped, mining_point_cols_cropped = np.where(label_img_cropped == 1)
num_points = len(mining_point_rows_cropped)
print(f"Found {num_points} mining points.")

if num_points == 0:
    print("Warning: No mining points found in the label image (pixels with value 1).")
else:
    generated_count = 0
    half_window = window_size // 2
    # Traverse all the discovered mining sites
    for i in range(num_points):
        r_top_left = mining_point_rows_cropped[i]
        c_top_left = mining_point_cols_cropped[i]

        if (r_top_left >= 0 and r_top_left <= all_array.shape[0] - window_size and
            c_top_left >= 0 and c_top_left <= all_array.shape[1] - window_size):
            
            # Extract window data (H, W, C)
            sample_data = all_array[r_top_left:r_top_left+window_size, c_top_left:c_top_left+window_size, :]
            # Add batch dimension (1, H, W, C)
            sample_data_with_batch = np.expand_dims(sample_data, axis=0)
            
            # Calculate the center point coordinates
            center_r = r_top_left + half_window
            center_c = c_top_left + half_window
            point_name = f"deposit_point_({center_r},{center_c})" 
            
            process_sample_gradcam(model, sample_data_with_batch, point_name, TARGET_CONV_LAYER_NAME)
            generated_count += 1
        else:
            print(f"Warning: ({r_top_left}, {c_top_left})")

    print(f"Completed {generated_count}/{num_points} mining point Grad-CAM generation.")


print("All mining point Grad-CAM generation completed.")
