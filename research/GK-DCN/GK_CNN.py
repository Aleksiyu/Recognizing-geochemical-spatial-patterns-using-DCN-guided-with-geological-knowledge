"""
In the loss function of the convolutional neural network, geological constraints are added as soft constraints.
"""


import pandas as pd
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import numpy as np
import cv2 as cv
import math
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, f1_score
)
from tensorflow.keras import backend as K


# Read data
train_data = np.load("train_data.npy")          # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")        # Shape: (M, 9, 9, 39)
verify_labels = np.load("verify_labels.npy")


# =============================================================================
# Load geological constraint weights
# =============================================================================

geology_weights_path = "geology_weights.tif"
if not os.path.exists(geology_weights_path):
    raise FileNotFoundError(f"Cannot find: {geology_weights_path}")

# Read raw data type
geology_weights_full = cv.imread(geology_weights_path, cv.IMREAD_UNCHANGED)
if geology_weights_full is None:
    raise ValueError(f"Unable to read: {geology_weights_path}")

geology_weights_full = geology_weights_full.astype(np.float32)

# Normalization
w_min = geology_weights_full.min()
w_max = geology_weights_full.max()
if w_max > w_min:
    geology_weights_full = (geology_weights_full - w_min) / (w_max - w_min)
else:
    geology_weights_full = np.zeros_like(geology_weights_full, dtype=np.float32)
    print("Warning: Geological weight error.")

print(f"Geological weight map shape: {geology_weights_full.shape}")


window_size = 9
all_channel = 39

# Spatial coordinates for preparing training data
try:
    geo_array_shape = np.load('Geochemical_array.npy').shape
    H_ori, W_ori = geo_array_shape[0], geo_array_shape[1]
    print(f"Size of the primitive geochemical data array: {H_ori} x {W_ori}")
except FileNotFoundError:
    raise FileNotFoundError("Cannot find 'Geochemical_array.npy'.")

H_data, W_data = H_ori - window_size + 1, W_ori - window_size + 1
if train_data.shape[0] > H_data * W_data:
     raise ValueError("wrong data.")

# Generate the center coordinates (row, column) corresponding to the training data
train_indices = np.arange(train_data.shape[0])
train_rows = train_indices // W_data
train_cols = train_indices % W_data

# Calculate the center point coordinates
center_offset = window_size // 2
train_center_rows = train_rows + center_offset
train_center_cols = train_cols + center_offset

# Ensure the coordinates are within bounds
train_center_rows = np.clip(train_center_rows, 0, geology_weights_full.shape[0] - 1)
train_center_cols = np.clip(train_center_cols, 0, geology_weights_full.shape[1] - 1)

# Extract the corresponding geological weights
train_geo_weights_np = geology_weights_full[train_center_rows, train_center_cols]
print(f"Shape of training sample geological weight array: {train_geo_weights_np.shape}")


# =============================================================================
# Define a custom loss function with geological constraints
# =============================================================================

@tf.keras.utils.register_keras_serializable(package='Custom', name='geo_weighted_bce_loss')
def geological_weighted_bce_loss(y_true, y_pred, sample_weight=None):
    """
    BCE Loss Function with Geological Weighting
    Args:
        y_true: True label
        y_pred: Predicted probability
        sample_weight: Sample Weight (from tf.data.Dataset)
    Returns:
        Weighted Average BCE Loss
    """
    # Ensure the input is a float
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate the standard BCE Loss for each sample
    bce = K.binary_crossentropy(y_true, y_pred)

    # --- Incorporate geological constraints --- 

    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        weighted_bce = bce * sample_weight
        # Return the average weighted loss
        return K.mean(weighted_bce)
    else:
        return K.mean(bce)


# --- Create a TensorFlow Dataset with weights ---
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_data, train_labels, train_geo_weights_np)
)
verify_dataset = tf.data.Dataset.from_tensor_slices(
    (verify_data, verify_labels)
)

BATCH_SIZE = 64

# Shuffling and batch processing of the training set
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).repeat()
verify_dataset = verify_dataset.batch(BATCH_SIZE).repeat()


# Model structure construction
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(window_size, window_size, all_channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

geo_weight_coeff = 0.1

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=geological_weighted_bce_loss, 
    metrics=['acc']
)

steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(verify_data) // BATCH_SIZE

# Check
if steps_per_epoch == 0:
    steps_per_epoch = 1
    print("Warning: The training data size is smaller than the batch size, so steps_per_epoch has been set to 1.")
if validation_steps == 0:
    validation_steps = 1
    print("Warning: The amount of validation data is smaller than the batch size, so validation_steps is set to 1.")

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")


history = model.fit(
    train_dataset,
    epochs=80,
    steps_per_epoch=steps_per_epoch,
    validation_data=verify_dataset,
    validation_steps=validation_steps,
    verbose=2
)
model.save('model_GK_CNN.h5')


def safe_plot(epochs, values, label, title, ylabel):
    """Safe drawing function to prevent errors caused by missing keys"""
    if values is not None and len(values) > 0:
        plt.plot(epochs, values, label=label)
    else:
        print(f"Warning: '{label.split()[-1]}' data not found in history.history.")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

# Training process output
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
safe_plot(history.epoch, history.history.get('acc'), 'Training set', 'Model Accuracy', 'Accuracy')
safe_plot(history.epoch, history.history.get('val_acc'), 'Validation set', 'Model Accuracy', 'Accuracy')
plt.grid(True)
plt.subplot(1, 2, 2)
safe_plot(history.epoch, history.history.get('loss'), 'Training set', 'Model Loss', 'Loss')
safe_plot(history.epoch, history.history.get('val_loss'), 'Validation set', 'Model Loss', 'Loss')
plt.grid(True)
plt.tight_layout()
plt.show()


try:
    model = tf.keras.models.load_model('model_GK_CNN.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model loading failed: {e}")
    # If loading fails, redefine the model and compile it (as a fallback, usually not necessary)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(window_size, window_size, all_channel)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=geological_weighted_bce_loss, 
        metrics=['acc']
    )
    print("The model structure has been redefined.")



probability_value = []

# Read the prediction dataset
all_array = np.load('Geochemical_array.npy')

# Model Prediction
for row in range(all_array.shape[0] - (window_size - 1)):
    for col in range(all_array.shape[1] - (window_size - 1)):
        value = all_array[row:row + window_size, col:col + window_size, :]
        value = value.reshape(1, window_size, window_size, all_channel)
        output = model.predict(value, verbose=0)
        probability_value.append(output[0][0])

XX = cv.imread('./New_data/coordinate/XX.tif', cv.IMREAD_UNCHANGED)
YY = cv.imread('./New_data/coordinate/YY.tif', cv.IMREAD_UNCHANGED)

if XX is None or YY is None:
    raise FileNotFoundError("Cannot find the coordinate file './New_data/coordinate/XX.tif' or './New_data/coordinate/YY.tif'")

XX_result = XX[math.floor(window_size/2):XX.shape[0]-math.floor(window_size/2),
              math.floor(window_size/2):XX.shape[1]-math.floor(window_size/2)]
YY_result = YY[math.floor(window_size/2):YY.shape[0]-math.floor(window_size/2),
              math.floor(window_size/2):YY.shape[1]-math.floor(window_size/2)]
XX_result_list = XX_result.flatten()
YY_result_list = YY_result.flatten()
dataframe = pd.DataFrame({'XX': XX_result_list, 'YY': YY_result_list, 'probability': probability_value})
dataframe.to_csv('GK_cnn_output.csv')


# =============================================================================
# Model Evaluation
# =============================================================================

# 1. Read and process the real labels
try:
    label_img = cv.imread('./New_data/label/Au_deposits0.tif', cv.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError("Unable to read the file './New_data/label/Au_deposits0.tif'")
    
    label_cropped = label_img[
        math.floor(window_size/2):label_img.shape[0]-math.floor(window_size/2),
        math.floor(window_size/2):label_img.shape[1]-math.floor(window_size/2)
    ]
    y_true = label_cropped.flatten()
    print(f"Number of real labels: {len(y_true)}")

    unique_labels = np.unique(y_true)
    print(f"Unique values in the original labels: {unique_labels}")
    
    if len(unique_labels) > 2:
        print("Warning: The labels do not appear to be binary. Attempting to convert to 0/1.")
        y_true_binary = ((y_true == unique_labels.max()) | (y_true > 0)).astype(int)
    elif 255 in unique_labels:
         y_true_binary = (y_true == 255).astype(int)
    else:
         y_true_binary = y_true.astype(int)

    valid_indices = (y_true_binary == 0) | (y_true_binary == 1)
    y_true_final = y_true_binary[valid_indices]
    y_prob_final = np.array(probability_value)[valid_indices]
    if len(y_true_final) != len(y_prob_final):
        raise ValueError(f"The number of processed labels ({len(y_true_final)}) does not match the number of probabilities ({len(y_prob_final)}).")

    print(f"Number of valid samples used for evaluation: {len(y_true_final)}")

except (FileNotFoundError, ValueError) as e:
    print(f"Error occurred during evaluation preparation: {e}")
    raise

# 2. Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_final, y_prob_final)
roc_auc = auc(fpr, tpr)
print(f"\nAUC (Area Under Curve): {roc_auc:.4f}")

# 3. Find the optimal threshold
youden_j = tpr - fpr
ix_optimal = np.argmax(youden_j)
optimal_threshold = thresholds[ix_optimal]
print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
print(f"  - TPR at Optimal Threshold: {tpr[ix_optimal]:.4f}")
print(f"  - FPR at Optimal Threshold: {fpr[ix_optimal]:.4f}")
print(f"  - Youden's J at Optimal Threshold: {youden_j[ix_optimal]:.4f}")

# 4. Perform classification prediction using the optimal threshold
y_pred_optimal = (y_prob_final >= optimal_threshold).astype(int)

# 5. Calculate the other six metrics using the prediction results under the optimal threshold.
accuracy = accuracy_score(y_true_final, y_pred_optimal)
kappa = cohen_kappa_score(y_true_final, y_pred_optimal)
mcc = matthews_corrcoef(y_true_final, y_pred_optimal)
precision = precision_score(y_true_final, y_pred_optimal, zero_division=0)
recall = recall_score(y_true_final, y_pred_optimal, zero_division=0)
f1 = f1_score(y_true_final, y_pred_optimal, zero_division=0)

print("\n----- Model evaluation metrics -----")
print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC:       {roc_auc:.4f}")
print(f"Kappa:     {kappa:.4f}")
print(f"MCC:       {mcc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("--------------------------------------")

# 7. Plot the ROC curve and mark the optimal threshold point on the graph
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.scatter(fpr[ix_optimal], tpr[ix_optimal], marker='o', color='red', s=50,
            label=f'Optimal Threshold = {optimal_threshold:.3f}')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
