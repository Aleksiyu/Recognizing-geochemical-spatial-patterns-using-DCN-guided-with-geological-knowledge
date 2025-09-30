"""
Incorporating Geological Knowledge (GK) into Standard Convolutional Networks (CNN) 
- Replacing Deformable Convolution with Standard Convolution.
- Using the penalty-based soft constraint loss.
"""


import math
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, roc_curve, auc
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read data
train_data = np.load("train_data.npy")             # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")           # Shape: (M, 9, 9, 39)
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
    print("Warning: Geological weight error. All weights set to zero.")

window_size = 9
all_channel = 39

# Spatial coordinates for preparing training data
try:
    geo_array_shape = np.load('Geochemical_array.npy').shape
    H_ori, W_ori = geo_array_shape[0], geo_array_shape[1]
except FileNotFoundError:
    # Fallback to estimate size if the file is missing
    H_ori = int(np.sqrt(train_data.shape[0])) + window_size - 1
    W_ori = H_ori

H_data, W_data = H_ori - window_size + 1, W_ori - window_size + 1
if train_data.shape[0] > H_data * W_data:
     raise ValueError("wrong data.")

train_indices = np.arange(train_data.shape[0])
train_rows = train_indices // W_data
train_cols = train_indices % W_data

center_offset = window_size // 2
train_center_rows = train_rows + center_offset
train_center_cols = train_cols + center_offset

train_center_rows = np.clip(train_center_rows, 0, geology_weights_full.shape[0] - 1)
train_center_cols = np.clip(train_center_cols, 0, geology_weights_full.shape[1] - 1)

train_geo_weights_np = geology_weights_full[train_center_rows, train_center_cols]
train_geo_weights_tensor = torch.from_numpy(train_geo_weights_np).float().to(device)


# =============================================================================
# Define Custom Loss Function (Geological Penalty)
# =============================================================================

geo_weight_coeff = 0.1 

class GeologicalConstrainedBCELoss(nn.Module):
    """BCE Loss with an added geological constraint penalty term."""
    def __init__(self, geo_weight_coeff=0.1):
        super(GeologicalConstrainedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.geo_weight_coeff = geo_weight_coeff

    def forward(self, inputs, targets, geo_weights):
        # inputs: The model's predicted probability (sigmoid output)
        # targets: True label
        # geo_weights: Geological Constraint Weight (N)

        # 1. Base Loss (BCE)
        base_loss = self.bce_loss(inputs, targets)

        # 2. Constraint Term (Penalty): geo_weights * |inputs - targets|
        # Expand the geo_weights dimension to match the input dimension (N, 1)
        geo_weights_expanded = geo_weights.view_as(targets)
        # Compute prediction error
        prediction_error = torch.abs(inputs - targets)
        # Calculate penalty term
        geo_constraint = prediction_error * geo_weights_expanded
        
        # 3. Total Loss: Base Loss + Coefficient * Constraint Term
        total_loss = base_loss + self.geo_weight_coeff * torch.mean(geo_constraint)
        
        return total_loss

# =============================================================================
# Define the Model (Replace DCN with Standard CNN)
# =============================================================================

class GK_CNN_Penalty(nn.Module):
    def __init__(self, in_channels=39, out_channels=1, window_size=9):
        super(GK_CNN_Penalty, self).__init__() # Input dimension: (N, C, H, W) = (N, 39, 9, 9)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2: (64 -> 64) + MaxPool
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 3: (64 -> 128)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        # Flatten size: 128 * 4 * 4 = 2048
        self.flatten_size = 128 * 4 * 4
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (N, C, H, W)
        # Layer 1: Conv -> BN -> ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        # Layer 2: Conv -> BN -> ReLU -> MaxPool
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # Layer 3: Conv -> BN -> ReLU
        x = self.relu(self.bn3(self.conv3(x)))
        # Flatten
        x = x.view(-1, self.flatten_size)
        # Fully Connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)


# =============================================================================
# Data Loading and Training Setup
# =============================================================================

# (N, H, W, C) -> (N, C, H, W)
train_data_tensor = torch.from_numpy(train_data).permute(0, 3, 1, 2).float().to(device)
train_labels_tensor = torch.from_numpy(train_labels).float().view(-1, 1).to(device)

verify_data_tensor = torch.from_numpy(verify_data).permute(0, 3, 1, 2).float().to(device)
verify_labels_tensor = torch.from_numpy(verify_labels).float().view(-1, 1).to(device)

# PyTorch Data Loaders
train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
verify_dataset = torch.utils.data.TensorDataset(verify_data_tensor, verify_labels_tensor)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
verify_loader = torch.utils.data.DataLoader(verify_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize Model, Loss, and Optimizer
model = GK_CNN_Penalty(in_channels=all_channel, window_size=window_size).to(device)
criterion = GeologicalConstrainedBCELoss(geo_weight_coeff=geo_weight_coeff)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 80
history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Extract the geological weights of the current batch
        current_indices = i * BATCH_SIZE
        batch_size = labels.size(0)
        batch_geo_weights = train_geo_weights_tensor[current_indices:current_indices + batch_size]
        
        # Using a penalty term loss function
        loss = criterion(outputs, labels, batch_geo_weights)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Accuracy calculation
        predicted = (outputs >= 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in verify_loader:
            outputs = model(inputs)

            val_loss += nn.BCELoss()(outputs, labels).item()
            
            predicted = (outputs >= 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(verify_loader)
    train_acc = correct_train / total_train
    val_acc = correct_val / total_val

    history['loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the model
model_save_path = 'model_GK_CNN.pth'
torch.save(model.state_dict(), model_save_path)


# Plotting the training process
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), history['acc'], label='Training set')
plt.plot(range(1, num_epochs + 1), history['val_acc'], label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy (CNN + Penalty Loss)')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), history['loss'], label='Training set')
plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss (CNN + Penalty Loss)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =============================================================================
# Prediction and Evaluation
# =============================================================================

# Load model for prediction
model_pred = GK_CNN_Penalty(in_channels=all_channel, window_size=window_size).to(device)
try:
    model_pred.load_state_dict(torch.load(model_save_path))
    model_pred.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

probability_value = []

# Read the prediction dataset
try:
    all_array = np.load('Geochemical_array.npy')
except FileNotFoundError:
    print("Error: Geochemical_array.npy not found. Skipping full prediction.")
    exit()


# Model Prediction
with torch.no_grad():
    for row in range(all_array.shape[0] - (window_size - 1)):
        for col in range(all_array.shape[1] - (window_size - 1)):
            value = all_array[row:row + window_size, col:col + window_size, :]
            # (H, W, C) -> (1, H, W, C) -> (1, C, H, W)
            value = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output = model_pred(value)
            probability_value.append(output.cpu().numpy()[0][0])

# Coordinate and CSV output
XX_path = './New_data/coordinate/XX.tif'
YY_path = './New_data/coordinate/YY.tif'

XX = cv.imread(XX_path, cv.IMREAD_UNCHANGED)
YY = cv.imread(YY_path, cv.IMREAD_UNCHANGED)

if XX is None or YY is None:
    print("Warning: Cannot find the coordinate file. Skipping CSV output.")
else:
    center_offset = math.floor(window_size/2)
    XX_result = XX[center_offset:XX.shape[0]-center_offset, center_offset:XX.shape[1]-center_offset]
    YY_result = YY[center_offset:YY.shape[0]-center_offset, center_offset:YY.shape[1]-center_offset]
    
    XX_result_list = XX_result.flatten()
    YY_result_list = YY_result.flatten()
    dataframe = pd.DataFrame({'XX': XX_result_list, 'YY': YY_result_list, 'probability': probability_value})
    dataframe.to_csv('GK_cnn_output.csv', index=False)
    print("\nPrediction probability map saved to 'GK_cnn_output.csv'")


# Model Evaluation
try:
    # 1. Read and process the real labels
    label_img = cv.imread('./New_data/label/Au_deposits0.tif', cv.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError("Unable to read the file './New_data/label/Au_deposits0.tif'")
    
    center_offset = math.floor(window_size/2)
    label_cropped = label_img[
        center_offset:label_img.shape[0]-center_offset,
        center_offset:label_img.shape[1]-center_offset
    ]
    y_true = label_cropped.flatten()
    
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 2:
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

    print(f"\nNumber of valid samples used for evaluation: {len(y_true_final)}")

except (FileNotFoundError, ValueError) as e:
    print(f"Error occurred during evaluation preparation: {e}")
    exit()

# 2. Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_final, y_prob_final)
roc_auc = auc(fpr, tpr)

# 3. Find the optimal F1 threshold
f1_scores = [f1_score(y_true_final, (y_prob_final >= t).astype(int), zero_division=0) for t in thresholds]
ix_optimal = np.argmax(f1_scores)
optimal_threshold = thresholds[ix_optimal]
optimal_f1 = f1_scores[ix_optimal]

# 4. Perform classification prediction using the optimal threshold
y_pred_optimal = (y_prob_final >= optimal_threshold).astype(int)

# 5. Calculate the other six metrics using the prediction results under the optimal threshold.
accuracy = accuracy_score(y_true_final, y_pred_optimal)
kappa = cohen_kappa_score(y_true_final, y_pred_optimal)
mcc = matthews_corrcoef(y_true_final, y_pred_optimal)
precision = precision_score(y_true_final, y_pred_optimal, zero_division=0)
recall = recall_score(y_true_final, y_pred_optimal, zero_division=0)
f1 = optimal_f1

print("\n--- Model evaluation metrics ---")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC:       {roc_auc:.4f}")
print(f"Kappa:     {kappa:.4f}")
print(f"MCC:       {mcc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("--------------------------------------")


# Plot the ROC curve and mark the optimal F1 threshold point on the graph.
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.plot(fpr, tpr, label='AUC=({:.3f})'.format(roc_auc))

# Find the point on the ROC curve corresponding to the optimal F1 threshold
# We need to find the FPR and TPR at the optimal threshold
try:
    # A more robust way to find the closest point
    t_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    fpr_optimal = fpr[t_idx]
    tpr_optimal = tpr[t_idx]
except:
    # Fallback if the above fails
    fpr_optimal = 1 - accuracy_score(y_true_final[y_true_final == 0], y_pred_optimal[y_true_final == 0]) if np.sum(y_true_final == 0) > 0 else 0
    tpr_optimal = recall_score(y_true_final, y_pred_optimal, zero_division=0)

plt.scatter(fpr_optimal, tpr_optimal, marker='o', color='red', s=50,
            label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()