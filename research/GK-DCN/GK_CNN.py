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

# =============================================================================
# Initialization Settings and Data Loading
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window_size = 9
all_channel = 39
model_save_path = 'GK_CNN_model.pth'
geo_lambda = 0.1 # Weight coefficient of L_geology
num_epochs = 80
BATCH_SIZE = 64

# Check required data files
def check_data_files():
    required_files = [
        "train_data.npy", "train_labels.npy", "verify_data.npy", 
        "verify_labels.npy", "Geochemical_array.npy", "geology_weights.tif"
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Error: Cannot find required file: {file}")
            

# Read data
train_data = np.load("train_data.npy")             # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")           # Shape: (M, 9, 9, 39)
verify_labels = np.load("verify_labels.npy")


# =============================================================================
# Load the geological potential map (w) and extract the corresponding values for the training samples
# =============================================================================

geology_weights_path = "geology_weights.tif"
geology_weights_full = cv.imread(geology_weights_path, cv.IMREAD_UNCHANGED).astype(np.float32)

# Normalization (w -> [0, 1])
w_min = geology_weights_full.min()
w_max = geology_weights_full.max()
if w_max > w_min:
    geology_weights_full = (geology_weights_full - w_min) / (w_max - w_min)
else:
    geology_weights_full = np.zeros_like(geology_weights_full, dtype=np.float32)

# Calculate the central coordinates and geological potential value w corresponding to the training samples
geo_array_shape = np.load('Geochemical_array.npy').shape
H_ori, W_ori = geo_array_shape[0], geo_array_shape[1]
H_data, W_data = H_ori - window_size + 1, W_ori - window_size + 1
train_data_num = train_data.shape[0]

train_indices = np.arange(train_data_num)
train_rows = train_indices // W_data
train_cols = train_indices % W_data
center_offset = window_size // 2

train_center_rows = np.clip(train_rows + center_offset, 0, geology_weights_full.shape[0] - 1)
train_center_cols = np.clip(train_cols + center_offset, 0, geology_weights_full.shape[1] - 1)

# Extract geological potential value w (as input for P_geology)
train_geo_potential_np = geology_weights_full[train_center_rows, train_center_cols]
train_geo_potential_tensor = torch.from_numpy(train_geo_potential_np).float().to(device).view(-1, 1) 


# =============================================================================
# Define the model
# =============================================================================

class GK_CNN(nn.Module):
    """
    Standard CNN structure, with an additional branch for Geological Alignment.
    """
    def __init__(self):
        super(GK_CNN, self).__init__()
        self.conv1 = nn.Conv2d(all_channel, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.geo_constraint_conv = nn.Conv2d(1, 1, kernel_size=1) 
        nn.init.constant_(self.geo_constraint_conv.weight, 0.1) 
        nn.init.constant_(self.geo_constraint_conv.bias, 0.0)    

    def forward(self, x, geo_potential_patch=None):
        # x shape: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Feature Extraction (Standard CNN)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))

        # Classification Path (P_model)
        x_flat = self.global_avg_pool(x).view(x.size(0), -1)
        x_flat = self.relu(self.fc1(x_flat))
        x_flat = self.dropout(x_flat)
        x_flat = self.fc2(x_flat)
        p_model = torch.sigmoid(x_flat) 
        
        # Geological Alignment Path (P_geology)
        p_geology = None
        if geo_potential_patch is not None:
            # P_geology = sigmoid(a*w + b)
            geo_input = geo_potential_patch.view(-1, 1, 1, 1)
            p_geology = self.geo_constraint_conv(geo_input)
            p_geology = torch.sigmoid(p_geology).view(-1, 1)

        return p_model, p_geology


# =============================================================================
# Define the geological alignment loss function (L_CNN + λ * L_geology)
# =============================================================================

class GeologicalAlignmentLoss(nn.Module):
    """Total Loss = BCE Loss + lambda * L2_Loss(P_model, P_geology)"""
    def __init__(self, geo_lambda=0.1):
        super(GeologicalAlignmentLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.geo_lambda = geo_lambda

    def forward(self, p_model, targets, p_geology):
        base_loss = self.bce_loss(p_model, targets)
        if p_geology is not None:
            geo_constraint_loss = self.l2_loss(p_model, p_geology)
            total_loss = base_loss + self.geo_lambda * geo_constraint_loss
        else:
            total_loss = base_loss
        
        return total_loss

# =============================================================================
# Data, model initialization and training
# =============================================================================

# Tensors
train_data_tensor = torch.from_numpy(train_data).float().to(device)
train_labels_tensor = torch.from_numpy(train_labels).float().to(device)
verify_data_tensor = torch.from_numpy(verify_data).float().to(device)
verify_labels_tensor = torch.from_numpy(verify_labels).float().to(device)

model = GK_CNN().to(device)
criterion = GeologicalAlignmentLoss(geo_lambda=geo_lambda)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
data_indices = list(range(train_data_num))
history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}


# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    np.random.shuffle(data_indices) 
    
    for i in range(0, train_data_num, BATCH_SIZE):
        batch_indices = data_indices[i:i + BATCH_SIZE]
        
        inputs = train_data_tensor[batch_indices]
        labels = train_labels_tensor[batch_indices]
        batch_geo_potential = train_geo_potential_tensor[batch_indices] 
        
        optimizer.zero_grad()
        p_model, p_geology = model(inputs, batch_geo_potential)
        
        loss = criterion(p_model.squeeze(), labels, p_geology)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        predicted = (p_model.squeeze() >= 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    
    with torch.no_grad():
        val_p_model, _ = model(verify_data_tensor)
        val_loss = nn.BCELoss(reduction='mean')(val_p_model.squeeze(), verify_labels_tensor).item()
        
        predicted = (val_p_model.squeeze() >= 0.5).float()
        total_val = verify_labels_tensor.size(0)
        correct_val = (predicted == verify_labels_tensor).sum().item()

    avg_train_loss = running_loss / (len(data_indices) // BATCH_SIZE)
    train_acc = correct_train / total_train
    val_acc = correct_val / total_val

    history['loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the model
torch.save(model.state_dict(), model_save_path)


# Plotting the training process
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), history['acc'], label='Training set')
plt.plot(range(1, num_epochs + 1), history['val_acc'], label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy (GK-CNN Alignment)')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), history['loss'], label='Training set')
plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss (GK-CNN Alignment)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =============================================================================
# Prediction and Evaluation
# =============================================================================

# Load model for prediction
model_pred = GK_CNN().to(device)
try:
    model_pred.load_state_dict(torch.load(model_save_path, map_location=device))
    model_pred.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

probability_value = []
all_array = np.load('Geochemical_array.npy')

# Model Prediction
print(f"\n--- Start full-image prediction ({all_array.shape[0]}x{all_array.shape[1]}) ---")
with torch.no_grad():
    H_pred = all_array.shape[0] - (window_size - 1)
    W_pred = all_array.shape[1] - (window_size - 1)
    all_patches = []
    
    for row in range(H_pred):
        for col in range(W_pred):
            value = all_array[row:row+window_size, col:col+window_size, :]
            all_patches.append(value.reshape(1, window_size, window_size, all_channel))
            
    if all_patches:
        all_patches_tensor = torch.from_numpy(np.concatenate(all_patches, axis=0)).float().to(device)

        full_outputs, _ = model_pred(all_patches_tensor, geo_potential_patch=None)
        probability_value = full_outputs.squeeze().cpu().numpy().tolist()


# Coordinate and CSV output
XX_path = './New_data/coordinate/XX.tif'
YY_path = './New_data/coordinate/YY.tif'
Label_path = './New_data/label/Au_deposits0.tif'

XX = cv.imread(XX_path, cv.IMREAD_UNCHANGED)
YY = cv.imread(YY_path, cv.IMREAD_UNCHANGED)

if XX is None or YY is None:
    print("Warning: Cannot find the coordinate file. Skipping CSV output.")
else:
    center_offset_pred = math.floor(window_size/2)
    XX_result = XX[center_offset_pred:XX.shape[0]-center_offset_pred, center_offset_pred:XX.shape[1]-center_offset_pred]
    YY_result = YY[center_offset_pred:YY.shape[0]-center_offset_pred, center_offset_pred:YY.shape[1]-center_offset_pred]
    
    XX_result_list = XX_result.flatten()
    YY_result_list = YY_result.flatten()
    dataframe = pd.DataFrame({'XX': XX_result_list, 'YY': YY_result_list, 'probability': probability_value})
    dataframe.to_csv('GK_CNN_output.csv', index=False)
    print("\nPrediction probability map saved to 'GK_CNN_output.csv'")


# Model Evaluation
print("\n--- Start model evaluation ---")

try:
    label_img = cv.imread(Label_path, cv.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError("Unable to read the file './New_data/label/Au_deposits0.tif'")
    
    center_offset_eval = math.floor(window_size/2)
    label_cropped = label_img[
        center_offset_eval:label_img.shape[0]-center_offset_eval,
        center_offset_eval:label_img.shape[1]-center_offset_eval
    ]
    y_true = label_cropped.flatten()
    
    unique_labels = np.unique(y_true)
    if 255 in unique_labels:
         y_true_binary = (y_true == 255).astype(int)
    elif unique_labels.max() > 1:
         y_true_binary = (y_true == unique_labels.max()).astype(int)
    else:
         y_true_binary = y_true.astype(int)

    if len(y_true_binary) != len(probability_value):
         raise ValueError(f"The number of processed labels ({len(y_true_binary)}) does not match the number of probabilities ({len(probability_value)}).")

    y_true_final = y_true_binary
    y_prob_final = np.array(probability_value)
    
except (FileNotFoundError, ValueError) as e:
    print(f"Error occurred during evaluation preparation: {e}")
    exit()

print(f"Number of valid samples for final evaluation: {len(y_true_final)}")

# 1. Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_final, y_prob_final)
roc_auc = auc(fpr, tpr)

# 2. Find the optimal F1 threshold
f1_scores = [f1_score(y_true_final, (y_prob_final >= t).astype(int), zero_division=0) for t in thresholds]
ix_optimal = np.argmax(f1_scores)
optimal_threshold = thresholds[ix_optimal]
optimal_f1 = f1_scores[ix_optimal]

# 3. Perform classification prediction using the optimal threshold
y_pred_optimal = (y_prob_final >= optimal_threshold).astype(int)

# 4. Calculate the other six metrics
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

fpr_optimal = fpr[ix_optimal]
tpr_optimal = tpr[ix_optimal]

plt.scatter(fpr_optimal, tpr_optimal, marker='o', color='red', s=50,
            label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()