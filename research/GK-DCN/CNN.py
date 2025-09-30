"""
Create a CNN model
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
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, roc_curve, auc
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_data = np.load("train_data.npy")             # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")           # Shape: (M, 9, 9, 39)
verify_labels = np.load("verify_labels.npy")

window_size = 9
all_channel = 39

# Define a custom convolutional neural network model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Input channels: 39, output channels: 64, convolution kernel: 3x3, keep size ('same' padding)
        self.conv1 = nn.Conv2d(in_channels=all_channel, out_channels=64, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64) 
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv1_2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

model = CustomModel().to(device)


l2_lambda = 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
criterion = nn.BCELoss()

train_data_tensor = torch.from_numpy(train_data).float().to(device)
train_labels_tensor = torch.from_numpy(train_labels).float().to(device)
verify_data_tensor = torch.from_numpy(verify_data).float().to(device)
verify_labels_tensor = torch.from_numpy(verify_labels).float().to(device)

# Model Training
history = []
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data_tensor)
    loss = criterion(outputs.squeeze(), train_labels_tensor)
    loss.backward()
    optimizer.step()
    print(epoch, loss.detach().numpy())
    model.eval()
    with torch.no_grad():
        verify_outputs = model(verify_data_tensor)
        verify_loss = criterion(verify_outputs.squeeze(), verify_labels_tensor)
        verify_acc = ((verify_outputs.squeeze() > 0.5) == verify_labels_tensor).float().mean()

    train_acc = ((outputs.squeeze() > 0.5) == train_labels_tensor).float().mean()
    history.append({'epoch': epoch, 'train_loss': loss.item(), 'test_loss': verify_loss.item(), 'train_acc': train_acc.item(), 'test_acc': verify_acc.item()})

torch.save(model.state_dict(), 'model_cnn.pth')


# Training process output
train_acc_history = [entry['train_acc'] for entry in history]
val_acc_history = [entry['test_acc'] for entry in history]
plt.plot([entry['epoch'] for entry in history], train_acc_history, label='Training set')
plt.plot([entry['epoch'] for entry in history], val_acc_history, label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_loss_history = [entry['train_loss'] for entry in history]
val_loss_history = [entry['test_loss'] for entry in history]
plt.plot([entry['epoch'] for entry in history], train_loss_history, label='Training set')
plt.plot([entry['epoch'] for entry in history], val_loss_history, label='Validation set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Perform full-image prediction using the trained model
model = CustomModel().to(device)
model.load_state_dict(torch.load('model_cnn.pth', map_location=device))
model.eval()
probability_value = []

all_array = np.load('Geochemical_array.npy')

print(f"\n--- Start full map prediction ({all_array.shape[0]}x{all_array.shape[1]}) ---")
with torch.no_grad():
    H_pred = all_array.shape[0] - (window_size - 1)
    W_pred = all_array.shape[1] - (window_size - 1)
    all_patches = []
    
    for row in range(H_pred):
        for col in range(W_pred):
            value = all_array[row:row+window_size, col:col+window_size, :]
            # (H, W, C) -> (1, H, W, C)
            all_patches.append(value.reshape(1, window_size, window_size, all_channel))
            
    if all_patches:
        # Stack into a batch (N, H, W, C)
        all_patches_tensor = torch.from_numpy(np.concatenate(all_patches, axis=0)).float().to(device)
        
        # Batch prediction
        full_outputs = model(all_patches_tensor).squeeze()
        probability_value = full_outputs.cpu().numpy().tolist()


XX_path = './New_data/coordinate/xx.tif'
YY_path = './New_data/coordinate/yy.tif'
Label_path = './New_data/label/Au_deposits0.tif'

XX = cv.imread(XX_path, cv.IMREAD_UNCHANGED)
YY = cv.imread(YY_path, cv.IMREAD_UNCHANGED)

center_offset = math.floor(window_size/2)
XX_result = XX[center_offset:XX.shape[0]-center_offset, center_offset:XX.shape[1]-center_offset]
YY_result = YY[center_offset:YY.shape[0]-center_offset, center_offset:YY.shape[1]-center_offset]

XX_result_list = XX_result.flatten()
YY_result_list = YY_result.flatten()

dataframe = pd.DataFrame({'XX': XX_result_list, 'YY': YY_result_list, 'probability': probability_value})
dataframe.to_csv('cnn_output.csv', index=False)
print("\nPrediction probability map saved to 'cnn_output.csv'")


# Model Evaluation
print("\n--- Start model evaluation (using the optimal F1 threshold) ---")

label_img = cv.imread(Label_path, cv.IMREAD_UNCHANGED)
if label_img is None:
    print("Error: Cannot read the label file, skip evaluation.")
    exit()

label_cropped = label_img[
    center_offset:label_img.shape[0]-center_offset,
    center_offset:label_img.shape[1]-center_offset
]
y_true = label_cropped.flatten()

unique_labels = np.unique(y_true)
if unique_labels.max() > 1:
    y_true_binary = (y_true == unique_labels.max()).astype(int)
else:
    y_true_binary = y_true.astype(int)

if len(y_true_binary) != len(probability_value):
    print(f"Error: The length of labels ({len(y_true_binary)}) does not match the length of probabilities ({len(probability_value)}), skipping evaluation.")
    exit()

y_true_final = y_true_binary
y_prob_final = np.array(probability_value)
print(f"Number of valid samples for final evaluation: {len(y_true_final)}")

# Calculate the FPR, TPR, and thresholds required for the ROC curve
fpr, tpr, thresholds = roc_curve(y_true_final, y_prob_final)
roc_auc = auc(fpr, tpr)
print(f"\nAUC (Area Under Curve): {roc_auc:.4f}")

# Finding the optimal threshold
f1_scores = []
for threshold in thresholds:
    y_pred = (y_prob_final >= threshold).astype(int)
    f1 = f1_score(y_true_final, y_pred, zero_division=0)
    f1_scores.append(f1)

f1_scores_np = np.array(f1_scores)
ix_optimal = np.argmax(f1_scores_np)
optimal_threshold = thresholds[ix_optimal]
optimal_f1 = f1_scores_np[ix_optimal]

# Perform classification prediction using the optimal threshold
y_pred_optimal = (y_prob_final >= optimal_threshold).astype(int)

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

tpr_optimal = recall_score(y_true_final, y_pred_optimal, zero_division=0)
tnr_optimal = accuracy_score(y_true_final[y_true_final == 0], y_pred_optimal[y_true_final == 0]) if np.sum(y_true_final == 0) > 0 else 0
fpr_optimal = 1 - tnr_optimal

fpr_at_optimal = fpr[ix_optimal]
tpr_at_optimal = tpr[ix_optimal]

plt.scatter(fpr_at_optimal, tpr_at_optimal, marker='o', color='red', s=50,
            label=f'Optimal F1 Threshold ({optimal_threshold:.3f})')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right', prop={"family": "Times New Roman", "size": 12})
plt.grid(True)

plt.show()
