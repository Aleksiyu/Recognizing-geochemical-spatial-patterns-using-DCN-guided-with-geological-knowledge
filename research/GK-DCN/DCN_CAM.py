"""
Grad-CAM for DCN
Display the class activation patterns of the first layer variable convolutional layer
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
from torchvision.ops import DeformConv2d

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = np.load("train_data.npy")            # Shape: (N, 9, 9, 39)
train_labels = np.load("train_labels.npy")
verify_data = np.load("verify_data.npy")          # Shape: (M, 9, 9, 39)
verify_labels = np.load("verify_labels.npy")

window_size = 9
all_channel = 39

os.makedirs('Grad-CAM_DCN', exist_ok=True)

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        k = kernel_size
        self.offset_conv = nn.Conv2d(in_channels, 2 * k * k, kernel_size=3, padding=1, bias=True)
        self.mask_conv = nn.Conv2d(in_channels, k * k, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)
        self.dcn = DeformConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.dcn(x, offset, mask)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=all_channel, out_channels=64, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.offsetconv1 = DeformableConv2d(64, 64)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.offsetconv2 = DeformableConv2d(64, 128)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        # (128*4*4) -> 2048
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x=x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.offsetconv1(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.offsetconv2(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

model = CustomModel().to(device)
torch.save(model.state_dict(), 'model.pth')
model.eval()

# ------------------ Define generating Grad-CAM ------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM.
        Args:
            input_tensor: Input tensor, the shape should be (B, H, W, C), consistent with the input format of the model forward.
            class_idx: Target class index. If None, use the class with the maximum probability output by the model.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward propagation
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output).item()

        # Backward propagation
        self.model.zero_grad()
        # Create a one-hot vector. Note that the shape of output is (B, 1)
        one_hot_output = torch.zeros_like(output, dtype=torch.float32, device=output.device)
        one_hot_output[0][0] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        if gradients is None or activations is None:
            print("Warning: Gradients or activations not found. Returning zero CAM.")
            return np.zeros((input_tensor.shape[1], input_tensor.shape[2]))

        # Global average pooling gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # (C,)

        # Weighted sum
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)  # (H, W)
        for i, w in enumerate(pooled_gradients):
            cam += w * activations[0, i, :, :]

        # ReLU
        cam = torch.relu(cam)

        # Convert to numpy and normalize
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def process_sample_gradcam(model, sample_data, sample_name):
    """Process a single sample, generate Grad-CAM and save (include original data visualization and CAM to rgbcam folder)"""
    print(f"Processing sample: {sample_name}")
    
    # Convert to Tensor, shape (1, 9, 9, 39)
    input_tensor = torch.from_numpy(sample_data).float().to(device)
    
    grad_cam = GradCAM(model, model.offsetconv1) 
    cam = grad_cam.generate_cam(input_tensor)
    
    # Extract original input data (remove batch dimension) from sample_data
    input_data_np = sample_data[0] # (H, W, C)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display normalized and superimposed original data
    normalized_data = np.zeros_like(input_data_np, dtype=np.float32)
    for c in range(input_data_np.shape[2]):
        channel_data = input_data_np[:, :, c]
        min_val = channel_data.min()
        max_val = channel_data.max()
        if max_val > min_val:
             normalized_data[:, :, c] = (channel_data - min_val) / (max_val - min_val)
        else:
             normalized_data[:, :, c] = 0

    combined_data = np.mean(normalized_data, axis=2)
    combined_min = combined_data.min()
    combined_max = combined_data.max()
    if combined_max > combined_min:
        combined_data_final = (combined_data - combined_min) / (combined_max - combined_min)
    else:
        combined_data_final = np.zeros_like(combined_data)

    ax1 = axes[0]
    im1 = ax1.imshow(combined_data_final, cmap='viridis')
    ax1.set_title(f'{sample_name}')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('')

    # Display Grad-CAM
    ax2 = axes[1]
    im2 = ax2.imshow(cam, cmap='jet', interpolation='nearest') 
    ax2.set_title(f'{sample_name} - DCN - CAM')
    ax2.set_xticks(np.arange(-.5, cam.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, cam.shape[0], 1), minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax2.tick_params(which='minor', size=0)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('')

    plt.tight_layout()
    OUTPUT_DIR = 'Grad-CAM_DCN'
    output_path = f'{OUTPUT_DIR}/{sample_name}_gradcam.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

train_data_tensor = torch.from_numpy(train_data).float().to(device)
train_labels_tensor = torch.from_numpy(train_labels).float().to(device)
verify_data_tensor = torch.from_numpy(verify_data).float().to(device)
verify_labels_tensor = torch.from_numpy(verify_labels).float().to(device)

# Model training
history = []
for epoch in range(80):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data_tensor)
    loss = criterion(outputs.squeeze(), train_labels_tensor)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        verify_outputs = model(verify_data_tensor)
        verify_loss = criterion(verify_outputs.squeeze(), verify_labels_tensor)
        verify_acc = ((verify_outputs.squeeze() > 0.5) == verify_labels_tensor).float().mean()

    train_acc = ((outputs.squeeze() > 0.5) == train_labels_tensor).float().mean()
    history.append({'epoch': epoch, 'train_loss': loss.item(), 'test_loss': verify_loss.item(), 'train_acc': train_acc.item(), 'test_acc': verify_acc.item()})
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/300], Train Loss: {loss.item():.4f}, Val Loss: {verify_loss.item():.4f}, Train Acc: {train_acc.item():.4f}, Val Acc: {verify_acc.item():.4f}")

torch.save(model.state_dict(), 'model.pth')


# Call the trained model
model = CustomModel().to(device)
state_dict = torch.load('model.pth', map_location=device) 
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

# ------------------ Add: Generate Grad-CAM for all deposit points  ------------------
print("Start generating Grad-CAM for all known deposit points ...")

# Read Geochemical_array and label image
try:
    all_array = np.load('Geochemical_array.npy')
    print(f"Geochemical_array.npy loaded successfully, shape: {all_array.shape}")
    label_img = cv.imread('./New_data/label/Au_deposits0.tif', cv.IMREAD_GRAYSCALE)
    if label_img is None:
        raise FileNotFoundError("Unable to read label image ./New_data/label/Au_deposits0.tif")
    print(f"Label image Au_deposits0.tif loaded successfully, shape: {label_img.shape}")
except (FileNotFoundError, Exception) as e:
    raise e

required_height = all_array.shape[0] - (window_size - 1)
required_width = all_array.shape[1] - (window_size - 1)

if label_img.shape[0] < required_height or label_img.shape[1] < required_width:
    raise ValueError(f"Label image size {label_img.shape} is smaller than the required size for Geochemical_array ({required_height}, {required_width})")

# Crop the label image to match the effective prediction area of Geochemical_array
pad_h = (label_img.shape[0] - required_height) // 2
pad_w = (label_img.shape[1] - required_width) // 2
label_img_cropped = label_img[pad_h:pad_h+required_height, pad_w:pad_w+required_width]
print(f"Label image cropped size: {label_img_cropped.shape}")

# Find all deposit point coordinates (in the cropped label image)
mining_point_rows_cropped, mining_point_cols_cropped = np.where(label_img_cropped == 1)
num_points = len(mining_point_rows_cropped)
print(f"Found {num_points} deposit points.")

if num_points == 0:
    print("Warning: No deposit points found in the label image (pixels with value 1).")
else:
    generated_count = 0
    half_window = window_size // 2
    for i in range(num_points):
        r_top_left = mining_point_rows_cropped[i]
        c_top_left = mining_point_cols_cropped[i]

        # Check boundaries
        if (r_top_left >= 0 and r_top_left <= all_array.shape[0] - window_size and
            c_top_left >= 0 and c_top_left <= all_array.shape[1] - window_size):
            
            sample_data = all_array[r_top_left:r_top_left+window_size, c_top_left:c_top_left+window_size, :]
            sample_data_with_batch = np.expand_dims(sample_data, axis=0)
            
            # Calculate the center point coordinates
            center_r = r_top_left + half_window
            center_c = c_top_left + half_window
            point_name = f"deposit_point_({center_r},{center_c})" 
            
            # Generate and save Grad-CAM
            process_sample_gradcam(model, sample_data_with_batch, point_name)
            generated_count += 1
        else:
            print(f"Warning: Skipping deposit point coordinates ({r_top_left}, {c_top_left}) out of bounds")

    print(f"Completed {generated_count}/{num_points} deposit points Grad-CAM generation.")

print("All deposit points Grad-CAM generation completed.")

