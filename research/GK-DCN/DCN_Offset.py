"""
Offset situation of sampling points in a 9*9 window at the mining site
"""


import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.ops import DeformConv2d

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 9
all_channel = 39
output_dir = 'Offset'
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Network definition (exactly the same as during training, modified to return offsets)
# =============================================================================

class DeformableConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1, s=1):
        super().__init__()
        self.k, self.p, self.s = k, p, s
        
        # offset_conv
        self.offset_conv = nn.Conv2d(in_c, 2 * k * k, kernel_size=3, padding=1, bias=True)
        # mask_conv
        self.mask_conv = nn.Conv2d(in_c, k * k, kernel_size=3, padding=1, bias=True)

        # Deformable Convolutional Layer
        self.dcn = DeformConv2d(in_c, out_c, k, padding=p, stride=s, bias=True)

    def forward(self, x):
        # Calculate offsets and masks
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        out = self.dcn(x, offset, mask)
        return out, offset


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        """Do not add a pooling layer, ensuring that the input to the deformable convolution layer is 9*9"""
        self.conv1 = nn.Conv2d(all_channel, 64, 3, padding=1)
        self.bn1  = nn.BatchNorm2d(64)
        
        self.offsetconv1 = DeformableConv2d(64, 64, 3, 1)
        self.bn2   = nn.BatchNorm2d(64)
        
        # Keep subsequent layers to maintain weight compatibility
        self.pool2 = nn.MaxPool2d(2, 2)
        self.offsetconv2  = DeformableConv2d(64, 128, 3, 1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)       
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2) 
        
        # First layer convolution and BN (9x9)
        x = self.bn1(torch.relu(self.conv1(x)))
        
        # First layer DCN (9x9)
        x, off1 = self.offsetconv1(x)
        x = self.bn2(x)
        
        # For compatibility, return a fake probability value and return the offset of DCN1.
        prob = torch.tensor([0.5], device=x.device)
        return prob, off1, None # off2 is None


# Load Model
def load_model():
    # Read the weight file.
    model = CustomModel().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device), strict=False)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: Not found 'model.pth' ")
        return None
    
    model.eval()
    return model


# =============================================================================
# Offset situation at the visualized mining site window
# =============================================================================

def vis_deposit_offset():
    model = load_model()
    if model is None:
        return

    geo = np.load("Geochemical_array.npy")
    label = cv.imread('./New_data/label/Au_deposits0.tif', 2)

    # Find all mining sites
    deposits = np.argwhere(label == 1)
    print(f"A total of {len(deposits)} mining sites were found")


    for idx, (row, col) in enumerate(deposits):
        # Boundary check
        if row + window_size > geo.shape[0] or col + window_size > geo.shape[1]:
            continue

        patch = geo[row:row + window_size, col:col + window_size, :]
        # patch: (9, 9, 39) -> (1, 9, 9, 39)
        patch = torch.from_numpy(patch.reshape(1, window_size, window_size, all_channel)).float().to(device)

        # Forward propagation, obtain offsets
        with torch.no_grad(): 
            prob, off1, _ = model(patch)
        prob_value = prob.item() if prob.numel() == 1 else prob.mean().item()

        # --- Handle offset (off1 shape: [1, 18, 9, 9]) ---
        k = 3 # DCN kernel_size
        b, c, h, w = off1.shape # off1 shape: [1, 18, 9, 9] (H=W=9)

        ''' 1. Reshape the offset into [B, 2, k*k, H, W] '''
        off_map = off1.view(b, 2, k * k, h, w)

        ''' 2. Take the average of each k*k offset group to obtain [B, 2, H, W] '''
        # Get the average offset at this position, with a size of [1, 2, 9, 9]
        avg_off_map = off_map.mean(dim=2) 

        ''' 3. Remove the batch dimension and convert to numpy '''
        # The dimensions are [2, 9, 9], which can be used directly for plotting.
        avg_off_np = avg_off_map.squeeze(0).detach().cpu().numpy() 
        # U: X-direction offset, size [9, 9]
        U_plot = avg_off_np[0]
        # V: Y-direction offset, size [9, 9]
        V_plot = avg_off_np[1] 

        # The calculation range is used for color mapping
        M_plot = np.sqrt(U_plot ** 2 + V_plot ** 2)

        ''' 4. Use matplotlib to plot the improved offset visualization '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) 
        X, Y = np.meshgrid(np.arange(window_size), np.arange(window_size))
        
        # Subfigure 1: Offset Direction Vector Diagram
        q = ax1.quiver(X, Y, U_plot, V_plot, M_plot, cmap='viridis', 
                       units='xy', scale_units='xy', scale=0.5, # scale 参数可能需要根据偏移量大小调整
                       width=0.05, headwidth=4, headlength=4, alpha=0.8) 
        
        ax1.scatter(X, Y, c='black', s=20, zorder=5)
        
        ax1.set_title(f'Offset Direction - deposit_point ({row},{col})', fontsize=14)
        cbar1 = plt.colorbar(q, ax=ax1, shrink=0.8)
        cbar1.set_label('Offset Magnitude', fontsize=12)
        ax1.set_xlabel('X (Column Index)', fontsize=12)
        ax1.set_ylabel('Y (Row Index)', fontsize=12)
        ax1.set_xlim(-0.5, window_size - 0.5)
        ax1.set_ylim(-0.5, window_size - 0.5)
        ax1.invert_yaxis()
        ax1.set_xticks(range(window_size))
        ax1.set_yticks(range(window_size))
        
        for i in range(window_size + 1):
            ax1.axhline(i - 0.5, color='gray', linewidth=0.8, alpha=0.7)
            ax1.axvline(i - 0.5, color='gray', linewidth=0.8, alpha=0.7)
        
        # Subfigure 2: Heatmap of offset sizes
        im = ax2.imshow(M_plot, cmap='viridis', interpolation='none', origin='upper')
        ax2.set_title(f'Offset Magnitude Heatmap - deposit_point ({row},{col})', fontsize=14)
        cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar2.set_label('Offset Magnitude', fontsize=12)
        ax2.set_xlabel('X (Column Index)', fontsize=12)
        ax2.set_ylabel('Y (Row Index)', fontsize=12)
        ax2.set_xticks(range(window_size))
        ax2.set_yticks(range(window_size))
        
        for i in range(window_size + 1):
            ax2.axhline(i - 0.5, color='white', linewidth=1, alpha=0.8)
            ax2.axvline(i - 0.5, color='white', linewidth=1, alpha=0.8)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'offset_dep_{idx:03d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()



    print(f"✓ All mine site offset maps have been saved to {output_dir}/")

if __name__ == "__main__":
    vis_deposit_offset()