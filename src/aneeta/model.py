import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, _, _ = x.size()

        # Apply global average pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)

        # Shared MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        return x * out  # Apply attention weights


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # Keep output size same as input

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average along channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max along channel dimension

        # Concatenate average and max features
        concat = torch.cat([avg_out, max_out], dim=1)  # Shape: (batch_size, 2, H, W)
        out = self.sigmoid(self.conv(concat))  # Apply convolution and sigmoid

        return x * out  # Apply spatial attention weights


# CBAM Module (Combining Channel and Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention first, then Spatial Attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CBAMSpatialAttentionModule(nn.Module):
    def __init__(self, feature_dim, depth_dim):
        super(CBAMSpatialAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.depth_dim = depth_dim

        # Linear layer to combine object features and depth features
        self.fc = nn.Linear(feature_dim + depth_dim, feature_dim)

        # CBAM module for spatial attention
        self.cbam = CBAM(in_channels=feature_dim)

    def forward(self, f_obj, f_frame, f_depth):
        """
        Args:
            f_obj: (batch_size, T, N, D) - Object features
            f_frame: (batch_size, T, 1, D) - Frame-level features
            f_depth: (batch_size, T, N, d) - Depth features
        Returns:
            attention_output: (batch_size, T, N, D)
        """

        print("f_obj shape:", f_obj.shape)
        print("f_frame shape:", f_frame.shape)

        f_frame = f_frame.unsqueeze(1) if f_frame.dim() == 2 else f_frame

        # Concatenate object features with frame features
        combined_features = torch.cat([f_obj, f_frame], dim=2)  # (batch_size, 20, 4096)

        # Apply CBAM attention
        attention_features = self.cbam(combined_features)  # (batch_size, 20, 4096)

        # Remove the frame feature part to focus on objects
        object_attention_features = attention_features[:, :-1, :]  # (batch_size, 19, 4096)

        # Concatenate attention features with depth features
        depth_fused_features = torch.cat([object_attention_features, f_depth], dim=-1)  # (batch_size, 19, 4101)

        # Project back to 4096 dimensions
        output = self.fc(depth_fused_features)  # (batch_size, 19, 4096)

        return output


# Example usage
batch_size = 2
T = 50  # Number of frames
N = 19  # Number of objects
D = 4096  # Feature dimension
d = 5  # Depth feature dimension

# Dummy data
F_object = torch.rand(batch_size, T, N, D)       # Object features
F_frame = torch.rand(batch_size, T, 1, D)   # Frame features
F_depth = torch.rand(batch_size, T, N, d)   # Depth features

# Initialize and forward pass
cbam_attention = CBAMSpatialAttentionModule(D, d)
attention_output = cbam_attention(F_object, F_frame, F_depth)

print("Attention Output Shape:", attention_output.shape)  # (batch_size, T, N, D)
