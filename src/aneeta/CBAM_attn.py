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


# CBAM with Object, Depth, and Frame Features
class CBAMSpatialAttentionModule(nn.Module):
    def __init__(self, feature_dim, depth_dim):
        super(CBAMSpatialAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.depth_dim = depth_dim

        # Linear layer to combine object + depth + frame features
        self.fc = nn.Linear(feature_dim * 2 + depth_dim, feature_dim)

        # CBAM module
        self.cbam = CBAM(in_channels=feature_dim)

    def forward(self, F_obj, F_frame, F_depth):
        """
        Args:
            F_obj: (batch_size, T, N, D) - Object features
            F_frame: (batch_size, T, 1, D) - Frame-level features
            F_depth: (batch_size, T, N, d) - Depth features
        Returns:
            attention_output: (batch_size, T, N, D)
        """
        N = F_obj.shape[2]
        # Broadcast frame features to match object features
        Fframe_expanded = F_frame.expand(-1, -1, N, -1)  # (batch_size, T, N, D)

        # Combine object features, frame features, and depth features
        combined_features = torch.cat([F_obj, Fframe_expanded, F_depth], dim=-1)  # (batch_size, T, N, 2D + d)
        combined_features = self.fc(combined_features)  # Project to feature_dim (D)

        # Reshape for CBAM input (batch_size, channels, height, width)
        cbam_input = combined_features.permute(0, 3, 1, 2)  # (batch_size, D, T, N)

        # Apply CBAM
        output = self.cbam(cbam_input)

        # Reshape back to (batch_size, T, N, D)
        att_output = output.permute(0, 2, 3, 1)

        return att_output


# Example usage
#batch_size = 2
#T = 50   # Number of frames
#N = 19  # Number of objects
#D = 4096  # Feature dimension
#d = 16  # Depth feature dimension

# Dummy data
#F_obj = torch.rand(batch_size, T, N, D)       # Object features
#F_frame = torch.rand(batch_size, T, 1, D)   # Frame features
#F_depth = torch.rand(batch_size, T, N, d)   # Depth features

# Apply CBAM with frame features
#cbam_attention = CBAMSpatialAttentionModule(D, d)
#attention_output = cbam_attention(F_obj, F_frame, F_depth)

#print("Attention Output Shape:", attention_output.shape)  # (batch_size, T, N, D)

