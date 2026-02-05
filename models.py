#!/usr/bin/env python3
"""
U-Net Model Definition

Redesigned U-Net architecture to ensure correct channel count during skip connections.
Output layer uses Softplus activation to ensure strictly positive output (physical constraint).
Support for freezing low RMSE regions (bypass mode): regions with RMSE < 0.01 use input directly.

Input: (B, 1, 180, 360)
Output: (B, 1, 180, 360) - non-negative values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic Convolution Block
    
    Contains two convolution layers, each followed by BatchNorm and ReLU activation.
    Uses padding=1 to maintain input output size consistency.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder Block
    
    Downsampling + Convolution block.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block
    
    Upsampling + Convolution block + Skip connection.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size difference
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2])
        
        # Skip connection
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net Model (with physical constraint and low RMSE region freezing)
    
    Features:
    1. Encoder-decoder structure with skip connections
    2. Output layer uses Softplus activation to ensure strictly positive output
    3. Support for freezing low RMSE regions (RMSE < 0.01), these regions use input directly without changes
    
    Input: (B, 1, 180, 360)
    Output: (B, 1, 180, 360) - non-negative values
    
    Channel design (with base_channels=32):
    - encoder: 1 -> 32 -> 64 -> 128 -> 256 -> 512
    - decoder: 512 -> 256 -> 128 -> 64 -> 32 -> 1
    """
    
    def __init__(self, n_channels=1, n_classes=1, base_channels=32,
                 freeze_threshold=0.01, weight_map_path=None):
        """
        Initialize U-Net model
        
        Args:
            n_channels: Input channels (1 for tropospheric delay data)
            n_classes: Output channels (1 for tropospheric delay data)
            base_channels: Base channels (default 32)
            freeze_threshold: Freeze threshold, regions with RMSE below this value will be frozen (use input directly)
            weight_map_path: Weight file path, used to create freeze mask
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.freeze_threshold = freeze_threshold
        self.freeze_mask = None
        
        # ========================
        # Encoder path
        # ========================
        
        # Input block: 1 -> 32
        self.enc1 = ConvBlock(n_channels, base_channels)
        
        # Encoder blocks: 32 -> 64 -> 128 -> 256 -> 512
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        self.enc5 = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Bottleneck layer: 512 -> 512
        self.bottleneck = ConvBlock(base_channels * 16, base_channels * 16)
        
        # ========================
        # Decoder path
        # ========================
        
        # Decoder blocks: 512 -> 256 -> 128 -> 64 -> 32
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # ========================
        # Output layer (with physical constraint)
        # ========================
        
        # Use Softplus activation to ensure strictly positive output (mathematically always positive)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, n_classes, kernel_size=1),
            nn.Softplus()  # Physical constraint: Softplus(x) = log(1 + exp(x)) > 0
        )
        
        # Initialize weights
        self._init_weights()
        
        # Load freeze mask (if weight file provided)
        if weight_map_path is not None:
            self._load_freeze_mask(weight_map_path)
    
    def _load_freeze_mask(self, weight_map_path):
        """
        Load freeze mask from weight file
        
        Regions with RMSE < freeze_threshold will be frozen (output = input)
        Mask: 0 = frozen region (use input), 1 = normal region (use model output)
        """
        import numpy as np
        try:
            weight_map = np.load(weight_map_path)
            print(f"  Loading freeze mask from {weight_map_path}")
            print(f"  Original RMSE range: [{weight_map.min():.4f}, {weight_map.max():.4f}]")
            print(f"  Freeze threshold: {self.freeze_threshold}")
            
            # Create freeze mask: RMSE >= threshold -> 1 (normal), RMSE < threshold -> 0 (frozen)
            freeze_mask = (weight_map >= self.freeze_threshold).astype(np.float32)
            
            # Calculate freeze ratio
            frozen_ratio = 1.0 - freeze_mask.mean()
            print(f"  Frozen region ratio: {frozen_ratio*100:.2f}%")
            
            # Convert to PyTorch tensor
            self.freeze_mask = torch.FloatTensor(freeze_mask)
            print(f"  Mask shape: {self.freeze_mask.shape}")
            
        except Exception as e:
            print(f"  Warning: Cannot load freeze mask file {weight_map_path}: {e}")
            self.freeze_mask = None
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, freeze_mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 180, 360)
            freeze_mask: Freeze mask (1, H, W) or None (use preset mask)
                - 1: Normal region, use model output
                - 0: Frozen region, use input directly
            
        Returns:
            torch.Tensor: Deblurred tropospheric delay data (B, 1, 180, 360) - non-negative
        """
        # Save original input for frozen regions
        original_input = x
        
        # Encoder path
        e1 = self.enc1(x)           # (B, 32, 180, 360)
        e2 = self.enc2(e1)          # (B, 64, 90, 180)
        e3 = self.enc3(e2)          # (B, 128, 45, 90)
        e4 = self.enc4(e3)          # (B, 256, 22, 45)
        e5 = self.enc5(e4)          # (B, 512, 11, 22)
        
        # Bottleneck layer
        b = self.bottleneck(e5)     # (B, 512, 11, 22)
        
        # Decoder path (with skip connections)
        d4 = self.dec4(b, e4)       # (B, 256, 22, 45)
        d3 = self.dec3(d4, e3)      # (B, 128, 45, 90)
        d2 = self.dec2(d3, e2)      # (B, 64, 90, 180)
        d1 = self.dec1(d2, e1)      # (B, 32, 180, 360)
        
        # Output layer (with Softplus activation, ensure strictly positive)
        model_output = self.out(d1)  # (B, 1, 180, 360)
        
        # Determine which mask to use
        # Prefer passed mask, then preset mask
        active_mask = freeze_mask
        if active_mask is None and self.freeze_mask is not None:
            active_mask = self.freeze_mask
        
        # If freeze mask exists, blend input and model output
        if active_mask is not None:
            device = model_output.device
            active_mask = active_mask.to(device)
            
            # Ensure mask shape is correct (B, 1, H, W)
            while active_mask.dim() < 4:
                active_mask = active_mask.unsqueeze(0)
            
            # Frozen regions (mask=0) use original input, normal regions (mask=1) use model output
            # output = mask * model_output + (1 - mask) * input
            output = active_mask * model_output + (1 - active_mask) * original_input
        else:
            output = model_output
        
        return output


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_info(model):
    """Print model information"""
    total, trainable = count_parameters(model)
    print("\n" + "="*60)
    print("U-Net Model Information (with physical constraint)")
    print("="*60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Parameter magnitude: {total / 1e6:.2f} M")
    print(f"Output constraint: Softplus activation ensures strictly positive")
    print(f"Freeze threshold: Regions with RMSE < {model.freeze_threshold} will use input directly")
    print("="*60)


# ========================
# Test code
# ========================
if __name__ == "__main__":
    # Create model
    model = UNet(n_channels=1, n_classes=1, base_channels=32)
    print_model_info(model)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test input
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 180, 360).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        test_output = model(test_input)
        
        # Test freeze function
        if model.freeze_mask is not None:
            # Create random freeze mask for testing
            test_mask = torch.ones(1, 180, 360)
            test_mask[:, :90, :] = 0  # Freeze upper half (high latitude)
            test_output_masked = model(test_input, freeze_mask=test_mask)
            
            print(f"\nFreeze function test:")
            print(f"  Input upper half mean: {test_input[:, :, :90, :].mean().item():.4f}")
            print(f"  Model output upper half mean: {test_output[:, :, :90, :].mean().item():.4f}")
            print(f"  Masked output upper half mean: {test_output_masked[:, :, :90, :].mean().item():.4f}")
            print(f"  Frozen region output close to input: {torch.allclose(test_input[:, :, :90, :], test_output_masked[:, :, :90, :], atol=1e-6)}")
    
    print(f"\nTest result:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
    print(f"  Has negative values: {(test_output < 0).any().item()}")
    
    # Verify physical constraint
    assert test_output.min() >= 0, "Output contains negative values, physical constraint failed!"
    print("\n✓ Physical constraint verified: all output values are non-negative")
