#!/usr/bin/env python3
"""
Loss Function Definition

Contains:
- SSIMLoss: Structural Similarity Loss
- CombinedLoss: L1 + SSIM combined loss
- WeightedCombinedLoss: Spatial weighted loss (using historical RMSE weights)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Loss (SSIM Loss)
    
    SSIM is a perceptual quality metric that better matches human visual perception than MSE.
    """
    
    def __init__(self, window_size=11, data_range=1.0, channel=1):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.channel = channel
        self.window = None  # Delayed initialization
    
    def _create_window(self, window_size, channel):
        """Create Gaussian window"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _gaussian(self, window_size, sigma):
        """Generate Gaussian kernel"""
        gauss = torch.exp(torch.arange(window_size, dtype=torch.float)
                         .sub(window_size / 2).pow(2).mul(-0.5 / (sigma ** 2)))
        return gauss / gauss.sum()
    
    def forward(self, img1, img2):
        """Calculate SSIM loss"""
        # Get device
        device = img1.device
        
        # Ensure window is on correct device
        if self.window is None or self.window.device != device:
            self.window = self._create_window(self.window_size, self.channel).to(device)
        
        return 1 - ssim(img1, img2, self.data_range, self.window, self.window_size)


def ssim(img1, img2, data_range, window, window_size):
    """Calculate Structural Similarity Index (SSIM)"""
    channel = img1.size(1)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss Function
    
    Combines L1 loss (MAE) and SSIM loss.
    L1 loss focuses on pixel-level accuracy, SSIM loss focuses on perceptual quality.
    """

    def __init__(self, l1_weight=1.0, ssim_weight=0.1):
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        """Calculate combined loss"""
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        return self.l1_weight * l1 + self.ssim_weight * ssim


class WeightedCombinedLoss(nn.Module):
    """
    Spatial Weighted Combined Loss Function
    
    Uses historical RMSE weights for differentiated training on different regions:
    - High RMSE regions (large blur-clear difference): High weight, focus learning
    - Low RMSE regions (small blur-clear difference): Low weight, avoid over-correction
    
    Core idea: In regions with small difference, model should "move less" to avoid over-correction that makes it worse.
    """

    def __init__(self, weight_map_path=None, l1_weight=1.0, ssim_weight=0.1,
                 weight_mode='inverse', epsilon=1e-8):
        """
        Initialize spatial weighted loss function
        
        Args:
            weight_map_path: Weight file path (.npy format, 180x360 array)
                - Content is historical RMSE for each grid point
                - Smaller RMSE means smaller blur-clear difference, lower weight
            l1_weight: L1 loss weight
            ssim_weight: SSIM loss weight
            weight_mode: Weight mode
                - 'inverse': weight = 1 / (RMSE + epsilon), smaller RMSE means lower weight
                - 'normalized': weight normalized to [0,1], smaller RMSE means lower weight
                - 'squared': weight = RMSE^2, larger RMSE means higher weight
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.weight_mode = weight_mode
        self.epsilon = epsilon

        # Load and process weight map
        self.weight_map = self._load_weight_map(weight_map_path)

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def _load_weight_map(self, weight_map_path):
        """Load and process weight map"""
        if weight_map_path is None:
            print("  No weight file provided, using uniform weights")
            return None

        try:
            weight_map = np.load(weight_map_path)
            print(f"  Weight file loaded: {weight_map_path}")
            print(f"  Original RMSE range: [{weight_map.min():.4f}, {weight_map.max():.4f}]")

            # Convert to PyTorch tensor
            weight_tensor = torch.FloatTensor(weight_map)

            # Process based on weight mode
            if self.weight_mode == 'inverse':
                # weight = 1 / (RMSE + epsilon)
                # Smaller RMSE regions have lower weight, larger RMSE regions have higher weight
                weight_tensor = 1.0 / (weight_tensor + self.epsilon)
                print(f"  Weight mode: inverse")
            elif self.weight_mode == 'normalized':
                # Normalize to [0,1]
                weight_min = weight_tensor.min()
                weight_max = weight_tensor.max()
                if weight_max > weight_min:
                    weight_tensor = (weight_tensor - weight_min) / (weight_max - weight_min + self.epsilon)
                print(f"  Weight mode: normalized")
            elif self.weight_mode == 'squared':
                # weight = RMSE^2
                weight_tensor = weight_tensor ** 2
                print(f"  Weight mode: squared")
            else:
                raise ValueError(f"Unsupported weight mode: {self.weight_mode}")

            # Normalize weights to have mean 1 (maintain overall loss scale)
            weight_mean = weight_tensor.mean()
            if weight_mean > 0:
                weight_tensor = weight_tensor / weight_mean

            print(f"  Processed weight range: [{weight_tensor.min():.4f}, {weight_tensor.max():.4f}]")
            print(f"  Weight mean: {weight_tensor.mean():.4f}")

            return weight_tensor

        except Exception as e:
            print(f"  Failed to load weight file: {e}, using uniform weights")
            return None

    def _compute_weighted_l1(self, pred, target):
        """
        Calculate weighted L1 loss
        
        Core idea: In low RMSE regions (small original difference), reduce model correction;
        In high RMSE regions (large original difference), increase model correction.
        """
        if self.weight_map is None:
            # No weight map, use standard L1
            return self.l1_loss(pred, target)

        # Ensure weight map is on correct device
        device = pred.device
        weight = self.weight_map.to(device)

        # Ensure weight map shape matches (1, H, W)
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)

        # Expand to same shape as pred (B, 1, H, W)
        while weight.dim() < pred.dim():
            weight = weight.unsqueeze(0)

        # Calculate weighted absolute error
        abs_error = torch.abs(pred - target)  # (B, 1, H, W)

        # Apply spatial weight
        weighted_error = abs_error * weight  # (B, 1, H, W)

        # Return weighted average
        return weighted_error.mean()

    def forward(self, pred, target):
        """
        Calculate weighted combined loss
        
        Args:
            pred: Prediction (B, 1, H, W)
            target: Target (B, 1, H, W)

        Returns:
            torch.Tensor: Weighted combined loss
        """
        # Calculate weighted L1 loss
        weighted_l1 = self._compute_weighted_l1(pred, target)

        # SSIM loss (unweighted, maintain perceptual quality assessment)
        ssim = self.ssim_loss(pred, target)

        return self.l1_weight * weighted_l1 + self.ssim_weight * ssim


def load_rmse_weight_map(weight_path, device='cpu'):
    """
    Helper function: Load RMSE weight map and convert to tensor
    
    Args:
        weight_path: Weight file path
        device: Target device

    Returns:
        torch.Tensor: Weight tensor (1, 180, 360)
    """
    weight_map = np.load(weight_path)
    weight_tensor = torch.FloatTensor(weight_map).to(device)

    # Convert to (1, H, W) shape
    if weight_tensor.dim() == 2:
        weight_tensor = weight_tensor.unsqueeze(0)

    return weight_tensor


class FreezeAwareCombinedLoss(nn.Module):
    """
    Freeze-Aware Combined Loss Function
    
    Features:
    1. Support freeze mask, frozen regions don't participate in backpropagation
    2. Combine spatial weighted loss with freeze function
    3. Only non-frozen region errors update model parameters
    
    Freeze mask description:
    - freeze_mask = 1: Normal region, participate in loss calculation and parameter update
    - freeze_mask = 0: Frozen region, don't participate in loss calculation (output equals input directly)
    """

    def __init__(self, weight_map_path=None, l1_weight=1.0, ssim_weight=0.1,
                 weight_mode='inverse', freeze_threshold=0.01, epsilon=1e-8):
        """
        Initialize freeze-aware loss function
        
        Args:
            weight_map_path: Weight file path (.npy format, 180x360 array)
            l1_weight: L1 loss weight
            ssim_weight: SSIM loss weight
            weight_mode: Weight mode
            freeze_threshold: Freeze threshold, regions with RMSE below this value will be frozen
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.weight_mode = weight_mode
        self.freeze_threshold = freeze_threshold
        self.epsilon = epsilon

        # Load and process weight map and freeze mask
        self.weight_map = self._load_weight_map(weight_map_path)
        self.freeze_mask = self._load_freeze_mask(weight_map_path)

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def _load_freeze_mask(self, weight_map_path):
        """Load freeze mask"""
        if weight_map_path is None:
            return None

        try:
            weight_map = np.load(weight_map_path)
            # Create freeze mask: RMSE >= threshold -> 1 (normal), RMSE < threshold -> 0 (frozen)
            freeze_mask = (weight_map >= self.freeze_threshold).astype(np.float32)
            freeze_tensor = torch.FloatTensor(freeze_mask)
            frozen_ratio = 1.0 - freeze_mask.mean()
            print(f"  Freeze mask loaded: {frozen_ratio*100:.2f}% regions frozen")
            return freeze_tensor
        except Exception as e:
            print(f"  Failed to load freeze mask: {e}")
            return None

    def _load_weight_map(self, weight_map_path):
        """Load and process weight map"""
        if weight_map_path is None:
            return None

        try:
            weight_map = np.load(weight_map_path)
            weight_tensor = torch.FloatTensor(weight_map)

            if self.weight_mode == 'inverse':
                weight_tensor = 1.0 / (weight_tensor + self.epsilon)
            elif self.weight_mode == 'normalized':
                weight_min = weight_tensor.min()
                weight_max = weight_tensor.max()
                if weight_max > weight_min:
                    weight_tensor = (weight_tensor - weight_min) / (weight_max - weight_min + self.epsilon)
            elif self.weight_mode == 'squared':
                weight_tensor = weight_tensor ** 2

            # Normalize
            weight_mean = weight_tensor.mean()
            if weight_mean > 0:
                weight_tensor = weight_tensor / weight_mean

            return weight_tensor

        except Exception as e:
            print(f"  Failed to load weight file: {e}")
            return None

    def _apply_freeze_mask(self, pred, target, freeze_mask):
        """
        Apply freeze mask, only calculate loss for non-frozen regions
        
        Args:
            pred: Prediction (B, 1, H, W)
            target: Target (B, 1, H, W)
            freeze_mask: Freeze mask (1, H, W) or None

        Returns:
            tuple: (masked_pred, masked_target) - Masked tensors
        """
        if freeze_mask is None:
            # If no freeze mask, use instance variable
            freeze_mask = self.freeze_mask

        if freeze_mask is None:
            return pred, target

        device = pred.device
        mask = freeze_mask.to(device)

        # Ensure mask shape is correct (B, 1, H, W)
        while mask.dim() < 4:
            mask = mask.unsqueeze(0)

        # Set pred and target to same value for frozen regions, making loss 0
        # Frozen region error won't propagate gradient
        masked_pred = pred * mask
        masked_target = target * mask

        return masked_pred, masked_target

    def forward(self, pred, target, freeze_mask=None):
        """
        Calculate freeze-aware combined loss
        
        Args:
            pred: Prediction (B, 1, H, W)
            target: Target (B, 1, H, W)
            freeze_mask: Freeze mask (1, H, W) or None

        Returns:
            torch.Tensor: Freeze-aware combined loss
        """
        # Apply freeze mask
        masked_pred, masked_target = self._apply_freeze_mask(pred, target, freeze_mask)

        # Calculate weighted L1 loss
        if self.weight_map is not None:
            l1 = self._compute_weighted_l1(masked_pred, masked_target)
        else:
            l1 = self.l1_loss(masked_pred, masked_target)

        # SSIM loss
        ssim = self.ssim_loss(masked_pred, masked_target)

        return self.l1_weight * l1 + self.ssim_weight * ssim

    def _compute_weighted_l1(self, pred, target):
        """Calculate weighted L1 loss"""
        device = pred.device
        weight = self.weight_map.to(device)

        # Ensure weight map shape matches
        while weight.dim() < pred.dim():
            weight = weight.unsqueeze(0)

        abs_error = torch.abs(pred - target)
        weighted_error = abs_error * weight
        return weighted_error.mean()
