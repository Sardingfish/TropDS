#!/usr/bin/env python3
"""
Inference Script

Deblur 2024 data (blur version only).
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import UNet


class SingleSampleDataset(Dataset):
    """Single sample dataset - for inference"""
    
    def __init__(self, data, global_min=0.0, global_max=50.0):
        """
        Initialize
        
        Args:
            data: 2D or 3D numpy array
                - 2D: (H, W) single sample
                - 3D: (N, H, W) multiple samples
            global_min: Global minimum value of data
            global_max: Global maximum value of data
        """
        self.data = np.asarray(data)
        self.global_min = global_min
        self.global_max = global_max
        
        # Ensure data is 3D (N, H, W)
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, :, :]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx].copy()
        
        # Normalize to [0, 1]
        data = np.clip(data, self.global_min, self.global_max)
        data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
        
        # Convert to tensor (1, H, W)
        return torch.FloatTensor(data).unsqueeze(0)


class TroposphericDelayInference:
    """Tropospheric delay data inference class (supporting low RMSE region freezing)"""
    
    def __init__(self, model_path, global_min=0.0, global_max=50.0, 
                 base_channels=32, freeze_threshold=0.01, weight_map_path=None):
        """
        Initialize inference
        
        Args:
            model_path: Model file path
            global_min: Global minimum value of data
            global_max: Global maximum value of data
            base_channels: Model base channels (must match training)
            freeze_threshold: Freeze threshold, regions with RMSE below this value will be frozen
            weight_map_path: Weight file path, used to create freeze mask
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_min = global_min
        self.global_max = global_max
        
        # Load model - ensure same configuration
        print(f"Loading model: {model_path}")
        print(f"Model configuration: base_channels={base_channels}")
        
        self.model = UNet(
            n_channels=1,      # Input channels (tropospheric delay data)
            n_classes=1,       # Output channels
            base_channels=base_channels,
            freeze_threshold=freeze_threshold,
            weight_map_path=weight_map_path
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded to device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()
    def process(self, blur_data):
        """
        Process single blur tropospheric delay data
        
        Args:
            blur_data: Blur tropospheric delay data (H, W) or (N, H, W)
            
        Returns:
            numpy.ndarray: Deblurred tropospheric delay data
        """
        self.model.eval()
        
        # Save original shape
        original_shape = blur_data.shape
        original_ndim = blur_data.ndim
        
        # Ensure data is 3D (N, H, W)
        if blur_data.ndim == 2:
            blur_data = blur_data[np.newaxis, :, :]
        
        n_samples = len(blur_data)
        
        # Create dataset and dataloader
        dataset = SingleSampleDataset(blur_data, self.global_min, self.global_max)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        all_results = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch)  # (B, 1, H, W)
            
            # Denormalize
            output = output.cpu().numpy()  # (B, 1, H, W)
            output = output * (self.global_max - self.global_min) + self.global_min
            
            all_results.append(output)
        
        # Merge results
        all_results = np.concatenate(all_results, axis=0)  # (N, 1, H, W)
        
        # Restore original shape
        if original_ndim == 2:
            # Original is 2D, restore to 2D
            result = all_results[0, 0, :, :]  # (H, W)
        else:
            # Original is 3D, restore to 3D
            result = all_results[:, 0, :, :]  # (N, H, W)
        
        return result
    
    @torch.no_grad()
    def process_batch(self, blur_data_batch):
        """
        Batch process blur tropospheric delay data (compatible with train.py)
        
        Args:
            blur_data_batch: Batch blur data (N, H, W)
            
        Returns:
            numpy.ndarray: Batch deblur result (N, H, W)
        """
        self.model.eval()
        
        # Normalize
        data_norm = (blur_data_batch - self.global_min) / \
                   (self.global_max - self.global_min + 1e-8)
        
        # Convert to tensor (N, 1, H, W)
        data_tensor = torch.FloatTensor(data_norm).unsqueeze(1).to(self.device)
        
        # Forward pass
        output = self.model(data_tensor)  # (N, 1, H, W)
        
        # Denormalize
        output = output.cpu().numpy()  # (N, 1, H, W)
        output = output[:, 0, :, :]  # (N, H, W) - Remove channel dimension
        output = output * (self.global_max - self.global_min) + self.global_min
        
        return output


def main():
    """Main function"""
    print("\n" + "="*60)
    print("Start inference - processing 2024 data")
    print("="*60)
    
    # ========================
    # Configuration
    # ========================
    
    MODEL_PATH = './output/best_model.pth'
    INPUT_DIR = './data'
    OUTPUT_DIR = './output'
    YEAR_INFERENCE = 2024
    BLUR_SUFFIX = '_blur.npy'
    
    # Low RMSE region freezing configuration
    FREEZE_THRESHOLD = 0.01  # Regions with RMSE below this value will be frozen (use input directly)
    WEIGHT_MAP_PATH = './data/weight.npy'  # Weight file path
    
    # Get statistics from training data
    train_clear_path = f'{INPUT_DIR}/2020_clear.npy'
    if os.path.exists(train_clear_path):
        train_data = np.load(train_clear_path)
        global_min = float(train_data.min())
        global_max = float(train_data.max())
        print(f"Using training data statistics: min={global_min:.2f}, max={global_max:.2f}")
    else:
        # Try to get from validation data
        val_clear_path = f'{INPUT_DIR}/2024_clear.npy'
        if os.path.exists(val_clear_path):
            val_data = np.load(val_clear_path)
            global_min = float(val_data.min())
            global_max = float(val_data.max())
            print(f"Using validation data statistics: min={global_min:.2f}, max={global_max:.2f}")
        else:
            global_min, global_max = 0.0, 50.0
            print(f"Using default statistics: min={global_min:.2f}, max={global_max:.2f}")
    
    # ========================
    # Initialize inference
    # ========================
    
    # base_channels must match training
    inference = TroposphericDelayInference(
        model_path=MODEL_PATH,
        global_min=global_min,
        global_max=global_max,
        base_channels=32,  # Must match BASE_CHANNELS in train.py
        freeze_threshold=FREEZE_THRESHOLD,
        weight_map_path=WEIGHT_MAP_PATH if os.path.exists(WEIGHT_MAP_PATH) else None
    )
    
    # Print freeze information
    if hasattr(inference.model, 'freeze_mask') and inference.model.freeze_mask is not None:
        frozen_ratio = 1.0 - inference.model.freeze_mask.mean().item()
        print(f"\nFreeze function information:")
        print(f"  Freeze threshold: RMSE < {FREEZE_THRESHOLD}")
        print(f"  Frozen region ratio: {frozen_ratio*100:.2f}%")
    else:
        print(f"\nWarning: Freeze function not enabled")
    
    # ========================
    # Load data
    # ========================
    
    blur_path = f'{INPUT_DIR}/{YEAR_INFERENCE}{BLUR_SUFFIX}'
    
    if not os.path.exists(blur_path):
        print(f"Error: Data file does not exist - {blur_path}")
        return
    
    blur_data = np.load(blur_path)
    
    print(f"\nLoading data: {blur_path}")
    print(f"Original data shape: {blur_data.shape}")
    print(f"Data type: {blur_data.dtype}")
    
    # ========================
    # Process data
    # ========================
    
    print("\nStart processing...")
    
    if blur_data.ndim == 2:
        # Single sample
        result = inference.process(blur_data)
        print(f"Processing complete, output shape: {result.shape}")
    elif blur_data.ndim == 3:
        # Batch processing
        # Reshape to (N, 180, 360)
        original_shape = blur_data.shape
        
        if original_shape[1] != 180 or original_shape[2] != 360:
            # Need to reshape
            n_samples = blur_data.reshape(-1, 180, 360)
        else:
            blur_reshaped = blur_data
        
        result = inference.process(blur_data)
        print(f"Processing complete, output shape: {result.shape}")
        
        # Try to restore original shape (if possible)
        if len(original_shape) == 3:
            result = result.reshape(original_shape)
            print(f"Restore original shape: {original_shape}")
    else:
        print(f"Error: Unsupported data dimension {blur_data.ndim}")
        return
    
    # ========================
    # Save result
    # ========================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/{YEAR_INFERENCE}_deblurred.npy'
    np.save(output_path, result)
    
    print(f"\nResult saved: {output_path}")
    print(f"Output shape: {result.shape}")
    
    # ========================
    # Statistics
    # ========================
    
    print(f"\nOutput data statistics:")
    print(f"  Min: {result.min():.4f}")
    print(f"  Max: {result.max():.4f}")
    print(f"  Mean: {result.mean():.4f}")
    print(f"  Std: {result.std():.4f}")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
    
    return result


if __name__ == '__main__':
    result = main()
