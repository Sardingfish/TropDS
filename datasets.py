#!/usr/bin/env python3
"""
Dataset Definition

Contains tropospheric delay dataset class, supporting:
- Auto normalization
- Data augmentation (longitude shift, vertical flip)
- Batch data loading
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os


class TroposphericDelayDataset(Dataset):
    """
    Global tropospheric delay dataset class
    
    Used to load paired clear-blur tropospheric delay data.
    
    Input expected: (H, W) or (N, H, W)
    Output: (1, H, W) tensor
    """
    
    def __init__(self, blur_data, clear_data, augmentation=True,
                 shift_range=(-30, 30), global_min=0.0, global_max=50.0):
        """
        Initialize dataset
        
        Args:
            blur_data: Blurry version numpy array (N, 180, 360)
            clear_data: Clear version numpy array (N, 180, 360)
            augmentation: Whether to enable data augmentation
            shift_range: Longitude direction random shift range
            global_min: Global minimum value of data
            global_max: Global maximum value of data
        """
        # Check data
        assert blur_data is not None and clear_data is not None, \
            "Data cannot be None, please check data file path"
        
        # Ensure data is 3D (N, H, W)
        if blur_data.ndim == 2:
            blur_data = blur_data[np.newaxis, :, :]
            clear_data = clear_data[np.newaxis, :, :]
        elif blur_data.ndim != 3:
            raise ValueError(f"Unsupported data dimension: {blur_data.ndim}, expected 2D or 3D")
        
        assert blur_data.shape == clear_data.shape, \
            f"Data shape mismatch: blur={blur_data.shape}, clear={clear_data.shape}"
        
        # Check size
        assert blur_data.shape[1] == 180 and blur_data.shape[2] == 360, \
            f"Data size error: {blur_data.shape[1:]}, expected (180, 360)"
        
        self.blur_data = blur_data
        self.clear_data = clear_data
        self.augmentation = augmentation
        self.shift_range = shift_range
        self.global_min = global_min
        self.global_max = global_max
        
        print(f"Dataset loaded: {len(self)} samples")
        print(f"  - Data shape: {blur_data.shape}")
        print(f"  - Clear data range: [{clear_data.min():.2f}, {clear_data.max():.2f}]")
        print(f"  - Blur data range: [{blur_data.min():.2f}, {blur_data.max():.2f}]")
        
    def __len__(self):
        """Return dataset size"""
        return len(self.blur_data)
    
    def __getitem__(self, idx):
        """Get single sample"""
        # Load original data
        blur = self.blur_data[idx].copy()
        clear = self.clear_data[idx].copy()
        
        # Data augmentation: longitude direction random shift
        if self.augmentation and self.shift_range is not None:
            shift = np.random.randint(self.shift_range[0], self.shift_range[1] + 1)
            blur = np.roll(blur, shift, axis=1)
            clear = np.roll(clear, shift, axis=1)
        
        # Data augmentation: random vertical flip
        if self.augmentation and np.random.random() > 0.5:
            blur = np.flipud(blur).copy()
            clear = np.flipud(clear).copy()
        
        # Normalize to [0, 1]
        blur = self._normalize(blur)
        clear = self._normalize(clear)
        
        # Convert to PyTorch tensor (1, H, W)
        blur_tensor = torch.FloatTensor(blur).unsqueeze(0)
        clear_tensor = torch.FloatTensor(clear).unsqueeze(0)
        
        return blur_tensor, clear_tensor
    
    def _normalize(self, data):
        """Normalize data to [0, 1]"""
        data = np.clip(data, self.global_min, self.global_max)
        data = (data - self.global_min) / (self.global_max - self.global_min + 1e-8)
        return data


def load_tropospheric_delay_data(data_dir, years_train, year_val,
                                 clear_suffix='_clear.npy', blur_suffix='_blur.npy'):
    """
    Load tropospheric delay data
    
    Args:
        data_dir: Data directory path
        years_train: Training years list
        year_val: Validation year
        clear_suffix: Clear data file suffix
        blur_suffix: Blur data file suffix
        
    Returns:
        tuple: (training data, validation data, statistics)
        If validation data doesn't exist, returns (train_blur, train_clear, None, None, stats)
    """
    print("\n" + "="*60)
    print("Loading tropospheric delay data")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Training years: {years_train}")
    print(f"Validation year: {year_val}")
    
    train_blur_list = []
    train_clear_list = []
    val_blur_list = []
    val_clear_list = []
    
    # Load training data
    print("\n" + "-"*40)
    print("Training data:")
    print("-"*40)
    
    for year in years_train:
        blur_path = os.path.join(data_dir, f'{year}{blur_suffix}')
        clear_path = os.path.join(data_dir, f'{year}{clear_suffix}')
        
        print(f"  Checking {year} data...", end=' ')
        
        if os.path.exists(blur_path) and os.path.exists(clear_path):
            blur = np.load(blur_path)
            clear = np.load(clear_path)
            
            print(f"✓ Found (shape: {blur.shape})")
            
            # Ensure data is 3D
            if blur.ndim == 2:
                blur = blur[np.newaxis, :, :]
                clear = clear[np.newaxis, :, :]
            elif blur.ndim == 3:
                blur = blur.reshape(-1, 180, 360)
                clear = clear.reshape(-1, 180, 360)
            
            train_blur_list.append(blur)
            train_clear_list.append(clear)
            print(f"    -> Processed: {blur.shape}")
        else:
            missing = []
            if not os.path.exists(blur_path):
                missing.append(f'{year}{blur_suffix}')
            if not os.path.exists(clear_path):
                missing.append(f'{year}{clear_suffix}')
            print(f"✗ Missing files: {', '.join(missing)}")
    
    # Load validation data
    print("\n" + "-"*40)
    print("Validation data:")
    print("-"*40)
    
    blur_path = os.path.join(data_dir, f'{year_val}{blur_suffix}')
    clear_path = os.path.join(data_dir, f'{year_val}{clear_suffix}')
    
    print(f"  Checking {year_val} data...", end=' ')
    
    if os.path.exists(blur_path) and os.path.exists(clear_path):
        blur = np.load(blur_path)
        clear = np.load(clear_path)
        
        print(f"✓ Found (shape: {blur.shape})")
        
        if blur.ndim == 2:
            blur = blur[np.newaxis, :, :]
            clear = clear[np.newaxis, :, :]
        elif blur.ndim == 3:
            blur = blur.reshape(-1, 180, 360)
            clear = clear.reshape(-1, 180, 360)
        
        val_blur_list.append(blur)
        val_clear_list.append(clear)
        print(f"    -> Processed: {blur.shape}")
    else:
        missing = []
        if not os.path.exists(blur_path):
            missing.append(f'{year_val}{blur_suffix}')
        if not os.path.exists(clear_path):
            missing.append(f'{year_val}{clear_suffix}')
        print(f"✗ Missing files: {', '.join(missing)}")
    
    # Merge data
    train_blur = np.concatenate(train_blur_list, axis=0) if train_blur_list else None
    train_clear = np.concatenate(train_clear_list, axis=0) if train_clear_list else None
    val_blur = np.concatenate(val_blur_list, axis=0) if val_blur_list else None
    val_clear = np.concatenate(val_clear_list, axis=0) if val_clear_list else None
    
    # Calculate global statistics
    if train_clear is not None:
        global_min = float(min(train_clear.min(), train_blur.min()))
        global_max = float(max(train_clear.max(), train_blur.max()))
    else:
        raise ValueError("No training data found! Please check data file path.")
    
    # Print summary
    print("\n" + "="*60)
    print("Data loading summary")
    print("="*60)
    print(f"Global data range: [{global_min:.2f}, {global_max:.2f}]")
    print(f"Training samples: {train_blur.shape[0] if train_blur is not None else 0}")
    print(f"Validation samples: {val_blur.shape[0] if val_blur is not None else 0}")
    
    if val_blur is None:
        print("\n⚠️  Warning: No validation data found, using last 10% of training data for validation")
    
    return train_blur, train_clear, val_blur, val_clear, (global_min, global_max)


def split_train_val(train_blur, train_clear, val_ratio=0.1):
    """
    Split validation set from training data
    """
    n_samples = len(train_blur)
    val_size = int(n_samples * val_ratio)
    
    # Take validation set from end of training data (maintain temporal order)
    val_blur = train_blur[-val_size:]
    val_clear = train_clear[-val_size:]
    train_blur = train_blur[:-val_size]
    train_clear = train_clear[:-val_size]
    
    print(f"\nSplit validation set from training data:")
    print(f"  Training set: {len(train_blur)} samples")
    print(f"  Validation set: {len(val_blur)} samples")
    
    return train_blur, train_clear, val_blur, val_clear


def list_available_data(data_dir):
    """List available data files in data directory"""
    print("\n" + "="*60)
    print("Check data directory")
    print("="*60)
    print(f"Directory: {data_dir}")
    print("-"*60)
    
    if not os.path.exists(data_dir):
        print(f"✗ Directory does not exist!")
        return
    
    files = sorted(os.listdir(data_dir))
    
    years_data = {}
    for f in files:
        if f.endswith('.npy'):
            year = f.split('_')[0]
            if year not in years_data:
                years_data[year] = []
            years_data[year].append(f)
    
    for year in sorted(years_data.keys()):
        files_list = years_data[year]
        has_clear = any('_clear' in f for f in files_list)
        has_blur = any('_blur' in f for f in files_list)
        
        status = "✓" if (has_clear and has_blur) else "✗"
        print(f"  {year}: {status}")
        for f in files_list:
            print(f"    - {f}")
    
    print("-"*60)
