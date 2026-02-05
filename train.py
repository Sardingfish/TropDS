#!/usr/bin/env python3
"""
U-Net Training Script

Train model using 2020-2023 data, validate on 2024 data.
If 2024 data doesn't exist, split validation set from training data.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import UNet, count_parameters, print_model_info
from datasets import TroposphericDelayDataset, load_tropospheric_delay_data, split_train_val, list_available_data
from losses import CombinedLoss, WeightedCombinedLoss, FreezeAwareCombinedLoss
from metrics import TroposphericDelayMetrics

# Ignore warnings
warnings.filterwarnings('ignore')

# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class UNetTrainer:
    """U-Net Trainer Class"""
    
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate=1e-4, epochs=100, patience=15,
                 l1_weight=1.0, ssim_weight=0.1, weight_decay=1e-5,
                 T_0=20, T_mult=2, global_max=50.0,
                 use_weighted_loss=False, weight_map_path=None, weight_mode='inverse',
                 freeze_threshold=0.01, enable_freeze=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_max = global_max
        self.enable_freeze = enable_freeze
        self.freeze_threshold = freeze_threshold
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'lr': []
        }
        
        # Get freeze mask
        freeze_mask = None
        if enable_freeze and hasattr(model, 'freeze_mask') and model.freeze_mask is not None:
            freeze_mask = model.freeze_mask
        
        # Loss function
        if enable_freeze and weight_map_path is not None:
            # Use freeze-aware loss function
            self.criterion = FreezeAwareCombinedLoss(
                weight_map_path=weight_map_path,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
                weight_mode=weight_mode,
                freeze_threshold=freeze_threshold
            )
            print(f"\nUsing freeze-aware loss function")
            print(f"  Freeze threshold: RMSE < {freeze_threshold}")
            if freeze_mask is not None:
                frozen_ratio = 1.0 - freeze_mask.mean().item()
                print(f"  Frozen region ratio: {frozen_ratio*100:.2f}%")
        elif use_weighted_loss and weight_map_path is not None:
            self.criterion = WeightedCombinedLoss(
                weight_map_path=weight_map_path,
                l1_weight=l1_weight,
                ssim_weight=ssim_weight,
                weight_mode=weight_mode
            )
        else:
            if use_weighted_loss:
                print("  Warning: Weight file not found, using standard loss function")
            self.criterion = CombinedLoss(l1_weight=l1_weight, ssim_weight=ssim_weight)
        
        # Save freeze mask for loss calculation
        self.freeze_mask = freeze_mask
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        
        print("\nTrainer initialization complete")
        print(f"  Total training epochs: {epochs}")
        print(f"  Initial learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        total_rmse = 0.0
        
        for batch_idx, (blur, clear) in enumerate(self.train_loader):
            blur = blur.to(device)
            clear = clear.to(device)
            
            # Forward pass
            output = self.model(blur, freeze_mask=self.freeze_mask)
            
            # Calculate loss (pass freeze mask, frozen regions don't participate in backpropagation)
            loss = self.criterion(output, clear, freeze_mask=self.freeze_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate RMSE (use complete output, including frozen regions bypass)
            rmse = torch.sqrt(F.mse_loss(output, clear)).item()
            
            total_loss += loss.item()
            total_rmse += rmse
            
            # Print progress
            if (batch_idx + 1) % 200 == 0:
                print(f'  Batch [{batch_idx+1}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.6f} RMSE: {rmse:.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        avg_rmse = total_rmse / len(self.train_loader)
        
        return avg_loss, avg_rmse
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        
        all_preds = []
        all_targets = []
        
        for blur, clear in self.val_loader:
            blur = blur.to(device)
            clear = clear.to(device)
            
            output = self.model(blur, freeze_mask=self.freeze_mask)
            
            # Use freeze mask during validation
            loss = self.criterion(output, clear, freeze_mask=self.freeze_mask)
            
            rmse = torch.sqrt(F.mse_loss(output, clear)).item()
            
            total_loss += loss.item()
            total_rmse += rmse
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(clear.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_rmse = total_rmse / len(self.val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return avg_loss, avg_rmse, all_preds, all_targets
    
    def train(self, output_dir='./output'):
        """Complete training process"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Start training")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_rmse = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_rmse, val_preds, val_targets = self.validate()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print information
            print(f'\nEpoch {epoch}/{self.epochs} - {epoch_time:.1f}s')
            print(f'  Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.4f}')
            print(f'  Val Loss:   {val_loss:.6f} | Val RMSE:   {val_rmse:.4f}')
            print(f'  Learning Rate: {current_lr:.2e}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f'{output_dir}/best_model.pth', epoch)
                print(f'  *** Save best model (Val Loss: {val_loss:.6f}) ***')
                
                # Calculate detailed metrics
                val_preds_denorm = val_preds * self.global_max
                val_targets_denorm = val_targets * self.global_max
                metrics = TroposphericDelayMetrics.compute_all(
                    val_preds_denorm.reshape(-1),
                    val_targets_denorm.reshape(-1)
                )
                print(f'  Detailed metrics:')
                for name, value in metrics.items():
                    print(f'    {name}: {value:.4f}')
            else:
                self.patience_counter += 1
                print(f'  No improvement ({self.patience_counter}/{self.patience})')
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f'\nEarly stopping triggered! Stop training at epoch {epoch}')
                break
        
        total_time = time.time() - start_time
        print(f'\nTraining complete! Total time: {total_time/60:.2f} minutes')
        print(f'Best validation loss: {self.best_val_loss:.6f}')
        
        return self.history
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f'Checkpoint loaded: {filename}')
        return checkpoint['epoch']


def main():
    """Main function"""
    # ========================
    # Configuration parameters
    # ========================
    
    # Path configuration
    DATA_DIR = './data'
    OUTPUT_DIR = './output'
    
    # Data configuration
    YEARS_TRAIN = [2020, 2021, 2022, 2023]
    YEAR_VAL = 2024
    
    # Model configuration
    BASE_CHANNELS = 32
    N_CHANNELS = 1
    N_CLASSES = 1
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 200
    PATIENCE = 20
    NUM_WORKERS = 0
    WEIGHT_DECAY = 1e-5
    
    # Loss function configuration
    L1_WEIGHT = 1.0
    SSIM_WEIGHT = 0.1
    
    # Spatial weighted loss configuration (using historical RMSE weights)
    USE_WEIGHTED_LOSS = True  # Whether to use spatial weighted loss
    WEIGHT_MAP_PATH = './data/weight.npy'  # RMSE weight file path
    WEIGHT_MODE = 'inverse'  # Weight mode: 'inverse', 'normalized', 'squared'
    
    # Low RMSE region freezing configuration
    FREEZE_THRESHOLD = 0.01  # Regions with RMSE below this value will be frozen (use input directly)
    ENABLE_FREEZE = True  # Whether to enable freeze function
    
    # Learning rate scheduler configuration
    T_0 = 20
    T_mult = 2  # Changed to lowercase, consistent with parameter name below
    
    print("\n" + "="*60)
    print("Global tropospheric delay data deblurring - U-Net training")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================
    # Check data directory
    # ========================
    
    list_available_data(DATA_DIR)
    
    # ========================
    # Load data
    # ========================
    
    train_blur, train_clear, val_blur, val_clear, stats = \
        load_tropospheric_delay_data(DATA_DIR, YEARS_TRAIN, YEAR_VAL)
    
    global_min, global_max = stats
    
    # Check if validation data exists
    if val_blur is None or val_clear is None:
        print("\n⚠️  Validation data not found, splitting from training data...")
        train_blur, train_clear, val_blur, val_clear = \
            split_train_val(train_blur, train_clear, val_ratio=0.1)
    
    # ========================
    # Create dataset
    # ========================
    
    print("\nCreating dataset...")
    
    train_dataset = TroposphericDelayDataset(
        blur_data=train_blur,
        clear_data=train_clear,
        augmentation=True,
        shift_range=(-30, 30),
        global_min=global_min,
        global_max=global_max
    )
    
    val_dataset = TroposphericDelayDataset(
        blur_data=val_blur,
        clear_data=val_clear,
        augmentation=False,
        shift_range=None,
        global_min=global_min,
        global_max=global_max
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    # ========================
    # Create model
    # ========================
    
    print("\n" + "="*60)
    print("Create U-Net model")
    print("="*60)
    
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES,
                base_channels=BASE_CHANNELS,
                freeze_threshold=FREEZE_THRESHOLD,
                weight_map_path=WEIGHT_MAP_PATH if (ENABLE_FREEZE and os.path.exists(WEIGHT_MAP_PATH)) else None)
    print_model_info(model)
    
    # ========================
    # Train model
    # ========================
    
    # Check if weight file exists
    if USE_WEIGHTED_LOSS and os.path.exists(WEIGHT_MAP_PATH):
        print(f"\nUsing spatial weighted loss function")
        print(f"  Weight file: {WEIGHT_MAP_PATH}")
        print(f"  Weight mode: {WEIGHT_MODE}")
        print("  Principle: High RMSE regions (large blur-clear difference) high weight, low RMSE regions low weight")
    elif USE_WEIGHTED_LOSS:
        print(f"\nWarning: Weight file not found, will use standard loss function")
        print(f"  Search path: {WEIGHT_MAP_PATH}")
    
    # Check freeze function
    if ENABLE_FREEZE and hasattr(model, 'freeze_mask') and model.freeze_mask is not None:
        print(f"\nEnable low RMSE region freeze function")
        print(f"  Freeze threshold: RMSE < {FREEZE_THRESHOLD}")
        frozen_ratio = 1.0 - model.freeze_mask.mean().item()
        print(f"  Frozen region ratio: {frozen_ratio*100:.2f}%")
        print("  Frozen regions will use input directly without changes")
    elif ENABLE_FREEZE:
        print(f"\nWarning: Cannot load freeze mask, freeze function not enabled")
    
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
        l1_weight=L1_WEIGHT,
        ssim_weight=SSIM_WEIGHT,
        weight_decay=WEIGHT_DECAY,
        T_0=T_0,
        T_mult=T_mult,  # Unified use of T_mult
        global_max=global_max,
        use_weighted_loss=USE_WEIGHTED_LOSS,
        weight_map_path=WEIGHT_MAP_PATH if os.path.exists(WEIGHT_MAP_PATH) else None,
        weight_mode=WEIGHT_MODE,
        freeze_threshold=FREEZE_THRESHOLD,
        enable_freeze=ENABLE_FREEZE
    )
    
    history = trainer.train(output_dir=OUTPUT_DIR)
    
    # ========================
    # Save final model
    # ========================
    
    trainer.save_checkpoint(f'{OUTPUT_DIR}/final_model.pth', EPOCHS)
    np.save(f'{OUTPUT_DIR}/training_history.npy', history)
    print(f"Training history saved: {OUTPUT_DIR}/training_history.npy")
    
    # ========================
    # Final evaluation
    # ========================
    
    print("\n" + "="*60)
    print("Final model evaluation")
    print("="*60)
    
    trainer.load_checkpoint(f'{OUTPUT_DIR}/best_model.pth')
    val_loss, val_rmse, val_preds, val_targets = trainer.validate()
    
    print(f"\nValidation set final results:")
    print(f"  Validation loss: {val_loss:.6f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    
    val_preds_denorm = val_preds * global_max
    val_targets_denorm = val_targets * global_max
    
    metrics = TroposphericDelayMetrics.compute_all(
        val_preds_denorm.reshape(-1),
        val_targets_denorm.reshape(-1)
    )
    
    print(f"\nDetailed evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # ========================
    # Generate training report
    # ========================
    
    report_path = f'{OUTPUT_DIR}/training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Global tropospheric delay data deblurring U-Net model training report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Data configuration:\n")
        f.write(f"  Training years: {YEARS_TRAIN}\n")
        f.write(f"  Validation year: {YEAR_VAL}\n")
        f.write(f"  Training samples: {len(train_dataset)}\n")
        f.write(f"  Validation samples: {len(val_dataset)}\n\n")
        
        f.write("Model configuration:\n")
        f.write(f"  Base channels: {BASE_CHANNELS}\n")
        total_params, _ = count_parameters(model)
        f.write(f"  Total parameters: {total_params:,}\n\n")
        
        f.write("Training configuration:\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Training epochs: {EPOCHS}\n")
        f.write(f"  Early stopping patience: {PATIENCE}\n\n")
        
        f.write("Final results:\n")
        f.write(f"  Best validation loss: {trainer.best_val_loss:.6f}\n")
        f.write(f"  Final validation RMSE: {val_rmse:.4f}\n\n")
        
        f.write("Detailed metrics:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
    
    print(f"\nTraining report saved: {report_path}")
    
    return model, history


if __name__ == '__main__':
    model, history = main()
