#!/usr/bin/env python3
"""
Metrics Definition

Contains various evaluation metrics:
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- Relative Bias: Relative Bias
- Correlation: Correlation Coefficient
- R²: Coefficient of Determination
- Spatial Gradient RMSE: Spatial Gradient RMSE
"""

import numpy as np


class TroposphericDelayMetrics:
    """Tropospheric delay data evaluation metrics"""
    
    @staticmethod
    def rmse(pred, target):
        """Root Mean Square Error"""
        return np.sqrt(np.mean((pred - target) ** 2))
    
    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def relative_bias(pred, target):
        """Relative Bias (percentage)"""
        return np.mean((pred - target) / (target + 1e-8) * 100)
    
    @staticmethod
    def correlation(pred, target):
        """Pearson correlation coefficient"""
        return np.corrcoef(pred.flatten(), target.flatten())[0, 1]
    
    @staticmethod
    def r2_score(pred, target):
        """Coefficient of Determination R²"""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def spatial_gradient_rmse(pred, target):
        """
        Spatial Gradient RMSE
        
        Only calculates when data is 2D (spatial data).
        If data is 1D (already flattened), returns NaN.
        """
        # Check data dimension
        pred = np.asarray(pred)
        target = np.asarray(target)
        
        if pred.ndim == 1:
            # Data has been flattened, skip spatial gradient calculation
            return float('nan')
        
        if pred.ndim != 2:
            # Unsupported dimension
            return float('nan')
        
        def compute_gradient(data):
            """Calculate spatial gradient"""
            grad_x = np.diff(data, axis=1, prepend=data[:, :1])
            grad_y = np.diff(data, axis=0, prepend=data[:1, :])
            return grad_x, grad_y
        
        pred_x, pred_y = compute_gradient(pred)
        target_x, target_y = compute_gradient(target)
        
        grad_rmse_x = np.sqrt(np.mean((pred_x - target_x) ** 2))
        grad_rmse_y = np.sqrt(np.mean((pred_y - target_y) ** 2))
        
        return (grad_rmse_x + grad_rmse_y) / 2
    
    @staticmethod
    def spatial_gradient_rmse_2d(pred_2d, target_2d):
        """
        Calculate spatial gradient RMSE for 2D data
        
        Args:
            pred_2d: Prediction data (H, W)
            target_2d: Target data (H, W)
            
        Returns:
            float: Spatial gradient RMSE
        """
        def compute_gradient(data):
            """Calculate spatial gradient"""
            grad_x = np.diff(data, axis=1, prepend=data[:, :1])
            grad_y = np.diff(data, axis=0, prepend=data[:1, :])
            return grad_x, grad_y
        
        pred_x, pred_y = compute_gradient(pred_2d)
        target_x, target_y = compute_gradient(target_2d)
        
        grad_rmse_x = np.sqrt(np.mean((pred_x - target_x) ** 2))
        grad_rmse_y = np.sqrt(np.mean((pred_y - target_y) ** 2))
        
        return (grad_rmse_x + grad_rmse_y) / 2
    
    @staticmethod
    def compute_all(pred, target):
        """
        Calculate all metrics
        
        Args:
            pred: Prediction (can be flattened or 2D)
            target: Target (can be flattened or 2D)
            
        Returns:
            dict: All metrics
        """
        pred = np.asarray(pred)
        target = np.asarray(target)
        
        # Save original shape
        original_shape = pred.shape
        
        # Check if data is 2D
        is_2d = pred.ndim == 2
        
        # Flatten data for scalar calculation if needed
        pred_flat = pred.flatten() if pred.ndim > 1 else pred
        target_flat = target.flatten() if target.ndim > 1 else target
        
        metrics = {
            'RMSE': TroposphericDelayMetrics.rmse(pred_flat, target_flat),
            'MAE': TroposphericDelayMetrics.mae(pred_flat, target_flat),
            'Relative Bias (%)': TroposphericDelayMetrics.relative_bias(pred_flat, target_flat),
            'Correlation': TroposphericDelayMetrics.correlation(pred_flat, target_flat),
            'R²': TroposphericDelayMetrics.r2_score(pred_flat, target_flat),
        }
        
        # If 2D data, calculate spatial gradient RMSE
        if is_2d:
            metrics['Gradient RMSE'] = TroposphericDelayMetrics.spatial_gradient_rmse_2d(pred, target)
        else:
            metrics['Gradient RMSE'] = float('nan')
        
        return metrics
    
    @staticmethod
    def compute_all_batch(pred_batch, target_batch):
        """
        Batch calculate all metrics
        
        Suitable for batch 2D data (N, H, W)
        
        Args:
            pred_batch: Prediction batch data (N, H, W)
            target_batch: Target batch data (N, H, W)
            
        Returns:
            dict: Average of all metrics
        """
        pred_batch = np.asarray(pred_batch)
        target_batch = np.asarray(target_batch)
        
        n_samples = len(pred_batch)
        
        # Flatten batch data
        pred_flat = pred_batch.reshape(n_samples, -1)
        target_flat = target_batch.reshape(n_samples, -1)
        
        # Calculate metrics for flattened data
        metrics = {
            'RMSE': np.mean([TroposphericDelayMetrics.rmse(pred_flat[i], target_flat[i]) 
                           for i in range(n_samples)]),
            'MAE': np.mean([TroposphericDelayMetrics.mae(pred_flat[i], target_flat[i]) 
                           for i in range(n_samples)]),
            'Relative Bias (%)': np.mean([TroposphericDelayMetrics.relative_bias(pred_flat[i], target_flat[i]) 
                                         for i in range(n_samples)]),
            'Correlation': np.mean([TroposphericDelayMetrics.correlation(pred_flat[i], target_flat[i]) 
                                   for i in range(n_samples)]),
            'R²': np.mean([TroposphericDelayMetrics.r2_score(pred_flat[i], target_flat[i]) 
                          for i in range(n_samples)]),
        }
        
        # Calculate spatial gradient RMSE (for first sample)
        if pred_batch.ndim == 3:
            metrics['Gradient RMSE'] = TroposphericDelayMetrics.spatial_gradient_rmse_2d(
                pred_batch[0], target_batch[0]
            )
        else:
            metrics['Gradient RMSE'] = float('nan')
        
        return metrics
