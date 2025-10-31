"""
Financial data processing and fixing utilities.
Handles data cleaning, normalization, and preprocessing for financial datasets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers using z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
    outlier_mask = z_scores < threshold
    
    # For each feature, replace outliers with median
    cleaned_data = data.copy()
    for i in range(data.shape[-1]):
        if len(data.shape) == 2:
            median_val = np.median(data[outlier_mask[:, i], i])
            cleaned_data[~outlier_mask[:, i], i] = median_val
        elif len(data.shape) == 3:
            for j in range(data.shape[0]):
                median_val = np.median(data[j, outlier_mask[j, :, i], i])
                cleaned_data[j, ~outlier_mask[j, :, i], i] = median_val
    
    return cleaned_data, outlier_mask


def handle_missing_values(data: np.ndarray, method: str = 'forward_fill') -> np.ndarray:
    """
    Handle missing values in financial data.
    
    Args:
        data: Input data array with potential NaN values
        method: Method to handle missing values ('forward_fill', 'interpolate', 'mean')
        
    Returns:
        Data with missing values handled
    """
    data = data.copy()
    
    if method == 'forward_fill':
        # Forward fill for time series
        if len(data.shape) == 3:
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    series = data[i, :, j]
                    mask = np.isnan(series)
                    if mask.any():
                        # Forward fill
                        idx = np.where(~mask, np.arange(len(mask)), 0)
                        np.maximum.accumulate(idx, out=idx)
                        data[i, :, j] = series[idx]
        else:
            # Handle 2D case
            for j in range(data.shape[1]):
                series = data[:, j]
                mask = np.isnan(series)
                if mask.any():
                    idx = np.where(~mask, np.arange(len(mask)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    data[:, j] = series[idx]
                    
    elif method == 'interpolate':
        # Linear interpolation
        if len(data.shape) == 3:
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    series = data[i, :, j]
                    mask = np.isnan(series)
                    if mask.any():
                        x = np.arange(len(series))
                        data[i, :, j] = np.interp(x, x[~mask], series[~mask])
        else:
            for j in range(data.shape[1]):
                series = data[:, j]
                mask = np.isnan(series)
                if mask.any():
                    x = np.arange(len(series))
                    data[:, j] = np.interp(x, x[~mask], series[~mask])
                    
    elif method == 'mean':
        # Replace with mean
        if len(data.shape) == 3:
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    series = data[i, :, j]
                    mean_val = np.nanmean(series)
                    data[i, np.isnan(data[i, :, j]), j] = mean_val
        else:
            for j in range(data.shape[1]):
                mean_val = np.nanmean(data[:, j])
                data[np.isnan(data[:, j]), j] = mean_val
    
    # Replace any remaining NaNs with 0
    data = np.nan_to_num(data, nan=0.0)
    
    return data


def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize financial data.
    
    Args:
        data: Input data array
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    params = {}
    
    if method == 'standard':
        # Standardization (z-score normalization)
        mean = np.mean(data, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        std = np.std(data, axis=tuple(range(len(data.shape)-1)), keepdims=True) + 1e-8
        normalized_data = (data - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standard'}
        
    elif method == 'minmax':
        # Min-Max normalization
        min_val = np.min(data, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        max_val = np.max(data, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'robust':
        # Robust normalization (using median and IQR)
        median = np.median(data, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        q75 = np.percentile(data, 75, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        q25 = np.percentile(data, 25, axis=tuple(range(len(data.shape)-1)), keepdims=True)
        iqr = q75 - q25 + 1e-8
        normalized_data = (data - median) / iqr
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params


def calculate_technical_indicators(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Calculate technical indicators for financial data.
    
    Args:
        prices: Price data (N, sequence_length)
        window: Window size for indicators
        
    Returns:
        Technical indicators array
    """
    n_samples, seq_len = prices.shape
    indicators = np.zeros((n_samples, seq_len, 3))  # 3 indicators
    
    for i in range(n_samples):
        price_series = prices[i]
        
        # Simple Moving Average (SMA)
        sma = np.convolve(price_series, np.ones(window)/window, mode='same')
        indicators[i, :, 0] = sma
        
        # Rate of Change (ROC)
        roc = np.zeros_like(price_series)
        roc[window:] = (price_series[window:] - price_series[:-window]) / (price_series[:-window] + 1e-8)
        indicators[i, :, 1] = roc
        
        # Volatility (rolling standard deviation)
        volatility = np.zeros_like(price_series)
        for j in range(window, seq_len):
            volatility[j] = np.std(price_series[j-window:j])
        indicators[i, :, 2] = volatility
    
    return indicators


def fix_financial_data(time_series: np.ndarray, graph_features: np.ndarray,
                      targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline to fix and preprocess financial data.
    
    Args:
        time_series: Raw time series data
        graph_features: Raw graph features
        targets: Raw target values
        
    Returns:
        Tuple of (fixed_time_series, fixed_graph_features, fixed_targets)
    """
    # Handle missing values
    time_series = handle_missing_values(time_series, method='forward_fill')
    graph_features = handle_missing_values(graph_features, method='mean')
    
    # Remove outliers
    time_series, _ = remove_outliers(time_series, threshold=3.0)
    graph_features, _ = remove_outliers(graph_features, threshold=3.0)
    
    # Normalize data
    time_series, _ = normalize_data(time_series, method='standard')
    graph_features, _ = normalize_data(graph_features, method='standard')
    targets, _ = normalize_data(targets.reshape(-1, 1), method='minmax')
    targets = targets.flatten()
    
    return time_series, graph_features, targets
