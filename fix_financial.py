"""
Data quality and fixing utilities for financial data
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class FinancialDataFixer:
    """
    Class for fixing and cleaning financial data
    """
    
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.outlier_bounds = {}
    
    def detect_missing_values(self, data):
        """
        Detect missing values in data
        
        Args:
            data: Input data (numpy array or pandas DataFrame)
        
        Returns:
            Dictionary with missing value statistics
        """
        if isinstance(data, pd.DataFrame):
            missing_count = data.isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            return {
                'missing_count': missing_count.to_dict(),
                'missing_percentage': missing_pct.to_dict()
            }
        else:
            missing_count = np.isnan(data).sum(axis=0)
            missing_pct = (missing_count / data.shape[0]) * 100
            return {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }
    
    def fix_missing_values(self, data, strategy='mean'):
        """
        Fix missing values using imputation
        
        Args:
            data: Input data with missing values
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
        Returns:
            Data with missing values imputed
        """
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy=strategy)
            data_fixed = self.imputer.fit_transform(data)
        else:
            data_fixed = self.imputer.transform(data)
        
        return data_fixed
    
    def detect_outliers(self, data, method='iqr', threshold=3):
        """
        Detect outliers in data
        
        Args:
            data: Input data
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            Boolean mask of outliers
        """
        if method == 'iqr':
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            self.outlier_bounds = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-8))
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    
    def fix_outliers(self, data, outliers, method='clip'):
        """
        Fix outliers in data
        
        Args:
            data: Input data
            outliers: Boolean mask of outliers
            method: Fixing method ('clip', 'remove', 'winsorize')
        
        Returns:
            Data with outliers fixed
        """
        data_fixed = data.copy()
        
        if method == 'clip':
            # Clip to bounds
            if self.outlier_bounds:
                lower = self.outlier_bounds['lower']
                upper = self.outlier_bounds['upper']
                data_fixed = np.clip(data_fixed, lower, upper)
        
        elif method == 'winsorize':
            # Replace with percentile values
            for col in range(data.shape[-1]):
                p5 = np.percentile(data[..., col], 5)
                p95 = np.percentile(data[..., col], 95)
                data_fixed[..., col] = np.clip(data_fixed[..., col], p5, p95)
        
        elif method == 'remove':
            # This requires returning indices to keep
            valid_mask = ~outliers.any(axis=-1)
            data_fixed = data[valid_mask]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data_fixed
    
    def check_data_quality(self, data):
        """
        Comprehensive data quality check
        
        Args:
            data: Input data
        
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {}
        
        # Check for missing values
        missing_stats = self.detect_missing_values(data)
        quality_report['missing_values'] = missing_stats
        
        # Check for outliers
        outliers = self.detect_outliers(data)
        outlier_count = outliers.sum(axis=0)
        outlier_pct = (outlier_count / data.shape[0]) * 100
        quality_report['outliers'] = {
            'count': outlier_count,
            'percentage': outlier_pct
        }
        
        # Check for constant features
        if data.ndim == 2:
            variance = np.var(data, axis=0)
            constant_features = variance < 1e-8
            quality_report['constant_features'] = {
                'count': constant_features.sum(),
                'indices': np.where(constant_features)[0].tolist()
            }
        
        # Check for infinite values
        inf_count = np.isinf(data).sum(axis=0)
        quality_report['infinite_values'] = {
            'count': inf_count
        }
        
        return quality_report
    
    def fix_all(self, data, fix_missing=True, fix_outliers_flag=True, 
                remove_constant=False):
        """
        Apply all fixes to data
        
        Args:
            data: Input data
            fix_missing: Whether to fix missing values
            fix_outliers_flag: Whether to fix outliers
            remove_constant: Whether to remove constant features
        
        Returns:
            Fixed data and quality report
        """
        print("Checking data quality...")
        quality_report = self.check_data_quality(data)
        
        data_fixed = data.copy()
        
        # Fix infinite values
        data_fixed = np.nan_to_num(data_fixed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fix missing values
        if fix_missing:
            print("Fixing missing values...")
            data_fixed = self.fix_missing_values(data_fixed, strategy='median')
        
        # Fix outliers
        if fix_outliers_flag:
            print("Fixing outliers...")
            outliers = self.detect_outliers(data_fixed)
            data_fixed = self.fix_outliers(data_fixed, outliers, method='clip')
        
        # Remove constant features
        if remove_constant and data_fixed.ndim == 2:
            variance = np.var(data_fixed, axis=0)
            constant_features = variance < 1e-8
            if constant_features.any():
                print(f"Removing {constant_features.sum()} constant features...")
                data_fixed = data_fixed[:, ~constant_features]
        
        print("Data quality fixes applied successfully!")
        
        return data_fixed, quality_report


def validate_financial_data(data, labels=None):
    """
    Validate financial data for model training
    
    Args:
        data: Input features
        labels: Target labels (optional)
    
    Returns:
        Boolean indicating if data is valid
    """
    # Check for NaN or Inf
    if np.isnan(data).any() or np.isinf(data).any():
        print("Warning: Data contains NaN or Inf values")
        return False
    
    # Check shape
    if data.ndim < 2:
        print("Warning: Data should be at least 2D")
        return False
    
    # Check labels if provided
    if labels is not None:
        if np.isnan(labels).any() or np.isinf(labels).any():
            print("Warning: Labels contain NaN or Inf values")
            return False
        
        if len(data) != len(labels):
            print("Warning: Data and labels have different lengths")
            return False
    
    print("Data validation passed!")
    return True


def normalize_financial_features(data, method='robust'):
    """
    Normalize financial features
    
    Args:
        data: Input data
        method: Normalization method ('robust', 'standard', 'minmax')
    
    Returns:
        Normalized data
    """
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    data_normalized = scaler.fit_transform(data_2d)
    data_normalized = data_normalized.reshape(original_shape)
    
    return data_normalized, scaler


if __name__ == '__main__':
    # Example usage
    print("Testing Financial Data Fixer...")
    
    # Create sample data with issues
    sample_data = np.random.randn(100, 10)
    sample_data[10:15, 2] = np.nan  # Missing values
    sample_data[20, 5] = 100  # Outlier
    sample_data[:, 7] = 1  # Constant feature
    
    # Fix data
    fixer = FinancialDataFixer()
    fixed_data, report = fixer.fix_all(sample_data, remove_constant=True)
    
    print("\nQuality Report:")
    print(f"Original shape: {sample_data.shape}")
    print(f"Fixed shape: {fixed_data.shape}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Outliers: {report['outliers']}")
    print(f"Constant features: {report['constant_features']}")
    
    # Validate
    is_valid = validate_financial_data(fixed_data)
    print(f"Data is valid: {is_valid}")
