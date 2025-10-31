"""
Financial data preprocessing and feature engineering.
Handles technical indicators, risk labels, and data quality checks.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change(periods=periods)


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(periods))


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std()


def calculate_moving_average(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate simple moving average."""
    return prices.rolling(window=window).mean()


def calculate_ema(prices: pd.Series, span: int = 20) -> pd.Series:
    """Calculate exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()


def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        periods: RSI period
        
    Returns:
        RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, 
                              window: int = 20, 
                              num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_financial_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common financial ratios if fundamental data is available.
    
    Expected columns: price, earnings, book_value, debt, equity, revenue, etc.
    """
    ratios = pd.DataFrame(index=data.index)
    
    # Price-to-Earnings Ratio
    if 'price' in data.columns and 'earnings' in data.columns:
        ratios['pe_ratio'] = data['price'] / data['earnings'].replace(0, np.nan)
    
    # Price-to-Book Ratio
    if 'price' in data.columns and 'book_value' in data.columns:
        ratios['pb_ratio'] = data['price'] / data['book_value'].replace(0, np.nan)
    
    # Debt-to-Equity Ratio
    if 'debt' in data.columns and 'equity' in data.columns:
        ratios['debt_equity'] = data['debt'] / data['equity'].replace(0, np.nan)
    
    # Return on Equity
    if 'net_income' in data.columns and 'equity' in data.columns:
        ratios['roe'] = data['net_income'] / data['equity'].replace(0, np.nan)
    
    # Return on Assets
    if 'net_income' in data.columns and 'assets' in data.columns:
        ratios['roa'] = data['net_income'] / data['assets'].replace(0, np.nan)
    
    return ratios


def engineer_technical_features(data: pd.DataFrame, 
                                price_col: str = 'price') -> pd.DataFrame:
    """
    Engineer technical indicators from price data.
    
    Args:
        data: DataFrame with price data
        price_col: Name of price column
        
    Returns:
        DataFrame with additional technical features
    """
    df = data.copy()
    prices = df[price_col]
    
    # Returns
    df['returns_1d'] = calculate_returns(prices, 1)
    df['returns_5d'] = calculate_returns(prices, 5)
    df['returns_20d'] = calculate_returns(prices, 20)
    
    # Log returns
    df['log_returns'] = calculate_log_returns(prices, 1)
    
    # Volatility
    df['volatility_10d'] = calculate_volatility(df['returns_1d'], 10)
    df['volatility_20d'] = calculate_volatility(df['returns_1d'], 20)
    
    # Moving averages
    df['sma_10'] = calculate_moving_average(prices, 10)
    df['sma_20'] = calculate_moving_average(prices, 20)
    df['sma_50'] = calculate_moving_average(prices, 50)
    df['ema_10'] = calculate_ema(prices, 10)
    df['ema_20'] = calculate_ema(prices, 20)
    
    # Price relative to moving averages
    df['price_sma_10_ratio'] = prices / df['sma_10']
    df['price_sma_20_ratio'] = prices / df['sma_20']
    
    # RSI
    df['rsi'] = calculate_rsi(prices, 14)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(prices)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
    
    return df


def create_risk_labels(data: pd.DataFrame,
                      method: str = 'volatility',
                      threshold: float = None) -> pd.Series:
    """
    Create risk labels based on different methods.
    
    Args:
        data: DataFrame with financial data
        method: Labeling method ('volatility', 'returns', 'default', 'custom')
        threshold: Threshold for binary classification
        
    Returns:
        Series of risk labels (0: low risk, 1: high risk)
    """
    if method == 'volatility':
        # Label based on volatility
        if 'volatility_20d' in data.columns:
            volatility = data['volatility_20d']
        elif 'returns_1d' in data.columns:
            volatility = data['returns_1d'].rolling(20).std()
        else:
            raise ValueError("No volatility data available")
        
        if threshold is None:
            threshold = volatility.median()
        
        labels = (volatility > threshold).astype(int)
    
    elif method == 'returns':
        # Label based on negative returns
        if 'returns_20d' in data.columns:
            returns = data['returns_20d']
        elif 'returns_1d' in data.columns:
            returns = data['returns_1d']
        else:
            raise ValueError("No returns data available")
        
        if threshold is None:
            threshold = 0
        
        labels = (returns < threshold).astype(int)
    
    elif method == 'default':
        # Use existing default/credit rating labels
        if 'default' in data.columns:
            labels = data['default']
        elif 'credit_rating' in data.columns:
            # Convert credit ratings to binary (below investment grade = high risk)
            labels = (data['credit_rating'] < 3).astype(int)  # Assuming numeric ratings
        else:
            raise ValueError("No default/credit rating data available")
    
    elif method == 'custom':
        # Use existing label column
        if 'label' in data.columns:
            labels = data['label']
        else:
            raise ValueError("No custom label column found")
    
    else:
        raise ValueError(f"Unknown labeling method: {method}")
    
    return labels


def handle_missing_data(data: pd.DataFrame, 
                       method: str = 'ffill',
                       fill_value: float = 0) -> pd.DataFrame:
    """
    Handle missing data in the dataset.
    
    Args:
        data: DataFrame with missing values
        method: Method to handle missing data ('ffill', 'bfill', 'interpolate', 'drop', 'fill')
        fill_value: Value to use for 'fill' method
        
    Returns:
        DataFrame with missing data handled
    """
    df = data.copy()
    
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
    elif method == 'drop':
        df = df.dropna()
    elif method == 'fill':
        df = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    return df


def detect_outliers(data: pd.DataFrame, 
                   columns: List[str],
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect and handle outliers in the data.
    
    Args:
        data: DataFrame
        columns: Columns to check for outliers
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers marked or removed
    """
    df = data.copy()
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask |= z_scores > threshold
    
    # Add outlier indicator column
    df['is_outlier'] = outlier_mask
    
    return df


def create_time_windows(data: pd.DataFrame,
                       window_size: int = 60,
                       stride: int = 1) -> List[pd.DataFrame]:
    """
    Create sliding time windows from data.
    
    Args:
        data: DataFrame with time series data
        window_size: Size of each window
        stride: Stride between windows
        
    Returns:
        List of DataFrame windows
    """
    windows = []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data.iloc[i:i + window_size]
        windows.append(window)
    
    return windows


def normalize_features(data: pd.DataFrame,
                      columns: List[str],
                      method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Normalize features in the dataset.
    
    Args:
        data: DataFrame
        columns: Columns to normalize
        method: Normalization method ('standard', 'minmax')
        
    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    df = data.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Select columns to normalize
    cols_to_normalize = [col for col in columns if col in df.columns]
    
    if len(cols_to_normalize) > 0:
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return df, scaler


def preprocess_financial_data(data_path: str,
                             output_path: str,
                             price_col: str = 'price',
                             entity_col: str = 'entity_id',
                             time_col: str = 'time_step',
                             label_method: str = 'volatility') -> None:
    """
    Complete preprocessing pipeline for financial data.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path to save processed data
        price_col: Name of price column
        entity_col: Name of entity ID column
        time_col: Name of time step column
        label_method: Method to create risk labels
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    print("Engineering technical features...")
    if price_col in data.columns:
        data = engineer_technical_features(data, price_col)
    
    print("Creating risk labels...")
    data['label'] = create_risk_labels(data, method=label_method)
    
    print("Handling missing data...")
    data = handle_missing_data(data, method='ffill')
    
    print("Detecting outliers...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [entity_col, time_col, 'label']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    data = detect_outliers(data, numeric_cols, method='iqr', threshold=3.0)
    
    print(f"Saving processed data to {output_path}...")
    data.to_csv(output_path, index=False)
    
    print("Preprocessing complete!")
    print(f"Final data shape: {data.shape}")
    print(f"Risk label distribution:\n{data['label'].value_counts()}")
