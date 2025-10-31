"""
Generate synthetic financial data for testing the pipeline.
"""
import numpy as np
import pandas as pd


def generate_sample_data(num_entities: int = 50,
                        sequence_length: int = 60,
                        num_features: int = 10,
                        output_path: str = 'sample_data.csv'):
    """
    Generate synthetic financial time series data.
    
    Args:
        num_entities: Number of financial entities
        sequence_length: Number of time steps per entity
        num_features: Number of temporal features
        output_path: Path to save the data
    """
    np.random.seed(42)
    
    data = []
    
    for entity_id in range(num_entities):
        # Determine risk level for this entity
        is_high_risk = np.random.rand() > 0.6  # 40% high risk
        
        # Generate price trajectory
        if is_high_risk:
            # High risk: higher volatility, potential downward trend
            drift = -0.001
            volatility = 0.05
        else:
            # Low risk: lower volatility, stable or upward trend
            drift = 0.001
            volatility = 0.02
        
        price = 100.0
        prices = []
        
        for t in range(sequence_length):
            # Geometric Brownian motion
            price *= (1 + drift + volatility * np.random.randn())
            prices.append(price)
        
        prices = np.array(prices)
        
        # Calculate returns
        returns = np.diff(prices, prepend=prices[0]) / prices
        
        # Calculate volatility
        volatility_series = pd.Series(returns).rolling(window=10, min_periods=1).std().values
        
        # Generate other features
        for t in range(sequence_length):
            row = {
                'entity_id': entity_id,
                'time_step': t,
                'price': prices[t],
                'returns': returns[t],
                'volatility': volatility_series[t],
            }
            
            # Add random features (correlated with risk)
            base_signal = 1.0 if is_high_risk else -1.0
            for i in range(1, num_features - 2):  # -2 because we already have returns and volatility
                feature_value = base_signal * np.random.rand() + np.random.randn() * 0.5
                row[f'feature_{i}'] = feature_value
            
            # Add label (same for all time steps of an entity)
            row['label'] = 1 if is_high_risk else 0
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some node-level features (aggregated from time series)
    node_features = []
    for entity_id in range(num_entities):
        entity_data = df[df['entity_id'] == entity_id]
        node_feat = {
            'entity_id': entity_id,
            'node_feature_1': entity_data['returns'].mean(),
            'node_feature_2': entity_data['volatility'].mean(),
            'node_feature_3': entity_data['price'].mean(),
            'node_feature_4': entity_data['returns'].std(),
            'node_feature_5': entity_data['price'].max() - entity_data['price'].min(),
        }
        node_features.append(node_feat)
    
    node_df = pd.DataFrame(node_features)
    
    # Merge with main data
    df = df.merge(node_df, on='entity_id', how='left')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated sample data with {num_entities} entities and {sequence_length} time steps")
    print(f"Total rows: {len(df)}")
    print(f"Label distribution:\n{df.groupby('entity_id')['label'].first().value_counts()}")
    print(f"Saved to: {output_path}")
    
    return df


if __name__ == '__main__':
    generate_sample_data()
