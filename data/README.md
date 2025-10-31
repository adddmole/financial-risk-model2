# Data Directory

This directory contains financial data for the risk modeling system.

## Expected Data Format

The system expects CSV files with the following structure:

### Required Columns

- **entity_id**: Unique identifier for each financial entity (company, asset, etc.)
- **time_step**: Time step index (0, 1, 2, ...) for temporal ordering
- **feature_1, feature_2, ..., feature_n**: Temporal features (e.g., price, returns, volume, volatility)
- **label**: Risk label (0: low risk, 1: high risk) - optional for training data

### Optional Columns

- **node_feature_1, node_feature_2, ..., node_feature_m**: Graph node features (e.g., sector, size, financial ratios)
- **price**: Asset price (used for technical indicator calculation)
- **volume**: Trading volume
- **returns**: Returns
- **volatility**: Volatility measure

## Data Files

### Sample Data
- `sample_data.csv` - Example dataset with synthetic data for testing the pipeline

### Processed Data
- `processed_data.csv` - Output from the preprocessing pipeline (generated)

### Custom Data
Place your custom financial data files here following the format above.

## Data Preprocessing

To preprocess your financial data, run:

```bash
python main.py --mode preprocess --data-path ./data/your_data.csv --output-path ./data/processed_data.csv
```

This will:
1. Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Engineer features from price data
3. Create risk labels based on volatility or returns
4. Handle missing data
5. Detect and mark outliers
6. Save the processed data

## Graph Construction

The system supports multiple methods for constructing the entity relationship graph:

1. **Correlation-based**: Creates edges between entities with correlated features
   - Set `--graph-method correlation --correlation-threshold 0.7`

2. **K-Nearest Neighbors**: Creates edges to K most similar entities
   - Set `--graph-method knn --knn-k 10`

3. **Predefined**: Load edges from a separate file
   - Create a file `edges.csv` with columns: `source, target, weight`
   - Set `--graph-method predefined`

## Example Data Structure

```csv
entity_id,time_step,feature_1,feature_2,feature_3,price,volume,label
0,0,0.5,1.2,-0.3,100.0,1000000,0
0,1,0.6,1.1,-0.2,101.5,1100000,0
0,2,0.4,1.3,-0.4,99.8,950000,0
...
1,0,-0.2,0.8,0.5,50.0,500000,1
1,1,-0.3,0.9,0.4,49.5,480000,1
1,2,-0.1,0.7,0.6,51.2,520000,1
...
```

## Adding Custom Data

1. Prepare your data in the format described above
2. Place the CSV file in this directory
3. Run preprocessing if needed
4. Update the config.py file if your data has different dimensions
5. Run training with `--data-path ./data/your_data.csv`

## Notes

- Each entity should have consistent time steps
- Missing values will be handled by the preprocessing pipeline
- Feature scaling is performed automatically
- For large datasets, consider using chunked processing
