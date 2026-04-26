import os
import pandas as pd
from src.data.load_data import load_transactions
from src.features.build_features import build_features_pipeline

def main():
    print("Loading raw data...")
    # Load entire dataset
    df = load_transactions('data/raw/HI-Small_Trans.csv')
    
    print("Running preprocessing pipeline...")
    processed_df = build_features_pipeline(df)
    
    print("Saving processed data to Parquet...")
    os.makedirs('data/processed', exist_ok=True)
    processed_path = 'data/processed/processed_transactions.parquet'
    
    # Parquet is highly efficient and preserves datatypes
    processed_df.to_parquet(processed_path, index=False)
    
    print(f"Data successfully processed and saved to {processed_path}")
    print(f"Final shape: {processed_df.shape}")
    print("Final columns:", processed_df.columns.tolist())

if __name__ == "__main__":
    main()
