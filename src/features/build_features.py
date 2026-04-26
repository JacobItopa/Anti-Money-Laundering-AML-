import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicates and handles missing values."""
    print("Cleaning data...")
    df = df.drop_duplicates().copy()
    return df

def engineer_temporal_features(df: pd.DataFrame, time_col: str = 'Timestamp') -> pd.DataFrame:
    """Extracts Hour, DayOfWeek, Month from a timestamp string."""
    print("Engineering temporal features...")
    df[time_col] = pd.to_datetime(df[time_col])
    df['Hour'] = df[time_col].dt.hour
    df['DayOfWeek'] = df[time_col].dt.dayofweek
    df['Month'] = df[time_col].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Applies log1p transformation to specified columns to handle extreme skew."""
    print("Scaling numerical features...")
    for col in columns:
        df[col] = np.log1p(df[col])
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Frequency Encoding to high cardinality, One-Hot to low cardinality."""
    print("Encoding categorical features...")
    
    # Low Cardinality: One-Hot Encode 'Payment Format'
    df = pd.get_dummies(df, columns=['Payment Format'], drop_first=True)
    
    # High Cardinality: Frequency Encode Banks & Currencies
    high_card_cols = ['Receiving Currency', 'Payment Currency', 'From Bank', 'To Bank']
    for col in high_card_cols:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_Freq'] = df[col].map(freq)
        
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drops raw text and redundant features to prepare for ML."""
    print("Selecting final features...")
    cols_to_drop = ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'From Bank', 'To Bank']
    # Ensure columns exist before dropping to avoid errors
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df

def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executes the full preprocessing pipeline."""
    df = clean_data(df)
    df = engineer_temporal_features(df, 'Timestamp')
    df = scale_features(df, ['Amount Received', 'Amount Paid'])
    df = encode_categorical_features(df)
    df = select_features(df)
    return df
