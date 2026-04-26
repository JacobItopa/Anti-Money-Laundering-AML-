import pandas as pd
from typing import Dict, Any

def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns basic shape and descriptive statistics for a dataframe."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

def get_class_imbalance(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Calculates class imbalance ratio for a specific target column."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    counts = df[target_col].value_counts().to_dict()
    ratio = None
    if 0 in counts and 1 in counts:
        ratio = counts[0] / counts[1]
    
    return {
        "counts": counts,
        "ratio_majority_to_minority": ratio
    }
