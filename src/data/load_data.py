import pandas as pd
from pathlib import Path

def load_transactions(filepath: str) -> pd.DataFrame:
    """Loads the transactions CSV into a pandas DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Transactions data not found at {filepath}")
    return pd.read_csv(path)

def load_accounts(filepath: str) -> pd.DataFrame:
    """Loads the accounts CSV into a pandas DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Accounts data not found at {filepath}")
    return pd.read_csv(path)
