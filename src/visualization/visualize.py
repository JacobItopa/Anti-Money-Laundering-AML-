import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def plot_amount_distribution(df: pd.DataFrame, amount_col: str, target_col: str, save_path: str = None):
    """Plots a log-scaled distribution of an amount column separated by the target variable."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=amount_col, hue=target_col, bins=50, log_scale=True, palette='bright')
    plt.title(f'Distribution of {amount_col} (Log Scale)')
    plt.xlabel(f'{amount_col} (Log)')
    plt.ylabel('Count')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_categorical_distribution(df: pd.DataFrame, cat_col: str, target_col: str, save_path: str = None):
    """Plots counts of a categorical variable separated by the target variable."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=cat_col, hue=target_col, palette='bright')
    plt.title(f'Transactions by {cat_col}')
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.xticks(rotation=45)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
