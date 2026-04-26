import os
import json
from src.data.load_data import load_transactions
from src.data.explore import get_basic_stats, get_class_imbalance
from src.visualization.visualize import plot_amount_distribution, plot_categorical_distribution

def main():
    print("Loading data...")
    df = load_transactions('data/raw/HI-Small_Trans.csv')
    
    print("Computing stats...")
    stats = get_basic_stats(df)
    imbalance = get_class_imbalance(df, 'Is Laundering')
    
    # Save the stats to a JSON file for easy reading later
    results = {
        "stats": stats,
        "imbalance": imbalance
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/eda_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Generating plots...")
    os.makedirs('reports/figures', exist_ok=True)
    plot_amount_distribution(df, 'Amount Received', 'Is Laundering', 'reports/figures/amount_received_dist.png')
    plot_categorical_distribution(df, 'Payment Format', 'Is Laundering', 'reports/figures/payment_format_dist.png')
    
    print("EDA execution complete!")

if __name__ == "__main__":
    main()
