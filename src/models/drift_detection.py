import pandas as pd
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

def detect_data_drift(reference_data_path, current_data_path, p_value_threshold=0.05):
    """
    Compares numerical features between reference (training) data and current (live) data
    using the Kolmogorov-Smirnov (KS) test to detect Data Drift.
    """
    print(f"Loading reference data: {reference_data_path}")
    ref_df = pd.read_parquet(reference_data_path)
    
    print(f"Loading current data: {current_data_path}")
    curr_df = pd.read_parquet(current_data_path)
    
    numerical_cols = ['Amount Received', 'Amount Paid', 'Hour', 'DayOfWeek']
    
    drift_detected = False
    print("\n--- Data Drift Analysis (KS Test) ---")
    for col in numerical_cols:
        if col in ref_df.columns and col in curr_df.columns:
            # Drop NaNs
            ref_vals = ref_df[col].dropna()
            curr_vals = curr_df[col].dropna()
            
            # Perform KS test
            stat, p_value = ks_2samp(ref_vals, curr_vals)
            
            if p_value < p_value_threshold:
                print(f"[ALERT] Drift detected in '{col}' (p-value: {p_value:.4f})")
                drift_detected = True
            else:
                print(f"[OK] No drift in '{col}' (p-value: {p_value:.4f})")
                
    if drift_detected:
        print("\n=> WARNING: Significant Data Drift detected. Retraining is recommended.")
    else:
        print("\n=> STATUS: Data distribution is stable.")
        
    return drift_detected

if __name__ == "__main__":
    # Example mock usage: comparing test set to a sample of itself
    print("Running mock drift detection...")
    detect_data_drift('data/interim/X_test.parquet', 'data/interim/X_test.parquet')
