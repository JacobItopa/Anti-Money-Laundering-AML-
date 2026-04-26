import pandas as pd
import joblib

def main():
    print("Loading raw data to extract frequency weights...")
    df = pd.read_csv('data/raw/HI-Small_Trans.csv')
    
    # Calculate frequency maps
    freq_maps = {}
    high_card_cols = ['Receiving Currency', 'Payment Currency', 'From Bank', 'To Bank']
    for col in high_card_cols:
        freq = df[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq
    
    # Load model to get exact expected columns
    xgb_model = joblib.load('models/advanced_xgboost.joblib')
    expected_cols = xgb_model.get_booster().feature_names
    
    artifacts = {
        'freq_maps': freq_maps,
        'expected_cols': expected_cols
    }
    
    joblib.dump(artifacts, 'models/preprocessing_artifacts.joblib')
    print("Saved preprocessing artifacts to models/preprocessing_artifacts.joblib")

if __name__ == "__main__":
    main()
