import os
import pandas as pd
from src.models.train_model import split_data, train_baseline_model, train_xgboost_model, save_model

def main():
    print("Loading processed data...")
    df = pd.read_parquet('data/processed/processed_transactions.parquet')
    
    # Step 1: Data Splitting
    X_train, X_test, y_train, y_test = split_data(df, target_col='Is Laundering')
    
    # Step 2: Establish a Baseline & Train
    baseline_model = train_baseline_model(X_train, y_train)
    
    # Step 3: Algorithm Selection & Train Advanced Model
    xgb_model = train_xgboost_model(X_train, y_train)
    
    # Step 4: Model Serialization
    os.makedirs('models', exist_ok=True)
    save_model(baseline_model, 'models/baseline_logreg.joblib')
    save_model(xgb_model, 'models/advanced_xgboost.joblib')
    
    # We will save the test sets to disk so Step 6 (Evaluation) can use them easily
    os.makedirs('data/interim', exist_ok=True)
    X_test.to_parquet('data/interim/X_test.parquet', index=False)
    pd.DataFrame(y_test, columns=['Is Laundering']).to_parquet('data/interim/y_test.parquet', index=False)
    print("Saved test sets to data/interim/ for evaluation.")
    
    print("Training pipeline complete!")

if __name__ == "__main__":
    main()
