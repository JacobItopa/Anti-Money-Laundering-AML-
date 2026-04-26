import pandas as pd
import joblib
from src.models.evaluate_model import evaluate_model_metrics, plot_confusion_matrix, plot_pr_curve
from src.models.tune_model import tune_xgboost

def main():
    print("Loading test data...")
    X_test = pd.read_parquet('data/interim/X_test.parquet')
    y_test = pd.read_parquet('data/interim/y_test.parquet')['Is Laundering']
    
    print("Loading a subset of training data for hyperparameter tuning...")
    df = pd.read_parquet('data/processed/processed_transactions.parquet')
    df_sample = df.sample(n=500000, random_state=42)
    X_train_sample = df_sample.drop(columns=['Is Laundering'])
    y_train_sample = df_sample['Is Laundering']
    
    print("Loading Base XGBoost Model...")
    base_xgb = joblib.load('models/advanced_xgboost.joblib')
    
    # 1. Evaluate Base Model
    base_probs, base_preds, base_auc = evaluate_model_metrics(base_xgb, X_test, y_test, "Base XGBoost")
    plot_confusion_matrix(y_test, base_preds, "Base XGBoost")
    
    # 2. Hyperparameter Tuning
    tuned_xgb = tune_xgboost(X_train_sample, y_train_sample, n_iter=5, cv=3)
    joblib.dump(tuned_xgb, 'models/tuned_xgboost.joblib')
    
    # 3. Evaluate Tuned Model
    tuned_probs, tuned_preds, tuned_auc = evaluate_model_metrics(tuned_xgb, X_test, y_test, "Tuned XGBoost")
    plot_confusion_matrix(y_test, tuned_preds, "Tuned XGBoost")
    
    # 4. Comparative PR Curve
    prob_dict = {
        "Base XGBoost": base_probs,
        "Tuned XGBoost": tuned_probs
    }
    plot_pr_curve(y_test, prob_dict)
    print("Evaluation and Tuning Complete!")

if __name__ == "__main__":
    main()
