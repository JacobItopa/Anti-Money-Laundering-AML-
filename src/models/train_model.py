import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Splits data into train and test sets, preserving class proportions (stratified)."""
    print(f"Splitting data... (Target: {target_col})")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set shape: {X_train.shape}, Illicit count: {y_train.sum()}")
    print(f"Test set shape: {X_test.shape}, Illicit count: {y_test.sum()}")
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, y_train):
    """Trains a Logistic Regression baseline model with class weighting."""
    print("Training Baseline Model (Logistic Regression)...")
    # 'balanced' automatically adjusts weights inversely proportional to class frequencies
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost_model(X_train, y_train):
    """Trains an optimized XGBoost model for severe class imbalance."""
    print("Training Advanced Model (XGBoost)...")
    
    # Calculate scale_pos_weight
    # ratio = count(negative examples) / count(positive examples)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Use tree_method='hist' for massive speedup on large datasets
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath: str):
    """Serializes the trained model to disk."""
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
