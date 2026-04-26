import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np

def tune_xgboost(X_train, y_train, n_iter=5, cv=3):
    """Tunes XGBoost using RandomizedSearchCV focusing on PR-AUC."""
    print(f"Starting Hyperparameter Tuning (n_iter={n_iter}, cv={cv})...")
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=42,
        eval_metric='aucpr',
        n_jobs=-1
    )
    
    param_grid = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'n_estimators': [100, 200]
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='average_precision',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1
    )
    
    search.fit(X_train, y_train)
    
    print("\nBest Parameters found:", search.best_params_)
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    
    return search.best_estimator_
