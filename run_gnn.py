import pandas as pd
import joblib
import warnings
from sklearn.metrics import average_precision_score
from src.features.build_features import build_features_pipeline
from src.models.train_gnn import build_graph, train_gnn
import torch

warnings.filterwarnings('ignore')

def main():
    print("Loading raw subgraph (500k transactions)...")
    df_raw = pd.read_csv('data/raw/HI-Small_Trans.csv', nrows=500000)
    
    print("Preprocessing subgraph features...")
    df_processed = build_features_pipeline(df_raw.copy())
    
    edge_features_cols = [c for c in df_processed.columns if c != 'Is Laundering']
    
    df_graph = df_processed.copy()
    df_graph['Account'] = df_raw['Account'].values
    df_graph['Account.1'] = df_raw['Account.1'].values
    
    print("Building PyG Graph...")
    data = build_graph(df_graph, edge_features_cols)
    
    print("\n--- Training GNN ---")
    gnn_model, train_idx, test_idx = train_gnn(data, epochs=10)
    
    print("\n--- Comparative Evaluation (Test Set) ---")
    gnn_model.eval()
    with torch.no_grad():
        out = gnn_model(data.x, data.edge_index, data.edge_attr)
        gnn_probs = torch.sigmoid(out[test_idx]).cpu().numpy()
        y_true = data.y[test_idx].cpu().numpy()
        gnn_pr_auc = average_precision_score(y_true, gnn_probs)
        print(f"GNN PR-AUC: {gnn_pr_auc:.4f}")
        
    print("Evaluating XGBoost on the same subset...")
    try:
        xgb_model = joblib.load('models/advanced_xgboost.joblib')
        
        # Align columns
        X_df = df_processed[edge_features_cols]
        xgb_cols = xgb_model.get_booster().feature_names
        
        for c in xgb_cols:
            if c not in X_df.columns:
                X_df[c] = 0
                
        X_df = X_df[xgb_cols]
        X_test_xgb = X_df.iloc[test_idx]
        
        xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
        xgb_pr_auc = average_precision_score(y_true, xgb_probs)
        print(f"XGBoost PR-AUC: {xgb_pr_auc:.4f}")
        
    except Exception as e:
        print("Could not evaluate XGBoost:", e)

if __name__ == "__main__":
    main()
