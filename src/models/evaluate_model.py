import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, confusion_matrix
import os

def evaluate_model_metrics(model, X_test, y_test, model_name="Model"):
    """Calculates PR-AUC and prints the classification report."""
    print(f"\n--- Evaluating {model_name} ---")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, probs)
    print(f"PR-AUC: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    return probs, preds, pr_auc

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_dir="reports/figures"):
    """Plots and saves the confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Illicit'], 
                yticklabels=['Normal', 'Illicit'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filepath = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}_cm.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {filepath}")

def plot_pr_curve(y_true, y_probs_dict, save_dir="reports/figures"):
    """Plots PR Curves for multiple models."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for model_name, probs in y_probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        plt.plot(recall, precision, lw=2, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="upper right")
    plt.grid(True)
    
    filepath = os.path.join(save_dir, "pr_curve_comparison.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved PR Curve comparison to {filepath}")
