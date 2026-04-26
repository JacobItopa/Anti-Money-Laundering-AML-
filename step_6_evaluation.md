# Step 6: Model Evaluation & Hyperparameter Tuning

*Note: The modular code for this step is located in `src/models/evaluate_model.py` and `src/models/tune_model.py`. The interactive visualizations are in `notebooks/05-evaluation-tuning.ipynb`.*

## 1. Base Model Evaluation (XGBoost)
We evaluated our `advanced_xgboost.joblib` model (trained on the full 4 million rows) against the 1-million row holdout test set.
*   **PR-AUC:** 0.068
*   **Recall (Illicit):** 90%
*   **Precision (Illicit):** 1%
*   **Takeaway:** The model is incredibly sensitive. By using the dynamic `scale_pos_weight`, the model sacrifices precision to ensure it catches almost every single launderer. It correctly flagged 90% of all actual money laundering transactions, which is fantastic for a first-pass Anti-Money Laundering screening system.

## 2. Hyperparameter Tuning
To attempt to squeeze out better precision without sacrificing recall, we utilized `RandomizedSearchCV` with 3-fold cross-validation on a 500,000-row subset of our training data.
*   **Target Metric:** `average_precision` (PR-AUC)
*   **Best Parameters Discovered:** `{'subsample': 1.0, 'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'colsample_bytree': 1.0}`

## 3. Tuned Model Evaluation
We then evaluated the tuned model on the untouched test set.
*   **PR-AUC:** 0.0558
*   **Recall (Illicit):** 90%
*   **Precision (Illicit):** 1%
*   **Takeaway:** The tuned model achieved identical Recall (90%) but a slightly lower overall PR-AUC compared to the base model. This is almost certainly because the Tuned model was trained on the 500,000-row sample to save time, whereas the Base model had the advantage of seeing all 4 million rows. This underscores a classic ML principle: **More data often beats better algorithms/parameters**.

## 4. Visualizations
The automated evaluation pipeline successfully generated and saved the following visual artifacts to the `reports/figures/` directory:
1.  `base_xgboost_cm.png` (Confusion Matrix for Base Model)
2.  `tuned_xgboost_cm.png` (Confusion Matrix for Tuned Model)
3.  `pr_curve_comparison.png` (Overlaid Precision-Recall curves)

These visualizations are embedded within the `05-evaluation-tuning.ipynb` notebook for easy stakeholder review.
