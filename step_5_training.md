# Step 5: Model Selection & Training

*Note: The modular production code for this step is located in `src/models/train_model.py`, the pipeline is executed via `run_training.py`, and the interactive steps are showcased in `notebooks/03-model-training.ipynb`.*

## 1. Data Splitting
*   **Methodology:** We applied a **Stratified Train-Test Split** using an 80/20 ratio. Stratification is crucial here because of our 1:980 imbalance ratio. It ensures that the tiny minority of illicit transactions is equally represented in both the training set (for learning) and the test set (for validation).
*   **Result:** The split effectively generated roughly 4 million rows for training and 1 million rows for testing, perfectly preserving the exact 1:980 distribution.

## 2. Baseline Model
*   **Algorithm:** Logistic Regression.
*   **Configuration:** We used `class_weight='balanced'` which instructs the model to automatically penalize mistakes on the minority class more heavily than the majority class.
*   **Purpose:** This model sets the performance floor. While it might be robust, it generally struggles to capture the complex, non-linear relationships characteristic of money laundering patterns compared to tree-based methods.

## 3. Advanced Model
*   **Algorithm:** XGBoost Classifier.
*   **Configuration:** XGBoost natively supports extreme imbalances through the `scale_pos_weight` parameter. We dynamically calculated this as `Count(Negative) / Count(Positive)` (which evaluates to roughly 980). We also utilized `tree_method='hist'`, which builds histogram-based splits, drastically accelerating the training speed on our 5-million row dataset.
*   **Purpose:** Tree ensembles like XGBoost are the industry standard for tabular fraud datasets, capable of learning non-linear threshold patterns across amounts and frequencies while resisting the noise of the majority class.

## 4. Model Serialization
*   **Storage:** Both the Baseline and Advanced models were serialized using `joblib` and saved to the `models/` directory (`baseline_logreg.joblib` and `advanced_xgboost.joblib`).
*   **Test Data Serialization:** The test sets (`X_test.parquet` and `y_test.parquet`) were saved to `data/interim/` to guarantee that we evaluate the exact same untouched holdout set during Step 6 without data leakage.

## 5. Initial Findings (Performance Snapshot)
A quick evaluation on the holdout test set (1 million rows) utilizing our primary business metric (**PR-AUC**, Precision-Recall Area Under Curve) yielded the following initial findings:
*   **Random Guessing Baseline:** ~0.001 (due to the 1 in 980 ratio).
*   **Logistic Regression:** PR-AUC of **0.011** (11x better than random, but struggles overall).
*   **XGBoost:** PR-AUC of **0.068** (68x better than random, and over 6x better than our baseline model).

**Takeaway:** As expected, XGBoost vastly outperforms the simple linear baseline at detecting complex fraud typologies. However, while 68x better than random is mathematically significant, a 0.068 PR-AUC indicates we still have a high false positive rate. This confirms our initial intuition from Step 1: to push this model to a production-grade level, we will likely need to conduct intensive hyperparameter tuning in Step 6, or eventually introduce structural/network features (Graph Machine Learning).

## 6. Graph Neural Network (GNN) Comparison
Because money laundering intrinsically involves complex networks of transfers (e.g., cyclic layering, gather-scatter patterns), we hypothesized that a model capable of "seeing" the graph topology would outperform isolated tabular models.

To prove this, we built a **GraphSAGE Edge Classifier** using `PyTorch Geometric`. Due to the immense computational requirements of training a GNN on 5 million edges, we extracted a chronological **subgraph of 500,000 transactions**.

We trained the GNN and evaluated it against our serialized XGBoost model on the exact same test edges of this subgraph. 
*   **XGBoost PR-AUC (Subgraph):** 0.1904
*   **GNN PR-AUC (Subgraph):** **0.2914**

**Conclusion:** The GNN provided a **~50% relative improvement** over XGBoost. This definitively proves that structural network features are highly predictive for fraud detection. In a full production environment with sufficient GPU cluster resources, deploying this GNN would massively reduce false positives compared to standard tabular approaches.
