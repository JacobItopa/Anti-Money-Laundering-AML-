# Step 1: Problem Definition & Scoping

## a. Identify the Business Objective: What is the actual problem?
In the real world, banks are legally required to monitor transactions, but their legacy rule-based systems generate an overwhelming number of false alarms.

**The Core Problem:** Human compliance analysts waste thousands of hours chasing down false positives, which costs the bank money and risks missing the actual, highly sophisticated money-laundering rings.

**The Business Objective:** Build a system that accurately flags illicit transaction networks while drastically minimizing the noise. The goal isn't just to "catch bad guys," but to optimize the investigative workflow so the human analysts are only reviewing high-risk, highly probable laundering events.

## b. Define Success Metrics: How will you know if the project is successful?
Because you are dealing with an anomaly (in the Low-Illicit dataset, laundering happens roughly 1 in every 1,750 transactions), standard metrics like Accuracy will be completely misleading. You need to split your metrics into technical and business goals:

### Machine Learning Metrics:
* **PR-AUC (Precision-Recall Area Under Curve):** This is your north star for extreme class imbalance.
* **Recall (Sensitivity):** Crucial because the cost of missing a true launderer (regulatory fines) is massive.
* **F1-Score (Minority Class):** To balance the trade-off between precision and recall.

### Business Metrics:
* **Alert Volume Reduction:** What percentage of false positives did you eliminate compared to a baseline rule-based system?
* **Investigation Yield:** The ratio of flagged alerts that actually turn out to be illicit.

### System/Ops Metrics:
* **Inference Latency:** If this model is meant to intercept live transactions, it needs to return a prediction in milliseconds. Complex models might require optimization before deployment.

## c. Determine the ML Task: Frame the problem
You have a few ways to frame this, depending on how advanced you want the architecture to be:

* **Baseline Approach (Supervised Binary Classification):** You can treat each transaction independently and predict `0` (Regular) or `1` (Illicit) using traditional models like XGBoost or LightGBM. You would need heavy feature engineering to capture rolling windows (e.g., "amount transferred in the last 24 hours").
* **State-of-the-Art Approach (Graph Machine Learning):** Money laundering is inherently a network problem (placement, layering, integration). You can frame this as an Edge Classification or Node Classification task. By modeling accounts as nodes and transactions as edges, you can use Graph Neural Networks (GNNs) to let the model naturally learn the structural typologies (like cyclic transfers or gather-scatter patterns) of illicit networks.

## d. Assess Feasibility: Do you have the right data and resources?
Yes, we will be using the **HI-Small dataset** from IBM Transactions for Anti Money Laundering (AML).
* **Dataset Link:** [IBM Transactions for AML on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data)
