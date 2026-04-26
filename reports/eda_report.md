# Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview
We loaded and executed our modular EDA scripts against `HI-Small_Trans.csv`.
* **Total Transactions (Rows):** 5,078,345
* **Total Features (Columns):** 11

### Columns Present:
`Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`.

## 2. Data Quality Checks
* **Missing Values:** `0` across all columns. The dataset is perfectly populated without NaNs.
* **Duplicates:** `9` duplicated rows were identified. This is a very small number but should be dropped in the preprocessing step to ensure data integrity.

## 3. Class Imbalance Analysis
We computed the exact distribution of our target variable `Is Laundering`.
* **Normal Transactions (Class 0):** 5,073,168
* **Illicit Transactions (Class 1):** 5,177
* **Ratio:** There is exactly **1 illicit transaction for every 980 normal transactions**. 

*Note: While previous documentation estimated 1-in-1,750, our exact programmatic calculation against the HI-Small dataset reveals a ~1:980 ratio. This remains an extreme class imbalance requiring PR-AUC metrics and specialized handling (e.g., SMOTE, class weights).*

## 4. Visualizations
The visual distributions have been successfully generated and saved to the `reports/figures/` directory.

### Transaction Amount Distribution
![Amount Received Distribution](file:///c:/Users/user/Desktop/2026%20PLAN%20AND%20PROJECT/projects/AML/reports/figures/amount_received_dist.png)
*The distribution is highly skewed, justifying our use of a logarithmic scale. The bulk of transactions are of lower value, but a long tail exists.*

### Payment Format Patterns
![Payment Format Distribution](file:///c:/Users/user/Desktop/2026%20PLAN%20AND%20PROJECT/projects/AML/reports/figures/payment_format_dist.png)
*This count plot (log scale) reveals the breakdown of normal vs. illicit transactions across different categorical payment formats.*
