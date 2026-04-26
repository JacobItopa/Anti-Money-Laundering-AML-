# Step 3: Exploratory Data Analysis (EDA)

*Note: The comprehensive code and visualizations for this step are located in `notebooks/01-eda-initial.ipynb`.*

## 1. Descriptive Statistics
Our initial pass over the IBM AML dataset (`HI-Small_Trans.csv`) reveals the following structural properties:
*   **Data Types:** A mix of categorical (`Account`, `Receiving Currency`, `Payment Format`) and numerical/continuous variables (`Amount Received`, `Amount Paid`).
*   **Target Variable (`Is Laundering`):** Represents a highly imbalanced binary classification problem.
*   **Class Imbalance Ratio:** There is roughly 1 illicit transaction for every 1,750 normal transactions. This extreme ratio confirms that standard accuracy metrics will be completely ineffective.

## 2. Data Visualization
Key insights derived from our planned visual analysis:
*   **Transaction Amount Distributions:** The `Amount Received` and `Amount Paid` features are heavily skewed to the right (many small transactions, a few massive ones). Using a logarithmic scale for visualization and modeling is critical.
*   **Payment Format Patterns:** Different payment formats (e.g., checks vs. wire transfers) will likely have different baseline probabilities for laundering. We must analyze proportional illicit activity across these formats.

## 3. Identify Issues
Based on the dataset's structural constraints, we have identified three primary issues that must be addressed in Step 4 (Data Preprocessing):
1.  **Extreme Class Imbalance:** We will need to implement techniques like SMOTE (Synthetic Minority Over-sampling Technique), adjusting class weights in the model, or formulating the problem as an anomaly detection/graph learning task.
2.  **Highly Skewed Features:** Continuous numerical features require transformation (e.g., Log Transformation, RobustScaler) to prevent large values from dominating the model.
3.  **Complex Temporal Features:** The `Timestamp` column is currently a raw string. It must be parsed to extract meaningful temporal signals (e.g., "Time Since Last Transaction", "Day of the Week", or "Hour of Day"), as money laundering often exhibits strict temporal patterns.
