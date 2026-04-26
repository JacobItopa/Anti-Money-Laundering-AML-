# Step 4: Data Preprocessing & Feature Engineering

*Note: The modular production code for this step is located in `src/features/build_features.py` and the pipeline is executed via `run_preprocessing.py`.*

## 1. Data Cleaning
*   **Deduplication:** We successfully dropped the 9 exact duplicate rows identified during our EDA.
*   **Missing Values:** The dataset contained 0 missing values, so no imputation strategy was required.

## 2. Feature Engineering (Temporal)
The raw string `Timestamp` column was parsed into explicit temporal signals, which are often highly predictive for fraud/laundering models:
*   `Hour` (0-23)
*   `DayOfWeek` (0-6)
*   `Month` (1-12)
*   `IsWeekend` (Binary 0 or 1)

## 3. Feature Scaling
*   **Log Transformation:** To address the extreme skewness identified in EDA, we applied `np.log1p()` (log(1+x)) to the `Amount Received` and `Amount Paid` continuous features. This pulls in the extreme outliers and normalizes the distributions for the model.

## 4. Categorical Encoding
*   **Low Cardinality (One-Hot Encoding):** The `Payment Format` categorical variable was converted into binary dummy variables (e.g., `Payment Format_Wire`, `Payment Format_Cash`).
*   **High Cardinality (Frequency Encoding):** Columns with thousands of unique text IDs (`Receiving Currency`, `Payment Currency`, `From Bank`, `To Bank`) were converted to their relative frequencies. This converts massive categorical spaces into a single numeric column representing the commonality of the entity, without exploding the memory footprint.

## 5. Feature Selection & Export
*   **Dropped Columns:** We dropped the original raw string columns (`Timestamp`, `Account`, `Account.1`, `Receiving Currency`, `Payment Currency`, `From Bank`, `To Bank`) as they are redundant or unreadable by basic ML algorithms.
*   **Output:** The final processed dataset contains **5,078,336 rows** and **17 entirely numeric features**. 
*   **Storage:** The cleaned data was serialized to the highly compressed Parquet format and saved at `data/processed/processed_transactions.parquet` for rapid loading in Step 5.
