# Step 8: Monitoring & Maintenance

*Note: The functional monitoring scripts are in `src/models/drift_detection.py` and `retrain_pipeline.py`. The production API now includes built-in latency middleware in `src/app/main.py`.*

## 1. Monitor System Metrics
The production API (`src/app/main.py`) now includes a **custom HTTP middleware** that intercepts every request and automatically logs:
- **Request path** (e.g., `/predict`, `/health`)
- **Latency** in seconds (e.g., `Metrics - Path: /predict | Latency: 0.0043 secs`)
- **X-Process-Time** header returned with every response

In a full production setup, these logs would be shipped to **Google Cloud Logging** (since we're targeting Cloud Run), and visualized in a **Grafana dashboard** connected to Prometheus or Cloud Monitoring.

## 2. Monitor Data Drift
Data Drift occurs when the statistical properties of incoming live transactions start to diverge from the distribution the model was trained on. For example:
- A new digital payment method (like CBDC transfers) suddenly appearing in live data.
- A change in average transaction amounts following an economic event.

We implemented `src/models/drift_detection.py` which runs **Kolmogorov-Smirnov (KS) tests** on key numerical features (`Amount Received`, `Amount Paid`, `Hour`, `DayOfWeek`) and compares the live distribution against the reference training distribution.

**Mock test result (comparing identical distributions):**
```
[OK] No drift in 'Amount Received' (p-value: 1.0000)
[OK] No drift in 'Amount Paid'     (p-value: 1.0000)
[OK] No drift in 'Hour'            (p-value: 1.0000)
[OK] No drift in 'DayOfWeek'       (p-value: 1.0000)
=> STATUS: Data distribution is stable.
```

When p-value drops below `0.05` on a real live dataset, the script flags `[ALERT]` and recommends immediate retraining.

## 3. Monitor Concept Drift
Concept Drift is more subtle than Data Drift. It occurs when the *relationship* between the input features and the fraud label changes, even if the raw data distribution stays the same. For AML specifically:
- **Example:** Criminal networks could shift from "Reinvestment" wire cycles to "Cash" micro-structuring patterns, meaning the same feature values now produce a different laundering outcome.
- **Detection Strategy:** Monitor the model's **live PR-AUC** over rolling 30-day windows. A sustained drop in PR-AUC without a shift in data distributions is a strong indicator of Concept Drift.
- **Tooling:** Services like [Evidently AI](https://www.evidentlyai.com/) or [WhyLabs](https://whylabs.ai/) can automate this detection.

## 4. Retraining Strategy (CI/CD/CT)
We have implemented a Continuous Training (CT) orchestration script (`retrain_pipeline.py`) that re-runs the entire pipeline end-to-end. In production, it would be triggered by:
1. **Scheduled Trigger:** A monthly cron job (e.g., Google Cloud Scheduler) to proactively retrain.
2. **Drift Alert Trigger:** An automated GitHub Actions workflow fired when `drift_detection.py` raises an `[ALERT]`.

**Retraining pipeline order:**
1. `run_preprocessing.py` — Ingest fresh data, rebuild features.
2. `generate_artifacts.py` — Re-extract frequency maps from updated training data.
3. `run_training.py` — Retrain XGBoost on the latest dataset.
4. `run_evaluation.py` — Validate the new model's PR-AUC against the previous baseline.
5. Re-deploy updated `models/` artifacts to the Cloud Run container.

This ensures the AML model always reflects current laundering typologies and remains financially and legally compliant.
