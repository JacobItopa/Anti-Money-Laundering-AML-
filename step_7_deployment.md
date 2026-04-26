# Step 7: Model Deployment (MLOps)

*Note: The production API code is located in `src/app/main.py` and the containerization setup is defined in the `Dockerfile`.*

## 1. MLOps Challenge Solved: Preprocessing State
When training an algorithm with millions of rows, statistical encoding techniques like "Frequency Encoding" map a text ID (like a Bank Name) to a numeric percentage representing how often it appears. 
However, when a live API receives a single transaction, it is impossible to calculate a "frequency" for one row. To solve this critical MLOps issue, we executed `generate_artifacts.py`, which extracted all historical frequency mappings and required column structures from our training data and saved them to `models/preprocessing_artifacts.joblib`. 

## 2. API Creation (FastAPI)
We built a highly robust, high-performance web server using **FastAPI** (`src/app/main.py`).
*   **Pydantic Schema:** The API strictly validates incoming JSON requests to ensure the data structure perfectly matches what our model expects (`TransactionInput`).
*   **Inference Pipeline:** When a request hits the `/predict` endpoint, the API loads the raw JSON, dynamically applies our custom temporal engineering, scales the numbers, applies the historical frequency mappings using our artifact dictionary, and feeds the resulting exact tensor to the serialized XGBoost model.
*   **Output:** It returns a JSON object containing the `fraud_probability`, the raw `is_laundering` prediction, and a boolean `flagged_for_review` trigger.

## 3. Containerization (Docker)
To guarantee that this API runs perfectly on any machine or cloud provider, we isolated it using **Docker**.
*   We built a `Dockerfile` utilizing a lightweight `python:3.9-slim` base image.
*   It securely packages our `requirements.txt`, our `src/app/` code, and our serialized `models/` directory, completely isolating it from the host OS system dependencies.
*   We utilized a `.dockerignore` file to prevent heavy datasets (`data/`) from bloating the image.

## 4. Google Cloud Platform (GCP) Readiness
Because you do not currently have an active GCP subscription, we *prepared* the deployment instead of executing it.
We created the `deploy_gcp.sh` script, which contains the exact, sequence-ordered `gcloud` commands required to:
1.  Configure the target Google Cloud Project ID.
2.  Use Google Cloud Build to construct the Docker container remotely and push it to the Google Container Registry (GCR).
3.  Deploy the GCR image as a fully serverless, highly-scalable microservice using **Google Cloud Run**.

Once your GCP account is active, simply open `deploy_gcp.sh`, input your project ID, uncomment the commands, and execute the file.
