import os

def run_continuous_training():
    """
    Orchestrates the Continuous Training (CT) pipeline.
    In a real production environment, this is triggered by Airflow, GitHub Actions, or cron
    when drift_detection.py throws an alert.
    """
    print("=========================================")
    print("  AML Continuous Training (CT) Pipeline  ")
    print("=========================================")
    
    print("\n1. Ingesting new data and running Preprocessing...")
    os.system("python run_preprocessing.py")
    
    print("\n2. Re-extracting Preprocessing Artifacts for API...")
    os.system("python generate_artifacts.py")
    
    print("\n3. Training XGBoost Model on fresh data...")
    os.system("python run_training.py")
    
    print("\n4. Evaluating newly trained model...")
    os.system("python run_evaluation.py")
    
    print("\n=> Retraining Pipeline Complete. New model and artifacts have been serialized.")

if __name__ == "__main__":
    # run_continuous_training()
    print("Mock Retrain Pipeline initialized. Execute run_continuous_training() to trigger full pipeline.")
