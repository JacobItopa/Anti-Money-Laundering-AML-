#!/bin/bash
# Exit if any command fails
set -e

echo "=== AML Model GCP Deployment Script ==="
echo "Note: You must have an active GCP subscription and gcloud CLI authenticated."

# Variables (Replace with your actual GCP Project ID)
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
SERVICE_NAME="aml-fraud-api"
IMAGE_URI="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "1. Configuring gcloud project..."
# gcloud config set project $PROJECT_ID

echo "2. Building the Docker image via Google Cloud Build..."
# gcloud builds submit --tag $IMAGE_URI

echo "3. Deploying to Google Cloud Run..."
# gcloud run deploy $SERVICE_NAME \
#   --image $IMAGE_URI \
#   --platform managed \
#   --region $REGION \
#   --allow-unauthenticated \
#   --memory 1Gi \
#   --cpu 1

echo "Deployment completed! (Uncomment commands above to execute)"
