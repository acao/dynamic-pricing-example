# Deployment Guide

This document provides instructions for deploying the insurance pricing model to production using Snowflake, Prefect, and AWS.

## Prerequisites

Before deploying the model to production, ensure you have the following:

- AWS account with appropriate permissions
- Snowflake account with appropriate permissions
- Prefect Cloud account or self-hosted Prefect server
- Python 3.9+ installed
- Docker and Docker Compose installed
- AWS CLI installed and configured

## Local Development and Testing

Before deploying to production, it's recommended to develop and test the model locally.

### Setting Up Local Environment

1. Clone the repository:

```bash
git clone <repository-url>
cd dynamic-pricing-example
```

2. Install dependencies:

```bash
pip install -r requirements.txt
npm install
```

3. Start the local development environment:

```bash
docker-compose up -d
```

This will start the following services:
- PostgreSQL database
- ML service
- API service
- Jupyter notebook
- Test runner

4. Access the services:
- API: http://localhost:8000
- Jupyter notebook: http://localhost:8888
- ML service: http://localhost:5000

### Running Tests

To run the tests, use the following commands:

```bash
# Run unit tests
docker-compose run test-runner pytest src/tests/unit

# Run integration tests
docker-compose run test-runner pytest src/tests/integration

# Run end-to-end tests
docker-compose run test-runner pytest src/tests/e2e
```

### Training the Model Locally

To train the model locally, use the following command:

```bash
python -m src.ml.train --num-policies 1000 --model-type decision_tree --output-dir ./models
```

This will generate synthetic data, train a decision tree model, and save it to the specified directory.

### Running the Model Locally with Prefect

To run the model training flow locally with Prefect, use the following command:

```bash
python -m infrastructure.prefect.run_local --num-training-policies 1000 --output-dir ./models
```

This will run the Prefect flow for training the model locally.

## Setting Up Snowflake

Before deploying to AWS, you need to set up Snowflake for storing model metadata and pricing history.

1. Connect to your Snowflake account:

```bash
snowsql -a <account> -u <username> -d <database> -r <role>
```

2. Run the Snowflake setup script:

```bash
snowsql -a <account> -u <username> -d <database> -r <role> -f infrastructure/snowflake/setup.sql
```

This will create the necessary tables and stored procedures in Snowflake.

## Deploying to AWS

### Setting Up AWS Infrastructure

1. Deploy the AWS infrastructure using CloudFormation:

```bash
python -m infrastructure.aws.deploy --stack-name insurance-pricing --environment dev --bucket-name insurance-pricing-models
```

This will create the following AWS resources:
- S3 bucket for storing models and data
- Lambda functions for model inference and training
- API Gateway for exposing the model API
- CloudWatch events for scheduling model training
- CloudWatch dashboard for monitoring

2. Upload the Lambda code:

```bash
python -m infrastructure.aws.deploy --stack-name insurance-pricing --environment dev --bucket-name insurance-pricing-models --lambda-source-dir ./lambda
```

This will upload the Lambda code to the S3 bucket.

### Deploying with Prefect

1. Create AWS credentials file:

```json
{
  "aws_access_key_id": "your-access-key",
  "aws_secret_access_key": "your-secret-key",
  "region_name": "us-east-1"
}
```

2. Create Snowflake credentials file:

```json
{
  "account": "your-account",
  "user": "your-username",
  "password": "your-password",
  "database": "INSURANCE_PRICING",
  "schema": "ML_MODELS",
  "warehouse": "INSURANCE_PRICING_WH",
  "role": "INSURANCE_PRICING_APP"
}
```

3. Deploy the model to AWS using Prefect:

```bash
python -m infrastructure.prefect.deploy_aws --bucket-name insurance-pricing-models-dev --aws-credentials ./aws-credentials.json --snowflake-credentials ./snowflake-credentials.json
```

This will create a Prefect deployment for training and deploying the model to AWS.

## Monitoring and Maintenance

### Monitoring

The deployed model can be monitored using the following:

- CloudWatch dashboard: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=insurance-pricing-dev
- Snowflake views:
  - `ACTIVE_MODEL`: Shows the currently active model
  - `PRICING_STATISTICS`: Shows statistics about pricing requests

### Maintenance

To update the model, you can:

1. Train a new model locally:

```bash
python -m src.ml.train --num-policies 5000 --model-type random_forest --output-dir ./models
```

2. Upload the model to S3:

```bash
aws s3 cp ./models/random_forest_<timestamp>.joblib s3://insurance-pricing-models-dev/models/
```

3. Register the model in Snowflake:

```sql
CALL REGISTER_MODEL(
  'random_forest',
  '<timestamp>',
  's3://insurance-pricing-models-dev/models/random_forest_<timestamp>.joblib',
  TRUE,
  PARSE_JSON('{"rmse": 120.5, "mae": 95.2, "r2": 0.85}')
);
```

4. Alternatively, you can trigger the Prefect flow to train and deploy a new model:

```bash
prefect deployment run train-insurance-pricing-model/insurance-pricing-aws
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check that the model file exists in the S3 bucket and that the Lambda function has permission to access it.

2. **API Gateway errors**: Check the CloudWatch logs for the Lambda function to see if there are any errors.

3. **Snowflake connection issues**: Check that the Snowflake credentials are correct and that the Lambda function has permission to access Snowflake.

### Logs

- Lambda logs: CloudWatch Logs
- API Gateway logs: CloudWatch Logs
- Prefect logs: Prefect UI or CloudWatch Logs

## Security Considerations

- Use IAM roles with least privilege
- Store credentials in AWS Secrets Manager
- Enable encryption for S3 buckets
- Use VPC for Lambda functions
- Enable API Gateway authorization
