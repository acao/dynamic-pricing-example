# Dynamic Pricing System for Car Insurance

A comprehensive system for dynamic pricing of car insurance policies using machine learning models, with real-time deployment capabilities on AWS and Snowflake.

## Overview

This project implements a dynamic pricing model for car insurance using decision tree-based machine learning algorithms. The system is designed to:

1. Generate synthetic insurance policy data for training
2. Train and evaluate ML models for premium prediction
3. Deploy models to production with real-time inference capabilities
4. Monitor model performance and detect drift
5. Provide a robust testing framework for local development

The architecture leverages AWS for cloud infrastructure, Snowflake for data storage and analytics, and Prefect for workflow orchestration.

## Key Features

- **Advanced ML Models**: Decision trees, random forests, and gradient boosting models with hyperparameter tuning
- **Sophisticated Feature Engineering**: Comprehensive feature extraction from policy data with risk curve modeling
- **Real-time Inference**: API endpoints for real-time pricing quotes
- **Model Monitoring**: Performance tracking and drift detection
- **Automated Deployment**: CI/CD pipeline for model training and deployment
- **Comprehensive Testing**: Unit, integration, and end-to-end testing framework

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Sources   │────▶│  ML Pipeline    │────▶│  Deployment     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Snowflake DB   │◀───▶│  AWS S3/Lambda  │◀───▶│  API Gateway    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  Client Apps    │
                                               │                 │
                                               └─────────────────┘
```

## Technical Components

### ML Models

- **Base Models**: Decision trees for interpretability
- **Advanced Models**: Random forests and gradient boosting for improved accuracy
- **Model Ensembles**: Combining multiple models for better performance
- **Explainability**: SHAP values for model interpretation

### Feature Engineering

- **Driver Features**: Age, experience, license history with sophisticated risk curves
- **Vehicle Features**: Make, model, age, value with category-based risk factors
- **Location Features**: Geographic risk assessment with territory classification
- **History Features**: Accident, violation, and claim history analysis

### Data Infrastructure

- **Snowflake**: Data warehouse for storing model metadata, training data, and pricing history
- **S3**: Storage for model artifacts and deployment packages
- **Lambda**: Serverless compute for model inference and training

### Workflow Orchestration

- **Prefect**: Workflow management for model training and deployment
- **CloudWatch Events**: Scheduled model retraining and monitoring
- **CI/CD Pipeline**: Automated testing and deployment

### Monitoring and Observability

- **Performance Metrics**: RMSE, MAE, R², MAPE tracking
- **Drift Detection**: Feature and prediction drift monitoring
- **Health Checks**: Automated model health assessment

## Local Development

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- AWS CLI (for deployment)
- Snowflake account (for data storage)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dynamic-pricing-example.git
   cd dynamic-pricing-example
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start local development environment:
   ```bash
   docker-compose up -d
   ```

4. Run tests:
   ```bash
   pytest src/tests/
   ```

### Local Training

Train a model locally:

```bash
python -m src.ml.train --num-policies 1000 --model-type decision_tree --output-dir ./models
```

### Local API Server

Start the API server:

```bash
python -m src.ml.app
```

## Production Deployment

### AWS Deployment

Deploy to AWS using CloudFormation:

```bash
python infrastructure/aws/deploy.py --stack-name insurance-pricing --environment dev
```

### Prefect Flows

Register and run Prefect flows:

```bash
python infrastructure/prefect/run_local.py --num-training-policies 5000 --model-type gradient_boosting
```

Deploy flows to AWS:

```bash
python infrastructure/prefect/deploy_aws.py --bucket-name insurance-pricing-models-dev
```

### Snowflake Setup

Initialize Snowflake database:

```bash
snowsql -f infrastructure/snowflake/setup.sql
```

## Model Monitoring

Monitor model performance:

```python
from src.ml.monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(model_path="models/decision_tree_latest.joblib")

# Calculate performance metrics
metrics = monitor.calculate_performance_metrics()

# Check model health
health = monitor.check_model_health()
```

## API Usage

Request a pricing quote:

```bash
curl -X POST http://localhost:5000/pricing \
  -H "Content-Type: application/json" \
  -d '{
    "effective_date": "2025-01-01",
    "expiration_date": "2026-01-01",
    "drivers": [...],
    "vehicles": [...],
    "locations": [...],
    "driving_history": [...]
  }'
```

## Project Structure

```
.
├── docker/                  # Docker configuration
├── docs/                    # Documentation
├── infrastructure/          # Infrastructure as code
│   ├── aws/                 # AWS CloudFormation templates
│   ├── prefect/             # Prefect workflows
│   └── snowflake/           # Snowflake SQL scripts
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── api/                 # API service
│   ├── config/              # Configuration
│   ├── data/                # Data generation and processing
│   ├── ml/                  # Machine learning models
│   └── tests/               # Tests
├── docker-compose.yml       # Local development environment
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Recent Improvements

### Enhanced ML Models

- Added gradient boosting models for improved accuracy
- Implemented model ensembles for more robust predictions
- Added SHAP-based explainability for model interpretability

### Advanced Feature Engineering

- Implemented sophisticated risk curves for driver age and experience
- Enhanced vehicle categorization with detailed risk factors
- Improved location-based risk assessment with territory classification

### Real-time Monitoring

- Added comprehensive model performance tracking
- Implemented feature and prediction drift detection
- Created automated model health assessment

### Production Readiness

- Enhanced Snowflake integration for data storage and analytics
- Improved AWS deployment with CloudFormation
- Added Prefect workflows for orchestration

## Running Prefect Training Locally

You can use Prefect to orchestrate the model training process. This provides better tracking, logging, and the ability to schedule and monitor training runs.

### Basic Local Training

To run a basic training job locally with Prefect:

```bash
# Navigate to the infrastructure/prefect directory
cd infrastructure/prefect

# Run with default parameters (1000 policies, decision tree model)
uv run run_local.py

# Run with custom parameters
uv run run_local.py --num-training-policies 5000 --model-type random_forest --max-depth 15 --output-dir ../../models
```

### Advanced Options

The Prefect flow supports additional options for cloud deployment:

```bash
# Upload model to S3 (requires AWS credentials)
uv run run_local.py --upload-to-s3 --bucket-name your-model-bucket

# Register model in Snowflake (requires Snowflake credentials)
uv run run_local.py --register-in-snowflake

# Upload training data to Snowflake
uv run run_local.py --upload-data-to-snowflake

# Use stored credentials blocks
uv run run_local.py --aws-credentials-block "aws-creds" --snowflake-credentials-block "snowflake-creds"
```

### Setting Up Prefect Credentials

To use cloud services, you'll need to set up Prefect credential blocks:

```bash
# Install Prefect CLI
pip install prefect

# Login to Prefect Cloud (optional)
prefect cloud login

# Create AWS credentials block
prefect block register -m prefect_aws.credentials
prefect block create aws-credentials --name aws-creds

# Create Snowflake credentials block
prefect block register -m prefect_snowflake
prefect block create snowflake-credentials --name snowflake-creds
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

### Automated Testing

- **Unit Tests**: Tests for individual components
- **Integration Tests**: Tests for component interactions
- **End-to-End Tests**: Tests for the complete system

Run tests locally:

```bash
# Run all tests
pytest

# Run specific test types
pytest src/tests/unit/
pytest src/tests/integration/
pytest src/tests/e2e/
```

### Code Quality Checks

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run code quality checks locally:

```bash
black src/
isort src/
flake8 src/
mypy src/
```

### Docker Image Building

Automatically builds and publishes Docker images for:
- API Service
- ML Service
- Jupyter Notebook Server

## License

MIT
