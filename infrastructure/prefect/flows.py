"""
Prefect flows for insurance pricing model training and deployment.
"""
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import boto3
import numpy as np
import pandas as pd
import snowflake.connector
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from prefect_aws import AwsCredentials
from prefect_snowflake import SnowflakeCredentials

from src.data.generator import InsuranceDataGenerator
from src.ml.model import InsurancePricingModel


@task
def generate_synthetic_data(
    num_policies: int = 5000, seed: int = 42
) -> Tuple[List, List]:
    """Generate synthetic data for model training."""
    logger = get_run_logger()
    logger.info(f"Generating {num_policies} synthetic policies with seed {seed}")
    
    data_generator = InsuranceDataGenerator(seed=seed)
    policies, premiums = data_generator.generate_dataset(num_policies)
    
    logger.info(f"Generated {len(policies)} policies with premiums")
    return policies, premiums


@task
def train_model(
    policies: List,
    premiums: List,
    model_type: str = "decision_tree",
    model_params: Optional[Dict] = None,
) -> Tuple[InsurancePricingModel, Dict]:
    """Train the insurance pricing model."""
    logger = get_run_logger()
    logger.info(f"Training {model_type} model")
    
    if model_params is None:
        model_params = {}
    
    model = InsurancePricingModel(model_type=model_type, model_params=model_params)
    metrics = model.train(policies, premiums)
    
    logger.info(f"Model trained with metrics: {metrics}")
    return model, metrics


@task
def save_model_locally(
    model: InsurancePricingModel, output_dir: str = "./models"
) -> str:
    """Save the model to a local file."""
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = model.save(output_dir)
    logger.info(f"Model saved to {model_path}")
    
    return model_path


@task
def upload_model_to_s3(
    model_path: str,
    bucket_name: str,
    aws_credentials: Optional[AwsCredentials] = None,
) -> str:
    """Upload the model to S3."""
    logger = get_run_logger()
    logger.info(f"Uploading model {model_path} to S3 bucket {bucket_name}")
    
    # Get AWS credentials
    if aws_credentials:
        aws_credentials_dict = aws_credentials.get_credentials_as_dict()
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_credentials_dict["aws_access_key_id"],
            aws_secret_access_key=aws_credentials_dict["aws_secret_access_key"],
            region_name=aws_credentials_dict.get("region_name", "us-east-1"),
        )
    else:
        s3 = boto3.client("s3")
    
    # Upload model
    model_filename = os.path.basename(model_path)
    s3_key = f"models/{model_filename}"
    
    s3.upload_file(model_path, bucket_name, s3_key)
    logger.info(f"Model uploaded to s3://{bucket_name}/{s3_key}")
    
    return f"s3://{bucket_name}/{s3_key}"


@task
def register_model_in_snowflake(
    model_name: str,
    model_version: str,
    model_path: str,
    metrics: Dict,
    is_active: bool = True,
    snowflake_credentials: Optional[SnowflakeCredentials] = None,
) -> str:
    """Register the model in Snowflake."""
    logger = get_run_logger()
    logger.info(f"Registering model {model_name} version {model_version} in Snowflake")
    
    # Get Snowflake credentials
    if snowflake_credentials:
        conn = snowflake_credentials.get_connection()
    else:
        # Get credentials from environment variables
        conn = snowflake.connector.connect(
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
        )
    
    # Register model
    cursor = conn.cursor()
    try:
        metrics_json = json.dumps(metrics)
        cursor.execute(
            "CALL ML_MODELS.REGISTER_MODEL(%s, %s, %s, %s, %s)",
            (model_name, model_version, model_path, is_active, metrics_json),
        )
        result = cursor.fetchone()
        model_id = result[0]
        logger.info(f"Model registered with ID: {model_id}")
        return model_id
    finally:
        cursor.close()
        conn.close()


@task
def upload_training_data_to_snowflake(
    policies: List,
    premiums: List,
    snowflake_credentials: Optional[SnowflakeCredentials] = None,
) -> int:
    """Upload training data to Snowflake."""
    logger = get_run_logger()
    logger.info(f"Uploading {len(policies)} policies to Snowflake")
    
    # Get Snowflake credentials
    if snowflake_credentials:
        conn = snowflake_credentials.get_connection()
    else:
        # Get credentials from environment variables
        conn = snowflake.connector.connect(
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
        )
    
    # Upload data
    cursor = conn.cursor()
    try:
        count = 0
        for policy, premium in zip(policies, premiums):
            policy_json = json.dumps(policy.to_dict())
            cursor.execute(
                "INSERT INTO STAGING.TRAINING_DATA (POLICY_DATA, PREMIUM) VALUES (%s, %s)",
                (policy_json, premium),
            )
            count += 1
        
        conn.commit()
        logger.info(f"Uploaded {count} policies to Snowflake")
        return count
    finally:
        cursor.close()
        conn.close()


@flow(name="train-insurance-pricing-model")
def train_insurance_pricing_model(
    num_policies: int = 5000,
    model_type: str = "decision_tree",
    model_params: Optional[Dict] = None,
    output_dir: str = "./models",
    upload_to_s3: bool = False,
    bucket_name: Optional[str] = None,
    register_in_snowflake: bool = False,
    upload_data_to_snowflake: bool = False,
    is_active: bool = True,
    aws_credentials_block: Optional[str] = None,
    snowflake_credentials_block: Optional[str] = None,
) -> Dict:
    """Train and deploy the insurance pricing model."""
    logger = get_run_logger()
    logger.info(f"Starting insurance pricing model training flow")
    
    # Get credentials
    aws_credentials = None
    snowflake_credentials = None
    
    if aws_credentials_block:
        aws_credentials = AwsCredentials.load(aws_credentials_block)
    
    if snowflake_credentials_block:
        snowflake_credentials = SnowflakeCredentials.load(snowflake_credentials_block)
    
    # Generate synthetic data
    policies, premiums = generate_synthetic_data(num_policies=num_policies)
    
    # Upload training data to Snowflake
    if upload_data_to_snowflake:
        upload_training_data_to_snowflake(
            policies=policies,
            premiums=premiums,
            snowflake_credentials=snowflake_credentials,
        )
    
    # Train model
    model, metrics = train_model(
        policies=policies,
        premiums=premiums,
        model_type=model_type,
        model_params=model_params,
    )
    
    # Save model locally
    model_path = save_model_locally(model=model, output_dir=output_dir)
    
    # Get model version from filename
    model_filename = os.path.basename(model_path)
    model_version = model_filename.split("_")[1].split(".")[0]
    
    # Upload model to S3
    s3_path = None
    if upload_to_s3 and bucket_name:
        s3_path = upload_model_to_s3(
            model_path=model_path,
            bucket_name=bucket_name,
            aws_credentials=aws_credentials,
        )
    
    # Register model in Snowflake
    model_id = None
    if register_in_snowflake:
        model_id = register_model_in_snowflake(
            model_name=model_type,
            model_version=model_version,
            model_path=s3_path or model_path,
            metrics=metrics,
            is_active=is_active,
            snowflake_credentials=snowflake_credentials,
        )
    
    # Return results
    results = {
        "model_path": model_path,
        "s3_path": s3_path,
        "model_id": model_id,
        "model_version": model_version,
        "metrics": metrics,
    }
    
    logger.info(f"Insurance pricing model training flow completed")
    logger.info(f"Results: {results}")
    
    return results


@flow(name="deploy-insurance-pricing-model")
def deploy_insurance_pricing_model(
    model_path: str,
    bucket_name: str,
    is_active: bool = True,
    aws_credentials_block: Optional[str] = None,
    snowflake_credentials_block: Optional[str] = None,
) -> Dict:
    """Deploy an existing insurance pricing model."""
    logger = get_run_logger()
    logger.info(f"Starting insurance pricing model deployment flow")
    
    # Get credentials
    aws_credentials = None
    snowflake_credentials = None
    
    if aws_credentials_block:
        aws_credentials = AwsCredentials.load(aws_credentials_block)
    
    if snowflake_credentials_block:
        snowflake_credentials = SnowflakeCredentials.load(snowflake_credentials_block)
    
    # Load model to get metrics
    model = InsurancePricingModel(model_path=model_path)
    metrics = {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 0.0,
    }
    
    # Get model version from filename
    model_filename = os.path.basename(model_path)
    model_name = model_filename.split("_")[0]
    model_version = model_filename.split("_")[1].split(".")[0]
    
    # Upload model to S3
    s3_path = upload_model_to_s3(
        model_path=model_path,
        bucket_name=bucket_name,
        aws_credentials=aws_credentials,
    )
    
    # Register model in Snowflake
    model_id = register_model_in_snowflake(
        model_name=model_name,
        model_version=model_version,
        model_path=s3_path,
        metrics=metrics,
        is_active=is_active,
        snowflake_credentials=snowflake_credentials,
    )
    
    # Return results
    results = {
        "model_path": model_path,
        "s3_path": s3_path,
        "model_id": model_id,
        "model_version": model_version,
        "metrics": metrics,
    }
    
    logger.info(f"Insurance pricing model deployment flow completed")
    logger.info(f"Results: {results}")
    
    return results


if __name__ == "__main__":
    # Run the flow for local testing
    train_insurance_pricing_model(
        num_policies=100,
        model_type="decision_tree",
        model_params={"max_depth": 10},
        output_dir="./models",
        upload_to_s3=False,
        register_in_snowflake=False,
        upload_data_to_snowflake=False,
    )
