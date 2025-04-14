#!/usr/bin/env python3
"""
Script to deploy the insurance pricing model infrastructure to AWS.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deploy")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy insurance pricing model to AWS")
    parser.add_argument(
        "--stack-name",
        type=str,
        default="insurance-pricing",
        help="Name of the CloudFormation stack",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Environment to deploy to",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="insurance-pricing-models",
        help="Name of the S3 bucket for storing models and data",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region to deploy to",
    )
    parser.add_argument(
        "--lambda-source-dir",
        type=str,
        default="lambda",
        help="Directory containing Lambda function source code",
    )
    parser.add_argument(
        "--model-training-schedule",
        type=str,
        default="rate(1 day)",
        help="Schedule expression for model training (cron or rate)",
    )
    parser.add_argument(
        "--model-training-policies",
        type=int,
        default=5000,
        help="Number of policies to generate for model training",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "random_forest"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--api-stage-name",
        type=str,
        default="v1",
        help="Stage name for the API Gateway",
    )
    parser.add_argument(
        "--skip-lambda-upload",
        action="store_true",
        help="Skip uploading Lambda functions",
    )
    parser.add_argument(
        "--skip-stack-creation",
        action="store_true",
        help="Skip creating CloudFormation stack",
    )
    parser.add_argument(
        "--skip-model-upload",
        action="store_true",
        help="Skip uploading model to S3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deployment even if stack already exists",
    )
    return parser.parse_args()


def create_lambda_zip(source_dir, output_dir, function_name):
    """Create a ZIP file for a Lambda function."""
    source_path = Path(source_dir) / function_name
    output_path = Path(output_dir) / f"{function_name}.zip"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create ZIP file
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_path)
                zipf.write(file_path, arcname)
    
    logger.info(f"Created Lambda ZIP file: {output_path}")
    return output_path


def upload_to_s3(bucket_name, file_path, s3_key):
    """Upload a file to S3."""
    s3 = boto3.client("s3")
    
    # Check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except Exception:
        logger.info(f"Creating bucket: {bucket_name}")
        s3.create_bucket(Bucket=bucket_name)
    
    # Upload file
    logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
    s3.upload_file(str(file_path), bucket_name, s3_key)
    return f"s3://{bucket_name}/{s3_key}"


def deploy_cloudformation_stack(args):
    """Deploy CloudFormation stack."""
    cloudformation = boto3.client("cloudformation", region_name=args.region)
    
    # Check if stack exists
    stack_exists = False
    try:
        cloudformation.describe_stacks(StackName=args.stack_name)
        stack_exists = True
    except Exception:
        pass
    
    # Prepare parameters
    parameters = [
        {"ParameterKey": "Environment", "ParameterValue": args.environment},
        {"ParameterKey": "BucketName", "ParameterValue": args.bucket_name},
        {"ParameterKey": "ApiStageName", "ParameterValue": args.api_stage_name},
        {"ParameterKey": "ModelTrainingSchedule", "ParameterValue": args.model_training_schedule},
        {"ParameterKey": "ModelTrainingPolicies", "ParameterValue": str(args.model_training_policies)},
        {"ParameterKey": "ModelType", "ParameterValue": args.model_type},
    ]
    
    # Read CloudFormation template
    template_path = Path(__file__).parent / "cloudformation.yaml"
    with open(template_path, "r") as f:
        template_body = f.read()
    
    # Deploy stack
    if stack_exists:
        if not args.force:
            logger.info(f"Stack {args.stack_name} already exists. Use --force to update it.")
            return
        
        logger.info(f"Updating stack: {args.stack_name}")
        try:
            cloudformation.update_stack(
                StackName=args.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
            )
            wait_for_stack_update(cloudformation, args.stack_name)
        except Exception as e:
            if "No updates are to be performed" in str(e):
                logger.info("No updates are to be performed.")
            else:
                logger.error(f"Error updating stack: {e}")
                sys.exit(1)
    else:
        logger.info(f"Creating stack: {args.stack_name}")
        try:
            cloudformation.create_stack(
                StackName=args.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
                OnFailure="DELETE",
            )
            wait_for_stack_creation(cloudformation, args.stack_name)
        except Exception as e:
            logger.error(f"Error creating stack: {e}")
            sys.exit(1)
    
    # Get stack outputs
    response = cloudformation.describe_stacks(StackName=args.stack_name)
    outputs = {
        output["OutputKey"]: output["OutputValue"]
        for output in response["Stacks"][0]["Outputs"]
    }
    
    logger.info("Stack outputs:")
    for key, value in outputs.items():
        logger.info(f"  {key}: {value}")
    
    return outputs


def wait_for_stack_creation(cloudformation, stack_name):
    """Wait for CloudFormation stack creation to complete."""
    logger.info(f"Waiting for stack {stack_name} to be created...")
    waiter = cloudformation.get_waiter("stack_create_complete")
    waiter.wait(StackName=stack_name)
    logger.info(f"Stack {stack_name} created successfully.")


def wait_for_stack_update(cloudformation, stack_name):
    """Wait for CloudFormation stack update to complete."""
    logger.info(f"Waiting for stack {stack_name} to be updated...")
    waiter = cloudformation.get_waiter("stack_update_complete")
    waiter.wait(StackName=stack_name)
    logger.info(f"Stack {stack_name} updated successfully.")


def upload_lambda_functions(args, bucket_name):
    """Upload Lambda functions to S3."""
    # Create temporary directory for Lambda ZIP files
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Create and upload Lambda ZIP files
    lambda_functions = ["inference", "training"]
    for function_name in lambda_functions:
        zip_path = create_lambda_zip(args.lambda_source_dir, temp_dir, function_name)
        s3_key = f"lambda/{function_name}.zip"
        upload_to_s3(bucket_name, zip_path, s3_key)


def upload_model(args, bucket_name):
    """Upload model to S3."""
    # Check if model exists
    model_dir = Path("models")
    model_files = list(model_dir.glob(f"{args.model_type}_*.joblib"))
    
    if not model_files:
        logger.warning(f"No {args.model_type} model found in {model_dir}. Skipping model upload.")
        return
    
    # Upload latest model
    latest_model = sorted(model_files)[-1]
    model_version = latest_model.stem.split("_")[1]
    s3_key = f"models/{latest_model.name}"
    
    upload_to_s3(bucket_name, latest_model, s3_key)
    logger.info(f"Uploaded model {latest_model} to S3.")
    
    # Register model in Snowflake
    register_model_in_snowflake(
        args,
        model_name=args.model_type,
        model_version=model_version,
        model_path=f"s3://{bucket_name}/{s3_key}",
    )


def register_model_in_snowflake(args, model_name, model_version, model_path):
    """Register model in Snowflake."""
    # This would typically use the Snowflake Python connector
    # For simplicity, we'll just log the command
    logger.info(f"Would register model in Snowflake:")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Model version: {model_version}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Environment: {args.environment}")


def main():
    """Main function."""
    args = parse_args()
    
    # Construct bucket name with environment
    bucket_name = f"{args.bucket_name}-{args.environment}"
    
    # Upload Lambda functions
    if not args.skip_lambda_upload:
        upload_lambda_functions(args, bucket_name)
    
    # Deploy CloudFormation stack
    if not args.skip_stack_creation:
        outputs = deploy_cloudformation_stack(args)
    
    # Upload model
    if not args.skip_model_upload:
        upload_model(args, bucket_name)
    
    logger.info("Deployment completed successfully.")


if __name__ == "__main__":
    main()
