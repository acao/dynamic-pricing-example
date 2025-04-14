#!/usr/bin/env python3
"""
Script to run the insurance pricing model training flow locally.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from flows import train_insurance_pricing_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_local")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run insurance pricing model training flow locally"
    )
    parser.add_argument(
        "--num-training-policies",
        type=int,
        default=1000,
        help="Number of policies to generate for training",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "random_forest"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of the decision tree",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload model to S3",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        help="Name of the S3 bucket for storing models",
    )
    parser.add_argument(
        "--register-in-snowflake",
        action="store_true",
        help="Register model in Snowflake",
    )
    parser.add_argument(
        "--upload-data-to-snowflake",
        action="store_true",
        help="Upload training data to Snowflake",
    )
    parser.add_argument(
        "--aws-credentials-block",
        type=str,
        help="Name of the AWS credentials block",
    )
    parser.add_argument(
        "--snowflake-credentials-block",
        type=str,
        help="Name of the Snowflake credentials block",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set up model parameters
    model_params = {
        "max_depth": args.max_depth,
    }
    
    # Run the flow
    logger.info("Running insurance pricing model training flow locally")
    result = train_insurance_pricing_model(
        num_policies=args.num_training_policies,
        model_type=args.model_type,
        model_params=model_params,
        output_dir=args.output_dir,
        upload_to_s3=args.upload_to_s3,
        bucket_name=args.bucket_name,
        register_in_snowflake=args.register_in_snowflake,
        upload_data_to_snowflake=args.upload_data_to_snowflake,
        aws_credentials_block=args.aws_credentials_block,
        snowflake_credentials_block=args.snowflake_credentials_block,
    )
    
    # Print results
    logger.info("Flow completed successfully")
    logger.info(f"Model saved to: {result['model_path']}")
    if result.get("s3_path"):
        logger.info(f"Model uploaded to: {result['s3_path']}")
    if result.get("model_id"):
        logger.info(f"Model registered with ID: {result['model_id']}")
    
    # Print metrics
    logger.info("Model metrics:")
    for metric, value in result["metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
