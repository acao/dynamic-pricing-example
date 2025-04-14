"""
Script to deploy the insurance pricing model to AWS using Prefect.
"""
import argparse
import json
import logging
import os
from datetime import datetime

from prefect_aws.credentials import AwsCredentials
from prefect_snowflake.database import SnowflakeConnector

from infrastructure.prefect.flows import create_aws_deployment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_aws_credentials(credentials_file: str = None) -> dict:
    """Load AWS credentials from a file or environment variables.
    
    Args:
        credentials_file: Path to the credentials file
        
    Returns:
        Dictionary with AWS credentials
    """
    if credentials_file and os.path.exists(credentials_file):
        logger.info(f"Loading AWS credentials from {credentials_file}")
        with open(credentials_file, "r") as f:
            credentials = json.load(f)
    else:
        logger.info("Loading AWS credentials from environment variables")
        credentials = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
    
    # Validate credentials
    if not credentials.get("aws_access_key_id") or not credentials.get("aws_secret_access_key"):
        raise ValueError("AWS credentials not found")
    
    return credentials


def load_snowflake_credentials(credentials_file: str = None) -> dict:
    """Load Snowflake credentials from a file or environment variables.
    
    Args:
        credentials_file: Path to the credentials file
        
    Returns:
        Dictionary with Snowflake credentials
    """
    if credentials_file and os.path.exists(credentials_file):
        logger.info(f"Loading Snowflake credentials from {credentials_file}")
        with open(credentials_file, "r") as f:
            credentials = json.load(f)
    else:
        logger.info("Loading Snowflake credentials from environment variables")
        credentials = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "user": os.environ.get("SNOWFLAKE_USER"),
            "password": os.environ.get("SNOWFLAKE_PASSWORD"),
            "database": os.environ.get("SNOWFLAKE_DATABASE"),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
            "role": os.environ.get("SNOWFLAKE_ROLE"),
        }
    
    # Validate credentials
    required_fields = ["account", "user", "password", "database", "schema", "warehouse"]
    missing_fields = [field for field in required_fields if not credentials.get(field)]
    if missing_fields:
        raise ValueError(f"Missing Snowflake credentials: {', '.join(missing_fields)}")
    
    return credentials


def create_snowflake_connector(credentials: dict) -> SnowflakeConnector:
    """Create a Snowflake connector.
    
    Args:
        credentials: Snowflake credentials
        
    Returns:
        Snowflake connector
    """
    logger.info("Creating Snowflake connector")
    
    # Create connector
    connector = SnowflakeConnector(
        account=credentials["account"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"],
        schema=credentials["schema"],
        warehouse=credentials["warehouse"],
        role=credentials.get("role"),
    )
    
    # Save connector
    connector.save("insurance-pricing-snowflake", overwrite=True)
    
    return connector


def deploy_to_aws(
    bucket_name: str,
    aws_credentials_file: str = None,
    snowflake_credentials_file: str = None
):
    """Deploy the insurance pricing model to AWS.
    
    Args:
        bucket_name: Name of the S3 bucket
        aws_credentials_file: Path to the AWS credentials file
        snowflake_credentials_file: Path to the Snowflake credentials file
    """
    logger.info(f"Deploying insurance pricing model to AWS S3 bucket {bucket_name}")
    
    # Load credentials
    aws_credentials = load_aws_credentials(aws_credentials_file)
    snowflake_credentials = load_snowflake_credentials(snowflake_credentials_file)
    
    # Create Snowflake connector
    snowflake_connector = create_snowflake_connector(snowflake_credentials)
    
    # Create AWS deployment
    create_aws_deployment(
        bucket_name=bucket_name,
        aws_credentials=aws_credentials,
        snowflake_connector=snowflake_connector
    )
    
    logger.info("AWS deployment created successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy insurance pricing model to AWS")
    parser.add_argument(
        "--bucket-name", type=str, required=True,
        help="Name of the S3 bucket"
    )
    parser.add_argument(
        "--aws-credentials", type=str, default=None,
        help="Path to the AWS credentials file"
    )
    parser.add_argument(
        "--snowflake-credentials", type=str, default=None,
        help="Path to the Snowflake credentials file"
    )
    
    args = parser.parse_args()
    
    deploy_to_aws(
        bucket_name=args.bucket_name,
        aws_credentials_file=args.aws_credentials,
        snowflake_credentials_file=args.snowflake_credentials
    )


if __name__ == "__main__":
    main()
