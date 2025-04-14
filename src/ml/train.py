"""
Script to train the insurance pricing model.
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generator import InsuranceDataGenerator
from src.ml.model import InsurancePricingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    num_policies: int = 1000,
    test_size: float = 0.2,
    model_type: str = "decision_tree",
    model_params: Optional[Dict] = None,
    output_dir: str = "./models",
    random_state: int = 42,
) -> str:
    """Train the insurance pricing model.
    
    Args:
        num_policies: Number of policies to generate for training
        test_size: Fraction of data to use for testing
        model_type: Type of model to use ("decision_tree" or "random_forest")
        model_params: Parameters for the model
        output_dir: Directory to save the model
        random_state: Random seed for reproducibility
        
    Returns:
        Path to the saved model
    """
    logger.info(f"Generating {num_policies} policies for training")
    
    # Generate synthetic data
    data_generator = InsuranceDataGenerator(seed=random_state)
    policies, premiums = data_generator.generate_dataset(num_policies)
    
    logger.info(f"Training {model_type} model")
    
    # Create and train model
    model = InsurancePricingModel(model_type=model_type, model_params=model_params)
    metrics = model.train(policies, premiums, test_size=test_size, random_state=random_state)
    
    logger.info(f"Model metrics: {metrics}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = model.save(output_dir)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def tune_model(
    num_policies: int = 1000,
    model_type: str = "decision_tree",
    output_dir: str = "./models",
    random_state: int = 42,
) -> Dict:
    """Tune the hyperparameters of the insurance pricing model.
    
    Args:
        num_policies: Number of policies to generate for training
        model_type: Type of model to use ("decision_tree" or "random_forest")
        output_dir: Directory to save the model
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with tuning results
    """
    logger.info(f"Generating {num_policies} policies for hyperparameter tuning")
    
    # Generate synthetic data
    data_generator = InsuranceDataGenerator(seed=random_state)
    policies, premiums = data_generator.generate_dataset(num_policies)
    
    logger.info(f"Tuning {model_type} model")
    
    # Create model
    model = InsurancePricingModel(model_type=model_type)
    
    # Define parameter grid
    if model_type == "decision_tree":
        param_grid = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        }
    elif model_type == "random_forest":
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Tune hyperparameters
    tuning_results = model.tune_hyperparameters(
        policies, premiums, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    
    logger.info(f"Best parameters: {tuning_results['best_params']}")
    logger.info(f"Best score: {tuning_results['best_score']}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = model.save(output_dir)
    
    logger.info(f"Model saved to {model_path}")
    
    return tuning_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train the insurance pricing model")
    parser.add_argument(
        "--num-policies", type=int, default=1000, help="Number of policies to generate for training"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "random_forest"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./models", help="Directory to save the model"
    )
    parser.add_argument(
        "--tune", action="store_true", help="Tune hyperparameters"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.tune:
        tune_model(
            num_policies=args.num_policies,
            model_type=args.model_type,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
    else:
        train_model(
            num_policies=args.num_policies,
            test_size=args.test_size,
            model_type=args.model_type,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
