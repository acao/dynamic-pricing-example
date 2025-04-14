"""
ML model implementation for insurance pricing.
"""
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.ml.features import FeatureProcessor
from src.ml.models import ModelVersion, Policy, PricingFactors

logger = logging.getLogger(__name__)


class InsurancePricingModel:
    """Decision tree-based model for insurance pricing."""
    
    def __init__(
        self,
        model_type: str = "decision_tree",
        model_params: Optional[Dict] = None,
        feature_processor: Optional[FeatureProcessor] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the model.
        
        Args:
            model_type: Type of model to use ("decision_tree" or "random_forest")
            model_params: Parameters for the model
            feature_processor: Feature processor for transforming raw data
            model_path: Path to a saved model to load
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.feature_processor = feature_processor or FeatureProcessor()
        self.model = None
        self.feature_importances = None
        self.model_version = None
        
        # Load model if path is provided
        if model_path:
            self.load(model_path)
    
    def _create_model(self) -> Union[DecisionTreeRegressor, RandomForestRegressor]:
        """Create a new model instance."""
        if self.model_type == "decision_tree":
            return DecisionTreeRegressor(**self.model_params)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(
        self,
        policies: List[Policy],
        premiums: List[float],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Train the model on the given data.
        
        Args:
            policies: List of policies
            premiums: List of premiums (target values)
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Process features
        X = self.feature_processor.fit_transform(policies)
        y = np.array(premiums)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Calculate feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(
                zip(self.feature_processor.feature_names, self.model.feature_importances_)
            )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Create model version
        self.model_version = ModelVersion(
            model_name=self.model_type,
            model_version=datetime.now().strftime("%Y%m%d%H%M%S"),
            model_path="",  # Will be set when saved
            is_active=True,
            metrics=metrics,
        )
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    
    def predict(self, policies: List[Policy]) -> np.ndarray:
        """Predict premiums for the given policies."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Process features
        X = self.feature_processor.transform(policies)
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_with_factors(
        self, policies: List[Policy]
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Predict premiums and return pricing factors."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Process features
        X = self.feature_processor.transform(policies)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate pricing factors
        factors_list = []
        
        if hasattr(self.model, "feature_importances_") and self.feature_processor.feature_names:
            # For each policy, calculate the contribution of each feature group
            for i in range(len(policies)):
                # Get feature values for this policy
                feature_values = X[i]
                
                # Calculate contribution of each feature
                contributions = feature_values * self.model.feature_importances_
                
                # Group contributions by feature type
                driver_contribution = sum(
                    contributions[j] for j, name in enumerate(self.feature_processor.feature_names)
                    if name.startswith("driver_")
                )
                
                vehicle_contribution = sum(
                    contributions[j] for j, name in enumerate(self.feature_processor.feature_names)
                    if name.startswith("vehicle_")
                )
                
                history_contribution = sum(
                    contributions[j] for j, name in enumerate(self.feature_processor.feature_names)
                    if name in [
                        "accident_count", "violation_count", "claim_count",
                        "at_fault_count", "total_claim_amount", "years_since_last_incident",
                        "incident_severity_score"
                    ]
                )
                
                location_contribution = sum(
                    contributions[j] for j, name in enumerate(self.feature_processor.feature_names)
                    if name.startswith("location_") or name in [
                        "crime_rate", "weather_risk", "traffic_density"
                    ]
                )
                
                # Normalize factors to sum to 1
                total_contribution = (
                    driver_contribution + vehicle_contribution +
                    history_contribution + location_contribution
                )
                
                if total_contribution > 0:
                    driver_factor = driver_contribution / total_contribution
                    vehicle_factor = vehicle_contribution / total_contribution
                    history_factor = history_contribution / total_contribution
                    location_factor = location_contribution / total_contribution
                else:
                    # Equal weights if no contribution
                    driver_factor = 0.25
                    vehicle_factor = 0.25
                    history_factor = 0.25
                    location_factor = 0.25
                
                factors = {
                    "driver_factor": driver_factor,
                    "vehicle_factor": vehicle_factor,
                    "history_factor": history_factor,
                    "location_factor": location_factor,
                }
                
                factors_list.append(factors)
        else:
            # If feature importances not available, use equal weights
            for _ in range(len(policies)):
                factors_list.append({
                    "driver_factor": 0.25,
                    "vehicle_factor": 0.25,
                    "history_factor": 0.25,
                    "location_factor": 0.25,
                })
        
        return predictions, factors_list
    
    def tune_hyperparameters(
        self,
        policies: List[Policy],
        premiums: List[float],
        param_grid: Dict,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
    ) -> Dict:
        """Tune hyperparameters using grid search."""
        # Process features
        X = self.feature_processor.fit_transform(policies)
        y = np.array(premiums)
        
        # Create base model
        base_model = self._create_model()
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Update model parameters
        self.model_params = grid_search.best_params_
        
        # Train model with best parameters
        self.model = self._create_model()
        self.model.fit(X, y)
        
        # Calculate feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(
                zip(self.feature_processor.feature_names, self.model.feature_importances_)
            )
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }
    
    def save(self, directory: str) -> str:
        """Save the model to the given directory."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.model_type}_{timestamp}.joblib"
        filepath = os.path.join(directory, filename)
        
        # Save model and feature processor
        model_data = {
            "model": self.model,
            "feature_processor": self.feature_processor,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "feature_importances": self.feature_importances,
            "model_version": self.model_version,
        }
        
        joblib.dump(model_data, filepath)
        
        # Update model version path
        if self.model_version:
            self.model_version.model_path = filepath
        
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """Load the model from the given file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Extract components
        self.model = model_data["model"]
        self.feature_processor = model_data["feature_processor"]
        self.model_type = model_data["model_type"]
        self.model_params = model_data["model_params"]
        self.feature_importances = model_data["feature_importances"]
        
        # Handle model version - create one if it doesn't exist
        if "model_version" in model_data and model_data["model_version"]:
            self.model_version = model_data["model_version"]
        else:
            # Create a new model version
            self.model_version = ModelVersion(
                model_name=self.model_type,
                model_version=os.path.basename(filepath),
                model_path=filepath,
                is_active=True,
                metrics=None,
            )
        
        logger.info(f"Model loaded from {filepath}")


class PricingService:
    """Service for insurance pricing."""
    
    def __init__(
        self,
        model: Optional[InsurancePricingModel] = None,
        model_path: Optional[str] = None,
        base_premium: float = 1000.0,
    ):
        """Initialize the pricing service.
        
        Args:
            model: Insurance pricing model
            model_path: Path to a saved model to load
            base_premium: Base premium for pricing
        """
        if model:
            self.model = model
        elif model_path:
            self.model = InsurancePricingModel(model_path=model_path)
        else:
            self.model = InsurancePricingModel()
        
        self.base_premium = base_premium
    
    def calculate_premium(
        self, policy: Policy, apply_factors: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate premium for the given policy."""
        # Make prediction
        predictions, factors_list = self.model.predict_with_factors([policy])
        
        # Get base prediction
        base_prediction = predictions[0]
        
        # Apply base premium with scaling to ensure reasonable values
        # Scale the prediction to a reasonable range (0.5 to 2.0)
        scaled_prediction = min(max(base_prediction / 1000, 0.5), 2.0)
        base_premium = self.base_premium * scaled_prediction
        
        # Get factors
        factors = factors_list[0]
        
        # Apply pricing factors if requested
        if apply_factors and policy.pricing_factors:
            # Apply credit score factor
            if policy.pricing_factors.credit_score:
                credit_factor = self._calculate_credit_factor(policy.pricing_factors.credit_score)
                factors["credit_factor"] = credit_factor
            else:
                credit_factor = 1.0
            
            # Apply insurance score factor
            if policy.pricing_factors.insurance_score:
                insurance_factor = self._calculate_insurance_factor(
                    policy.pricing_factors.insurance_score
                )
                factors["insurance_factor"] = insurance_factor
            else:
                insurance_factor = 1.0
            
            # Apply territory factor
            if policy.pricing_factors.territory_factor:
                territory_factor = policy.pricing_factors.territory_factor
                factors["territory_factor"] = territory_factor
            else:
                territory_factor = 1.0
            
            # Calculate final premium
            final_premium = base_premium * credit_factor * insurance_factor * territory_factor
        else:
            # Use base premium as final premium
            final_premium = base_premium
        
        return final_premium, factors
    
    def _calculate_credit_factor(self, credit_score: float) -> float:
        """Calculate factor based on credit score."""
        # Credit scores typically range from 300 to 850
        # Lower scores result in higher factors (more expensive)
        if credit_score >= 800:
            return 0.8  # Excellent credit
        elif credit_score >= 740:
            return 0.9  # Very good credit
        elif credit_score >= 670:
            return 1.0  # Good credit
        elif credit_score >= 580:
            return 1.2  # Fair credit
        else:
            return 1.5  # Poor credit
    
    def _calculate_insurance_factor(self, insurance_score: float) -> float:
        """Calculate factor based on insurance score."""
        # Insurance scores typically range from 0 to 100
        # Lower scores result in higher factors (more expensive)
        if insurance_score >= 90:
            return 0.8  # Excellent insurance history
        elif insurance_score >= 80:
            return 0.9  # Very good insurance history
        elif insurance_score >= 70:
            return 1.0  # Good insurance history
        elif insurance_score >= 60:
            return 1.1  # Fair insurance history
        elif insurance_score >= 50:
            return 1.2  # Below average insurance history
        else:
            return 1.3  # Poor insurance history
