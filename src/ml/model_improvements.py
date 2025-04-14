"""
Enhanced ML models for insurance pricing with improved performance and explainability.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.ml.features import FeatureProcessor
from src.ml.models import Policy

logger = logging.getLogger(__name__)


class EnhancedInsurancePricingModel:
    """Enhanced ML model for insurance pricing with improved performance and explainability."""
    
    def __init__(
        self,
        model_type: str = "gradient_boosting",
        model_params: Optional[Dict] = None,
        feature_processor: Optional[FeatureProcessor] = None,
    ):
        """Initialize the model.
        
        Args:
            model_type: Type of model to use ("decision_tree", "random_forest", or "gradient_boosting")
            model_params: Parameters for the model
            feature_processor: Feature processor for transforming raw data
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.feature_processor = feature_processor or FeatureProcessor()
        self.model = None
        self.feature_importances = None
        self.shap_values = None
        self.explainer = None
        self.X_train_sample = None
    
    def _create_model(self) -> Union[DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]:
        """Create a new model instance."""
        if self.model_type == "decision_tree":
            return DecisionTreeRegressor(**self.model_params)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(
        self,
        policies: List[Policy],
        premiums: List[float],
        test_size: float = 0.2,
        random_state: int = 42,
        calculate_shap: bool = True,
    ) -> Dict[str, float]:
        """Train the model on the given data.
        
        Args:
            policies: List of policies
            premiums: List of premiums (target values)
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            calculate_shap: Whether to calculate SHAP values for explainability
            
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
        
        # Save a sample of training data for SHAP values
        if calculate_shap:
            # Limit to 100 samples for efficiency
            self.X_train_sample = X_train[:100] if len(X_train) > 100 else X_train
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Calculate feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(
                zip(self.feature_processor.feature_names, self.model.feature_importances_)
            )
        
        # Calculate SHAP values for explainability
        if calculate_shap and self.X_train_sample is not None:
            self._calculate_shap_values()
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        return metrics
    
    def _calculate_shap_values(self):
        """Calculate SHAP values for model explainability."""
        try:
            # Create SHAP explainer based on model type
            if self.model_type == "decision_tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "random_forest":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "gradient_boosting":
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer for other model types
                self.explainer = shap.KernelExplainer(self.model.predict, self.X_train_sample)
            
            # Calculate SHAP values
            self.shap_values = self.explainer.shap_values(self.X_train_sample)
            
            logger.info("SHAP values calculated successfully")
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {e}")
            self.explainer = None
            self.shap_values = None
    
    def get_shap_summary_plot(self, max_display: int = 20):
        """Generate SHAP summary plot data.
        
        Args:
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with SHAP summary data
        """
        if self.shap_values is None or self.explainer is None:
            return None
        
        # Get feature names
        feature_names = self.feature_processor.feature_names
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Sort features by importance
        indices = np.argsort(mean_abs_shap)
        indices = indices[-min(max_display, len(indices)):]
        
        # Create summary data
        summary_data = {
            "features": [feature_names[i] for i in indices],
            "importance": [float(mean_abs_shap[i]) for i in indices],
            "shap_values": self.shap_values[:, indices].tolist(),
            "feature_values": self.X_train_sample[:, indices].tolist(),
        }
        
        return summary_data
    
    def get_feature_importance_plot(self, max_display: int = 20):
        """Generate feature importance plot data.
        
        Args:
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with feature importance data
        """
        if self.feature_importances is None:
            return None
        
        # Sort features by importance
        sorted_importances = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        
        # Limit to max_display
        sorted_importances = sorted_importances[:max_display]
        
        # Create plot data
        plot_data = {
            "features": [item[0] for item in sorted_importances],
            "importance": [float(item[1]) for item in sorted_importances],
        }
        
        return plot_data
    
    def explain_prediction(self, policy: Policy):
        """Explain the prediction for a single policy.
        
        Args:
            policy: Policy to explain
            
        Returns:
            Dictionary with explanation data
        """
        if self.explainer is None:
            return None
        
        # Process features
        X = self.feature_processor.transform([policy])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Calculate SHAP values for this prediction
        shap_values = self.explainer.shap_values(X)
        
        # Get feature names
        feature_names = self.feature_processor.feature_names
        
        # Create explanation data
        explanation = {
            "prediction": float(prediction),
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0,
            "features": [],
        }
        
        # Add feature contributions
        for i, name in enumerate(feature_names):
            explanation["features"].append({
                "name": name,
                "value": float(X[0, i]),
                "contribution": float(shap_values[0, i]),
            })
        
        # Sort by absolute contribution
        explanation["features"].sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        return explanation
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Median Absolute Error (MedAE)
        medae = np.median(np.abs(y_true - y_pred))
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "medae": medae,
        }


class ModelEnsemble:
    """Ensemble of multiple models for improved prediction accuracy."""
    
    def __init__(
        self,
        models: List[EnhancedInsurancePricingModel],
        weights: Optional[List[float]] = None,
    ):
        """Initialize the ensemble.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model (if None, equal weights are used)
        """
        self.models = models
        
        # Validate and normalize weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def predict(self, policies: List[Policy]) -> np.ndarray:
        """Predict premiums using the ensemble.
        
        Args:
            policies: List of policies
            
        Returns:
            Array of predicted premiums
        """
        # Get predictions from each model
        predictions = []
        for model in self.models:
            model_preds = model.predict(policies)
            predictions.append(model_preds)
        
        # Combine predictions using weights
        ensemble_predictions = np.zeros(len(policies))
        for i, model_preds in enumerate(predictions):
            ensemble_predictions += model_preds * self.weights[i]
        
        return ensemble_predictions
    
    def predict_with_explanations(self, policies: List[Policy]) -> Tuple[np.ndarray, List[Dict]]:
        """Predict premiums and provide explanations.
        
        Args:
            policies: List of policies
            
        Returns:
            Tuple of (predictions, explanations)
        """
        # Get predictions and explanations from each model
        predictions = []
        explanations = []
        
        for i, model in enumerate(self.models):
            model_preds = model.predict(policies)
            predictions.append(model_preds)
            
            # Get explanations if available
            model_explanations = []
            for policy in policies:
                explanation = model.explain_prediction(policy)
                if explanation:
                    explanation["model_weight"] = self.weights[i]
                    model_explanations.append(explanation)
                else:
                    model_explanations.append(None)
            
            explanations.append(model_explanations)
        
        # Combine predictions using weights
        ensemble_predictions = np.zeros(len(policies))
        for i, model_preds in enumerate(predictions):
            ensemble_predictions += model_preds * self.weights[i]
        
        # Combine explanations
        combined_explanations = []
        for i in range(len(policies)):
            policy_explanations = [
                model_explanations[i] for model_explanations, model in zip(explanations, self.models)
                if model_explanations[i] is not None
            ]
            
            if policy_explanations:
                combined_explanations.append({
                    "model_explanations": policy_explanations,
                    "ensemble_prediction": float(ensemble_predictions[i]),
                })
            else:
                combined_explanations.append(None)
        
        return ensemble_predictions, combined_explanations
