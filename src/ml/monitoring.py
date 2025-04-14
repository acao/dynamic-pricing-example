"""
Real-time monitoring and model performance tracking for insurance pricing models.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import snowflake.connector
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml.model import InsurancePricingModel, PricingService
from src.ml.models import ModelVersion, Policy, PricingResponse

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and drift in production."""

    def __init__(
        self,
        model: Optional[InsurancePricingModel] = None,
        model_path: Optional[str] = None,
        snowflake_config: Optional[Dict] = None,
        metrics_window_days: int = 30,
    ):
        """Initialize the model monitor.

        Args:
            model: Insurance pricing model
            model_path: Path to a saved model
            snowflake_config: Snowflake connection configuration
            metrics_window_days: Number of days to include in metrics window
        """
        # Initialize model
        if model:
            self.model = model
        elif model_path:
            self.model = InsurancePricingModel(model_path=model_path)
        else:
            raise ValueError("Either model or model_path must be provided")

        # Initialize Snowflake connection
        self.snowflake_config = (
            snowflake_config or self._get_snowflake_config_from_env()
        )
        self.conn = None

        # Metrics window
        self.metrics_window_days = metrics_window_days

        # Performance metrics
        self.performance_metrics = {}

        # Drift metrics
        self.feature_drift_metrics = {}
        self.prediction_drift_metrics = {}

    def _get_snowflake_config_from_env(self) -> Dict:
        """Get Snowflake configuration from environment variables."""
        return {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "user": os.environ.get("SNOWFLAKE_USER"),
            "password": os.environ.get("SNOWFLAKE_PASSWORD"),
            "database": os.environ.get("SNOWFLAKE_DATABASE", "INSURANCE_PRICING"),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA", "ML_MODELS"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "INSURANCE_PRICING_WH"),
            "role": os.environ.get("SNOWFLAKE_ROLE", "INSURANCE_PRICING_APP"),
        }

    def _connect_to_snowflake(self):
        """Connect to Snowflake."""
        if self.conn is None:
            self.conn = snowflake.connector.connect(
                account=self.snowflake_config["account"],
                user=self.snowflake_config["user"],
                password=self.snowflake_config["password"],
                database=self.snowflake_config["database"],
                schema=self.snowflake_config["schema"],
                warehouse=self.snowflake_config["warehouse"],
                role=self.snowflake_config["role"],
            )

    def _close_snowflake_connection(self):
        """Close Snowflake connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def calculate_performance_metrics(self) -> Dict:
        """Calculate model performance metrics from recent pricing history.

        Returns:
            Dictionary of performance metrics
        """
        try:
            self._connect_to_snowflake()

            # Get model ID
            model_id = str(self.model.model_version.model_id)

            # Query recent pricing history
            query = f"""
                SELECT
                    policy_id,
                    input_features,
                    base_premium,
                    final_premium,
                    adjustments
                FROM
                    pricing.pricing_history
                WHERE
                    model_id = '{model_id}'
                    AND created_at >= DATEADD(day, -{self.metrics_window_days}, CURRENT_TIMESTAMP())
                ORDER BY
                    created_at DESC
            """

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Process results
            policies = []
            actual_premiums = []
            predicted_premiums = []

            for row in cursor:
                policy_id, input_features, base_premium, final_premium, adjustments = (
                    row
                )

                # Parse input features
                features = json.loads(input_features)

                # Calculate predicted premium
                # In a real implementation, we would reconstruct the policy object
                # For now, we'll use the base premium as the predicted premium
                predicted_premium = base_premium

                # Add to lists
                policies.append(policy_id)
                predicted_premiums.append(predicted_premium)
                actual_premiums.append(final_premium)

            # Calculate metrics
            if len(actual_premiums) > 0:
                mse = mean_squared_error(actual_premiums, predicted_premiums)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_premiums, predicted_premiums)

                # Calculate Mean Absolute Percentage Error (MAPE)
                mape = (
                    np.mean(
                        np.abs(
                            (np.array(actual_premiums) - np.array(predicted_premiums))
                            / np.array(actual_premiums)
                        )
                    )
                    * 100
                )

                # Calculate R-squared
                r2 = r2_score(actual_premiums, predicted_premiums)

                # Calculate bias (average error)
                bias = np.mean(np.array(predicted_premiums) - np.array(actual_premiums))

                # Store metrics
                self.performance_metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "bias": bias,
                    "sample_size": len(actual_premiums),
                    "timestamp": datetime.now().isoformat(),
                }

                return self.performance_metrics
            else:
                logger.warning("No pricing history found for the specified time window")
                return {}

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

        finally:
            self._close_snowflake_connection()

    def detect_feature_drift(self) -> Dict:
        """Detect drift in feature distributions.

        Returns:
            Dictionary of feature drift metrics
        """
        try:
            self._connect_to_snowflake()

            # Get model ID
            model_id = str(self.model.model_version.model_id)

            # Query recent pricing history
            query = f"""
                SELECT
                    input_features,
                    created_at
                FROM
                    pricing.pricing_history
                WHERE
                    model_id = '{model_id}'
                    AND created_at >= DATEADD(day, -{self.metrics_window_days}, CURRENT_TIMESTAMP())
                ORDER BY
                    created_at ASC
            """

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Process results
            features_list = []
            timestamps = []

            for row in cursor:
                input_features, created_at = row

                # Parse input features
                features = json.loads(input_features)

                # Add to lists
                features_list.append(features)
                timestamps.append(created_at)

            # Calculate drift metrics
            if len(features_list) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(features_list)

                # Add timestamp column
                df["timestamp"] = timestamps

                # Split into reference and current periods
                midpoint = len(df) // 2
                reference_df = df.iloc[:midpoint]
                current_df = df.iloc[midpoint:]

                # Calculate drift metrics for each feature
                drift_metrics = {}

                for column in df.columns:
                    if column == "timestamp":
                        continue

                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        continue

                    # Calculate statistics for reference period
                    ref_mean = reference_df[column].mean()
                    ref_std = reference_df[column].std()
                    ref_min = reference_df[column].min()
                    ref_max = reference_df[column].max()

                    # Calculate statistics for current period
                    cur_mean = current_df[column].mean()
                    cur_std = current_df[column].std()
                    cur_min = current_df[column].min()
                    cur_max = current_df[column].max()

                    # Calculate drift metrics
                    mean_diff = (cur_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
                    std_diff = (cur_std - ref_std) / ref_std if ref_std != 0 else 0
                    range_diff = (
                        ((cur_max - cur_min) - (ref_max - ref_min))
                        / (ref_max - ref_min)
                        if (ref_max - ref_min) != 0
                        else 0
                    )

                    # Calculate Population Stability Index (PSI)
                    # Divide the range into 10 bins
                    bins = np.linspace(min(ref_min, cur_min), max(ref_max, cur_max), 11)

                    # Calculate bin counts
                    ref_counts, _ = np.histogram(reference_df[column], bins=bins)
                    cur_counts, _ = np.histogram(current_df[column], bins=bins)

                    # Calculate bin proportions
                    ref_props = ref_counts / ref_counts.sum()
                    cur_props = cur_counts / cur_counts.sum()

                    # Replace zeros with small value to avoid division by zero
                    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
                    cur_props = np.where(cur_props == 0, 0.0001, cur_props)

                    # Calculate PSI
                    psi = np.sum(
                        (cur_props - ref_props) * np.log(cur_props / ref_props)
                    )

                    # Store metrics
                    drift_metrics[column] = {
                        "mean_diff": mean_diff,
                        "std_diff": std_diff,
                        "range_diff": range_diff,
                        "psi": psi,
                        "ref_mean": ref_mean,
                        "ref_std": ref_std,
                        "ref_min": ref_min,
                        "ref_max": ref_max,
                        "cur_mean": cur_mean,
                        "cur_std": cur_std,
                        "cur_min": cur_min,
                        "cur_max": cur_max,
                    }

                # Store drift metrics
                self.feature_drift_metrics = {
                    "metrics": drift_metrics,
                    "sample_size": len(df),
                    "reference_period": {
                        "start": (
                            min(reference_df["timestamp"]).isoformat()
                            if len(reference_df) > 0
                            else None
                        ),
                        "end": (
                            max(reference_df["timestamp"]).isoformat()
                            if len(reference_df) > 0
                            else None
                        ),
                    },
                    "current_period": {
                        "start": (
                            min(current_df["timestamp"]).isoformat()
                            if len(current_df) > 0
                            else None
                        ),
                        "end": (
                            max(current_df["timestamp"]).isoformat()
                            if len(current_df) > 0
                            else None
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                return self.feature_drift_metrics
            else:
                logger.warning("No pricing history found for the specified time window")
                return {}

        except Exception as e:
            logger.error(f"Error detecting feature drift: {e}")
            return {}

        finally:
            self._close_snowflake_connection()

    def detect_prediction_drift(self) -> Dict:
        """Detect drift in model predictions.

        Returns:
            Dictionary of prediction drift metrics
        """
        try:
            self._connect_to_snowflake()

            # Get model ID
            model_id = str(self.model.model_version.model_id)

            # Query recent pricing history
            query = f"""
                SELECT
                    final_premium,
                    created_at
                FROM
                    pricing.pricing_history
                WHERE
                    model_id = '{model_id}'
                    AND created_at >= DATEADD(day, -{self.metrics_window_days}, CURRENT_TIMESTAMP())
                ORDER BY
                    created_at ASC
            """

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Process results
            premiums = []
            timestamps = []

            for row in cursor:
                final_premium, created_at = row

                # Add to lists
                premiums.append(final_premium)
                timestamps.append(created_at)

            # Calculate drift metrics
            if len(premiums) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(
                    {
                        "premium": premiums,
                        "timestamp": timestamps,
                    }
                )

                # Split into reference and current periods
                midpoint = len(df) // 2
                reference_df = df.iloc[:midpoint]
                current_df = df.iloc[midpoint:]

                # Calculate statistics for reference period
                ref_mean = reference_df["premium"].mean()
                ref_std = reference_df["premium"].std()
                ref_min = reference_df["premium"].min()
                ref_max = reference_df["premium"].max()

                # Calculate statistics for current period
                cur_mean = current_df["premium"].mean()
                cur_std = current_df["premium"].std()
                cur_min = current_df["premium"].min()
                cur_max = current_df["premium"].max()

                # Calculate drift metrics
                mean_diff = (cur_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
                std_diff = (cur_std - ref_std) / ref_std if ref_std != 0 else 0
                range_diff = (
                    ((cur_max - cur_min) - (ref_max - ref_min)) / (ref_max - ref_min)
                    if (ref_max - ref_min) != 0
                    else 0
                )

                # Calculate Population Stability Index (PSI)
                # Divide the range into 10 bins
                bins = np.linspace(min(ref_min, cur_min), max(ref_max, cur_max), 11)

                # Calculate bin counts
                ref_counts, _ = np.histogram(reference_df["premium"], bins=bins)
                cur_counts, _ = np.histogram(current_df["premium"], bins=bins)

                # Calculate bin proportions
                ref_props = ref_counts / ref_counts.sum()
                cur_props = cur_counts / cur_counts.sum()

                # Replace zeros with small value to avoid division by zero
                ref_props = np.where(ref_props == 0, 0.0001, ref_props)
                cur_props = np.where(cur_props == 0, 0.0001, cur_props)

                # Calculate PSI
                psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

                # Store metrics
                self.prediction_drift_metrics = {
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "range_diff": range_diff,
                    "psi": psi,
                    "ref_mean": ref_mean,
                    "ref_std": ref_std,
                    "ref_min": ref_min,
                    "ref_max": ref_max,
                    "cur_mean": cur_mean,
                    "cur_std": cur_std,
                    "cur_min": cur_min,
                    "cur_max": cur_max,
                    "sample_size": len(df),
                    "reference_period": {
                        "start": (
                            min(reference_df["timestamp"]).isoformat()
                            if len(reference_df) > 0
                            else None
                        ),
                        "end": (
                            max(reference_df["timestamp"]).isoformat()
                            if len(reference_df) > 0
                            else None
                        ),
                    },
                    "current_period": {
                        "start": (
                            min(current_df["timestamp"]).isoformat()
                            if len(current_df) > 0
                            else None
                        ),
                        "end": (
                            max(current_df["timestamp"]).isoformat()
                            if len(current_df) > 0
                            else None
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                return self.prediction_drift_metrics
            else:
                logger.warning("No pricing history found for the specified time window")
                return {}

        except Exception as e:
            logger.error(f"Error detecting prediction drift: {e}")
            return {}

        finally:
            self._close_snowflake_connection()

    def log_metrics_to_snowflake(self) -> bool:
        """Log metrics to Snowflake.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._connect_to_snowflake()

            # Get model ID
            model_id = str(self.model.model_version.model_id)

            # Calculate metrics if not already calculated
            if not self.performance_metrics:
                self.calculate_performance_metrics()

            if not self.feature_drift_metrics:
                self.detect_feature_drift()

            if not self.prediction_drift_metrics:
                self.detect_prediction_drift()

            # Insert performance metrics
            if self.performance_metrics:
                performance_metrics_json = json.dumps(self.performance_metrics)

                query = f"""
                    INSERT INTO ml_models.model_monitoring (
                        model_id,
                        metric_type,
                        metrics,
                        created_at
                    ) VALUES (
                        '{model_id}',
                        'performance',
                        PARSE_JSON('{performance_metrics_json}'),
                        CURRENT_TIMESTAMP()
                    )
                """

                cursor = self.conn.cursor()
                cursor.execute(query)

            # Insert feature drift metrics
            if self.feature_drift_metrics:
                feature_drift_metrics_json = json.dumps(self.feature_drift_metrics)

                query = f"""
                    INSERT INTO ml_models.model_monitoring (
                        model_id,
                        metric_type,
                        metrics,
                        created_at
                    ) VALUES (
                        '{model_id}',
                        'feature_drift',
                        PARSE_JSON('{feature_drift_metrics_json}'),
                        CURRENT_TIMESTAMP()
                    )
                """

                cursor = self.conn.cursor()
                cursor.execute(query)

            # Insert prediction drift metrics
            if self.prediction_drift_metrics:
                prediction_drift_metrics_json = json.dumps(
                    self.prediction_drift_metrics
                )

                query = f"""
                    INSERT INTO ml_models.model_monitoring (
                        model_id,
                        metric_type,
                        metrics,
                        created_at
                    ) VALUES (
                        '{model_id}',
                        'prediction_drift',
                        PARSE_JSON('{prediction_drift_metrics_json}'),
                        CURRENT_TIMESTAMP()
                    )
                """

                cursor = self.conn.cursor()
                cursor.execute(query)

            # Commit changes
            self.conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error logging metrics to Snowflake: {e}")
            return False

        finally:
            self._close_snowflake_connection()

    def check_model_health(self) -> Dict:
        """Check model health based on metrics.

        Returns:
            Dictionary with model health status
        """
        # Calculate metrics if not already calculated
        if not self.performance_metrics:
            self.calculate_performance_metrics()

        if not self.feature_drift_metrics:
            self.detect_feature_drift()

        if not self.prediction_drift_metrics:
            self.detect_prediction_drift()

        # Define thresholds
        performance_thresholds = {
            "mape": 15.0,  # MAPE should be less than 15%
            "r2": 0.7,  # R-squared should be at least 0.7
            "bias": 100.0,  # Bias should be less than $100
        }

        drift_thresholds = {
            "psi": 0.2,  # PSI should be less than 0.2
            "mean_diff": 0.1,  # Mean difference should be less than 10%
        }

        # Check performance metrics
        performance_status = "healthy"
        performance_issues = []

        if self.performance_metrics:
            if self.performance_metrics.get("mape", 0) > performance_thresholds["mape"]:
                performance_status = "degraded"
                performance_issues.append(
                    f"MAPE is {self.performance_metrics['mape']:.2f}%, which is above the threshold of {performance_thresholds['mape']}%"
                )

            if self.performance_metrics.get("r2", 1) < performance_thresholds["r2"]:
                performance_status = "degraded"
                performance_issues.append(
                    f"R-squared is {self.performance_metrics['r2']:.2f}, which is below the threshold of {performance_thresholds['r2']}"
                )

            if (
                abs(self.performance_metrics.get("bias", 0))
                > performance_thresholds["bias"]
            ):
                performance_status = "degraded"
                performance_issues.append(
                    f"Bias is ${abs(self.performance_metrics['bias']):.2f}, which is above the threshold of ${performance_thresholds['bias']}"
                )

        # Check feature drift
        feature_drift_status = "healthy"
        feature_drift_issues = []

        if self.feature_drift_metrics and "metrics" in self.feature_drift_metrics:
            for feature, metrics in self.feature_drift_metrics["metrics"].items():
                if metrics.get("psi", 0) > drift_thresholds["psi"]:
                    feature_drift_status = "degraded"
                    feature_drift_issues.append(
                        f"Feature '{feature}' has PSI of {metrics['psi']:.2f}, which is above the threshold of {drift_thresholds['psi']}"
                    )

                if abs(metrics.get("mean_diff", 0)) > drift_thresholds["mean_diff"]:
                    feature_drift_status = "degraded"
                    feature_drift_issues.append(
                        f"Feature '{feature}' has mean difference of {abs(metrics['mean_diff']):.2f}, which is above the threshold of {drift_thresholds['mean_diff']}"
                    )

        # Check prediction drift
        prediction_drift_status = "healthy"
        prediction_drift_issues = []

        if self.prediction_drift_metrics:
            if self.prediction_drift_metrics.get("psi", 0) > drift_thresholds["psi"]:
                prediction_drift_status = "degraded"
                prediction_drift_issues.append(
                    f"Prediction PSI is {self.prediction_drift_metrics['psi']:.2f}, which is above the threshold of {drift_thresholds['psi']}"
                )

            if (
                abs(self.prediction_drift_metrics.get("mean_diff", 0))
                > drift_thresholds["mean_diff"]
            ):
                prediction_drift_status = "degraded"
                prediction_drift_issues.append(
                    f"Prediction mean difference is {abs(self.prediction_drift_metrics['mean_diff']):.2f}, which is above the threshold of {drift_thresholds['mean_diff']}"
                )

        # Determine overall status
        if (
            performance_status == "degraded"
            or feature_drift_status == "degraded"
            or prediction_drift_status == "degraded"
        ):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Create health report
        health_report = {
            "overall_status": overall_status,
            "performance": {
                "status": performance_status,
                "issues": performance_issues,
                "metrics": self.performance_metrics,
            },
            "feature_drift": {
                "status": feature_drift_status,
                "issues": feature_drift_issues,
                "metrics": self.feature_drift_metrics,
            },
            "prediction_drift": {
                "status": prediction_drift_status,
                "issues": prediction_drift_issues,
                "metrics": self.prediction_drift_metrics,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return health_report


class ModelDeploymentManager:
    """Manage model deployment and versioning."""

    def __init__(
        self,
        snowflake_config: Optional[Dict] = None,
        s3_bucket: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        """Initialize the model deployment manager.

        Args:
            snowflake_config: Snowflake connection configuration
            s3_bucket: S3 bucket for storing models
            api_base_url: Base URL for the API service
        """
        # Initialize Snowflake connection
        self.snowflake_config = (
            snowflake_config or self._get_snowflake_config_from_env()
        )
        self.conn = None

        # S3 bucket
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET")

        # API base URL
        self.api_base_url = api_base_url or os.environ.get("API_BASE_URL")

    def _get_snowflake_config_from_env(self) -> Dict:
        """Get Snowflake configuration from environment variables."""
        return {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "user": os.environ.get("SNOWFLAKE_USER"),
            "password": os.environ.get("SNOWFLAKE_PASSWORD"),
            "database": os.environ.get("SNOWFLAKE_DATABASE", "INSURANCE_PRICING"),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA", "ML_MODELS"),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "INSURANCE_PRICING_WH"),
            "role": os.environ.get("SNOWFLAKE_ROLE", "INSURANCE_PRICING_APP"),
        }

    def _connect_to_snowflake(self):
        """Connect to Snowflake."""
        if self.conn is None:
            self.conn = snowflake.connector.connect(
                account=self.snowflake_config["account"],
                user=self.snowflake_config["user"],
                password=self.snowflake_config["password"],
                database=self.snowflake_config["database"],
                schema=self.snowflake_config["schema"],
                warehouse=self.snowflake_config["warehouse"],
                role=self.snowflake_config["role"],
            )

    def _close_snowflake_connection(self):
        """Close Snowflake connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model.

        Returns:
            ModelVersion object for the active model, or None if no active model
        """
        try:
            self._connect_to_snowflake()

            # Query active model
            query = """
                SELECT
                    model_id,
                    model_name,
                    model_version,
                    model_path,
                    metrics,
                    created_at
                FROM
                    ml_models.active_model
            """

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Get result
            row = cursor.fetchone()

            if row:
                model_id, model_name, model_version, model_path, metrics, created_at = (
                    row
                )

                # Parse metrics
                metrics_dict = json.loads(metrics) if metrics else {}

                # Create ModelVersion object
                model = ModelVersion(
                    model_id=model_id,
                    model_name=model_name,
                    model_version=model_version,
                    model_path=model_path,
                    is_active=True,
                    metrics=metrics_dict,
                    created_at=created_at,
                )

                return model
            else:
                logger.warning("No active model found")
                return None

        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

        finally:
            self._close_snowflake_connection()

    def get_model_versions(self, limit: int = 10) -> List[ModelVersion]:
        """Get recent model versions.

        Args:
            limit: Maximum number of versions to return

        Returns:
            List of ModelVersion objects
        """
        try:
            self._connect_to_snowflake()

            # Query model versions
            query = f"""
                SELECT
                    model_id,
                    model_name,
                    model_version,
                    model_path,
                    is_active,
                    metrics,
                    created_at
                FROM
                    ml_models.model_versions
                ORDER BY
                    created_at DESC
                LIMIT {limit}
            """

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Process results
            models = []

            for row in cursor:
                (
                    model_id,
                    model_name,
                    model_version,
                    model_path,
                    is_active,
                    metrics,
                    created_at,
                ) = row

                # Parse metrics
                metrics_dict = json.loads(metrics) if metrics else {}

                # Create ModelVersion object
                model = ModelVersion(
                    model_id=model_id,
                    model_name=model_name,
                    model_version=model_version,
                    model_path=model_path,
                    is_active=is_active,
                    metrics=metrics_dict,
                    created_at=created_at,
                )

                models.append(model)

            return models

        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []

        finally:
            self._close_snowflake_connection()

    def register_model(
        self,
        model: InsurancePricingModel,
        is_active: bool = False,
    ) -> Optional[str]:
        """Register a model in Snowflake.

        Args:
            model: InsurancePricingModel object
            is_active: Whether to set this model as active

        Returns:
            Model ID if successful, None otherwise
        """
        try:
            self._connect_to_snowflake()

            # Get model information
            model_name = model.model_type
            model_version = model.model_version.model_version
            model_path = model.model_version.model_path
            metrics = json.dumps(model.model_version.metrics or {})

            # Call stored procedure to register model
            cursor = self.conn.cursor()
            cursor.execute(
                "CALL ML_MODELS.REGISTER_MODEL(%s, %s, %s, %s, %s)",
                (model_name, model_version, model_path, is_active, metrics),
            )

            # Get result
            result = cursor.fetchone()
            model_id = result[0]

            # Check if model ID is returned
            if model_id and not model_id.startswith("Error"):
                logger.info(f"Model registered with ID: {model_id}")
                return model_id
            else:
                logger.error(f"Error registering model: {model_id}")
                return None

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None

        finally:
            self._close_snowflake_connection()
