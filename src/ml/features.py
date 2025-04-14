"""
Feature engineering for the ML service.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.models import Driver, DrivingHistory, Location, Policy, Vehicle, VehicleUse

logger = logging.getLogger(__name__)


class DriverFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from driver information."""

    def __init__(self):
        self.age_bands = [18, 25, 35, 45, 55, 65, 100]
        self.age_band_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        self.gender_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.marital_status_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.occupation_risk_map = self._load_occupation_risk_map()

    def _load_occupation_risk_map(self) -> Dict[str, float]:
        """Load occupation risk map from a predefined source."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple dictionary with some example occupations
        return {
            "driver": 1.5,
            "pilot": 1.3,
            "teacher": 0.9,
            "doctor": 0.8,
            "engineer": 0.85,
            "lawyer": 0.95,
            "accountant": 0.9,
            "student": 1.2,
            "retired": 1.0,
            "unemployed": 1.1,
            # Default value for unknown occupations
            "default": 1.0,
        }

    def fit(self, X: List[Driver], y=None):
        """Fit the feature extractor to the data."""
        # Extract gender and marital status for encoding
        genders = [driver.gender.value if driver.gender else "unknown" for driver in X]
        marital_statuses = [
            driver.marital_status.value if driver.marital_status else "unknown"
            for driver in X
        ]

        # Fit the encoders
        self.gender_encoder.fit(np.array(genders).reshape(-1, 1))
        self.marital_status_encoder.fit(np.array(marital_statuses).reshape(-1, 1))

        return self

    def transform(self, X: List[Driver]) -> pd.DataFrame:
        """Transform driver information into features."""
        features = []

        for driver in X:
            # Basic driver features
            driver_features = {
                "driver_age": driver.age,
                "driver_experience": driver.driving_experience,
            }

            # Age band - create one-hot encoding for age bands instead of categorical
            age_band = pd.cut(
                [driver.age],
                bins=self.age_bands,
                labels=self.age_band_labels,
                right=False,
            )[0]

            # Convert age band to one-hot features
            for band in self.age_band_labels:
                driver_features[f"age_band_{band}"] = 1 if age_band == band else 0

            # Gender encoding
            if driver.gender:
                gender_encoded = self.gender_encoder.transform(
                    np.array([driver.gender.value]).reshape(-1, 1)
                )
                for i, col in enumerate(self.gender_encoder.categories_[0]):
                    driver_features[f"gender_{col}"] = gender_encoded[0, i]

            # Marital status encoding
            if driver.marital_status:
                marital_encoded = self.marital_status_encoder.transform(
                    np.array([driver.marital_status.value]).reshape(-1, 1)
                )
                for i, col in enumerate(self.marital_status_encoder.categories_[0]):
                    driver_features[f"marital_status_{col}"] = marital_encoded[0, i]

            # Occupation risk
            if driver.occupation:
                driver_features["occupation_risk"] = self.occupation_risk_map.get(
                    driver.occupation.lower(), self.occupation_risk_map["default"]
                )
            else:
                driver_features["occupation_risk"] = self.occupation_risk_map["default"]

            features.append(driver_features)

        return pd.DataFrame(features)


class VehicleFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from vehicle information."""

    def __init__(self):
        self.make_model_risk_map = self._load_make_model_risk_map()
        self.use_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def _load_make_model_risk_map(self) -> Dict[str, float]:
        """Load make/model risk map from a predefined source."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple dictionary with some example vehicles
        return {
            "toyota_camry": 0.9,
            "toyota_corolla": 0.85,
            "honda_civic": 0.9,
            "honda_accord": 0.95,
            "ford_f150": 1.1,
            "ford_mustang": 1.4,
            "chevrolet_silverado": 1.15,
            "chevrolet_corvette": 1.5,
            "bmw_3series": 1.2,
            "bmw_5series": 1.25,
            "mercedes_cclass": 1.2,
            "mercedes_eclass": 1.25,
            "tesla_model3": 1.1,
            "tesla_modely": 1.15,
            # Default value for unknown make/model
            "default": 1.0,
        }

    def fit(self, X: List[Vehicle], y=None):
        """Fit the feature extractor to the data."""
        # Extract vehicle use for encoding
        uses = [vehicle.primary_use.value for vehicle in X]

        # Fit the encoder
        self.use_encoder.fit(np.array(uses).reshape(-1, 1))

        return self

    def transform(self, X: List[Vehicle]) -> pd.DataFrame:
        """Transform vehicle information into features."""
        features = []

        for vehicle in X:
            # Basic vehicle features
            vehicle_features = {
                "vehicle_age": vehicle.vehicle_age,
                "vehicle_value": vehicle.value,
                "annual_mileage": vehicle.annual_mileage,
                "anti_theft_device": int(vehicle.anti_theft_device),
            }

            # Make/model risk
            make_model_key = f"{vehicle.make.lower()}_{vehicle.model.lower()}"
            vehicle_features["make_model_risk"] = self.make_model_risk_map.get(
                make_model_key, self.make_model_risk_map["default"]
            )

            # Vehicle class based on make/model - one-hot encode
            # This is a simplified approach; in reality, you'd have a more comprehensive mapping
            vehicle_classes = ["sports", "truck", "luxury", "electric", "standard"]

            if "corvette" in make_model_key or "mustang" in make_model_key:
                vehicle_class = "sports"
            elif "f150" in make_model_key or "silverado" in make_model_key:
                vehicle_class = "truck"
            elif "5series" in make_model_key or "eclass" in make_model_key:
                vehicle_class = "luxury"
            elif "model3" in make_model_key or "modely" in make_model_key:
                vehicle_class = "electric"
            else:
                vehicle_class = "standard"

            # One-hot encode vehicle class
            for vc in vehicle_classes:
                vehicle_features[f"vehicle_class_{vc}"] = (
                    1 if vehicle_class == vc else 0
                )

            # Vehicle use encoding
            use_encoded = self.use_encoder.transform(
                np.array([vehicle.primary_use.value]).reshape(-1, 1)
            )
            for i, col in enumerate(self.use_encoder.categories_[0]):
                vehicle_features[f"use_{col}"] = use_encoded[0, i]

            features.append(vehicle_features)

        return pd.DataFrame(features)


class DrivingHistoryFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from driving history information."""

    def __init__(self):
        self.incident_type_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.severity_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )

    def fit(self, X: List[DrivingHistory], y=None):
        """Fit the feature extractor to the data."""
        # Extract incident types and severities for encoding
        incident_types = [history.incident_type.value for history in X]
        severities = [history.severity.value for history in X]

        # Fit the encoders
        self.incident_type_encoder.fit(np.array(incident_types).reshape(-1, 1))
        self.severity_encoder.fit(np.array(severities).reshape(-1, 1))

        return self

    def transform(self, X: List[DrivingHistory]) -> pd.DataFrame:
        """Transform driving history information into features."""
        if not X:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                {
                    "accident_count": [0],
                    "violation_count": [0],
                    "claim_count": [0],
                    "at_fault_count": [0],
                    "total_claim_amount": [0],
                    "years_since_last_incident": [
                        10
                    ],  # Assume 10 years if no incidents
                    "incident_severity_score": [0],
                }
            )

        # Count incidents by type
        accident_count = sum(1 for h in X if h.incident_type.value == "accident")
        violation_count = sum(1 for h in X if h.incident_type.value == "violation")
        claim_count = sum(1 for h in X if h.incident_type.value == "claim")

        # Count at-fault incidents
        at_fault_count = sum(1 for h in X if h.at_fault)

        # Calculate total claim amount
        total_claim_amount = sum(h.claim_amount or 0 for h in X)

        # Calculate years since last incident
        today = date.today()
        if X:
            most_recent_date = max(h.incident_date for h in X)
            years_since_last_incident = (today.year - most_recent_date.year) - (
                (today.month, today.day)
                < (most_recent_date.month, most_recent_date.day)
            )
        else:
            years_since_last_incident = 10  # Assume 10 years if no incidents

        # Calculate incident severity score
        severity_weights = {"minor": 1, "moderate": 2, "major": 3}
        incident_severity_score = sum(severity_weights[h.severity.value] for h in X)

        features = {
            "accident_count": [accident_count],
            "violation_count": [violation_count],
            "claim_count": [claim_count],
            "at_fault_count": [at_fault_count],
            "total_claim_amount": [total_claim_amount],
            "years_since_last_incident": [years_since_last_incident],
            "incident_severity_score": [incident_severity_score],
        }

        return pd.DataFrame(features)


class LocationFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from location information."""

    def __init__(self):
        self.territory_risk_map = self._load_territory_risk_map()
        self.state_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def _load_territory_risk_map(self) -> Dict[str, Dict[str, float]]:
        """Load territory risk map from a predefined source."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple dictionary with some example territories
        return {
            "urban": {
                "crime_rate": 1.2,
                "weather_risk": 1.0,
                "traffic_density": 1.3,
            },
            "suburban": {
                "crime_rate": 0.9,
                "weather_risk": 1.0,
                "traffic_density": 1.0,
            },
            "rural": {
                "crime_rate": 0.7,
                "weather_risk": 1.1,
                "traffic_density": 0.8,
            },
            # Default value for unknown territories
            "default": {
                "crime_rate": 1.0,
                "weather_risk": 1.0,
                "traffic_density": 1.0,
            },
        }

    def _classify_territory(self, zip_code: str) -> str:
        """Classify territory based on zip code."""
        # In a real implementation, this would use a more sophisticated approach
        # For now, we'll use a simple rule-based approach
        # Assume urban areas have zip codes starting with 1, 2, or 3
        # Assume suburban areas have zip codes starting with 4, 5, or 6
        # Assume rural areas have zip codes starting with 7, 8, or 9
        first_digit = zip_code[0]
        if first_digit in ["1", "2", "3"]:
            return "urban"
        elif first_digit in ["4", "5", "6"]:
            return "suburban"
        elif first_digit in ["7", "8", "9"]:
            return "rural"
        else:
            return "default"

    def fit(self, X: List[Location], y=None):
        """Fit the feature extractor to the data."""
        # Extract states for encoding
        states = [location.state for location in X]

        # Fit the encoder
        self.state_encoder.fit(np.array(states).reshape(-1, 1))

        return self

    def transform(self, X: List[Location]) -> pd.DataFrame:
        """Transform location information into features."""
        features = []

        for location in X:
            # Classify territory
            territory = self._classify_territory(location.zip_code)

            # Get territory risk factors
            territory_risks = self.territory_risk_map.get(
                territory, self.territory_risk_map["default"]
            )

            # Basic location features
            location_features = {
                "crime_rate": territory_risks["crime_rate"],
                "weather_risk": territory_risks["weather_risk"],
                "traffic_density": territory_risks["traffic_density"],
            }

            # One-hot encode territory
            for t in ["urban", "suburban", "rural", "default"]:
                location_features[f"territory_{t}"] = 1 if territory == t else 0

            # State encoding
            state_encoded = self.state_encoder.transform(
                np.array([location.state]).reshape(-1, 1)
            )
            for i, col in enumerate(self.state_encoder.categories_[0]):
                location_features[f"state_{col}"] = state_encoded[0, i]

            features.append(location_features)

        return pd.DataFrame(features)


class PolicyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from policy information."""

    def __init__(self):
        self.driver_extractor = DriverFeatureExtractor()
        self.vehicle_extractor = VehicleFeatureExtractor()
        self.history_extractor = DrivingHistoryFeatureExtractor()
        self.location_extractor = LocationFeatureExtractor()

    def fit(self, X: List[Policy], y=None):
        """Fit the feature extractor to the data."""
        # Collect all drivers, vehicles, histories, and locations
        all_drivers = [driver for policy in X for driver in policy.drivers]
        all_vehicles = [vehicle for policy in X for vehicle in policy.vehicles]
        all_histories = [history for policy in X for history in policy.driving_history]
        all_locations = [location for policy in X for location in policy.locations]

        # Fit the extractors
        self.driver_extractor.fit(all_drivers)
        self.vehicle_extractor.fit(all_vehicles)
        self.history_extractor.fit(all_histories)
        self.location_extractor.fit(all_locations)

        return self

    def transform(self, X: List[Policy]) -> pd.DataFrame:
        """Transform policy information into features."""
        all_features = []

        for policy in X:
            # Extract features for each component
            driver_features = self.driver_extractor.transform(policy.drivers)
            vehicle_features = self.vehicle_extractor.transform(policy.vehicles)
            history_features = self.history_extractor.transform(policy.driving_history)
            location_features = self.location_extractor.transform(policy.locations)

            # Aggregate driver features (take mean for numeric, mode for categorical)
            driver_agg = {}
            for col in driver_features.columns:
                if driver_features[col].dtype in [np.float64, np.int64]:
                    driver_agg[f"driver_{col}_mean"] = driver_features[col].mean()
                    driver_agg[f"driver_{col}_max"] = driver_features[col].max()
                    driver_agg[f"driver_{col}_min"] = driver_features[col].min()
                else:
                    # For categorical features, take the mode or first value if multiple modes
                    mode_values = driver_features[col].mode()
                    if len(mode_values) > 0:
                        driver_agg[f"driver_{col}_mode"] = mode_values[0]
                    else:
                        # Fallback to first value if no mode
                        driver_agg[f"driver_{col}_mode"] = (
                            driver_features[col].iloc[0]
                            if len(driver_features) > 0
                            else 0
                        )

            # Aggregate vehicle features
            vehicle_agg = {}
            for col in vehicle_features.columns:
                if vehicle_features[col].dtype in [np.float64, np.int64]:
                    vehicle_agg[f"vehicle_{col}_mean"] = vehicle_features[col].mean()
                    vehicle_agg[f"vehicle_{col}_max"] = vehicle_features[col].max()
                    vehicle_agg[f"vehicle_{col}_min"] = vehicle_features[col].min()
                else:
                    # For categorical features, take the mode or first value if multiple modes
                    mode_values = vehicle_features[col].mode()
                    if len(mode_values) > 0:
                        vehicle_agg[f"vehicle_{col}_mode"] = mode_values[0]
                    else:
                        # Fallback to first value if no mode
                        vehicle_agg[f"vehicle_{col}_mode"] = (
                            vehicle_features[col].iloc[0]
                            if len(vehicle_features) > 0
                            else 0
                        )

            # Aggregate location features
            location_agg = {}
            for col in location_features.columns:
                if location_features[col].dtype in [np.float64, np.int64]:
                    location_agg[f"location_{col}_mean"] = location_features[col].mean()
                    location_agg[f"location_{col}_max"] = location_features[col].max()
                    location_agg[f"location_{col}_min"] = location_features[col].min()
                else:
                    # For categorical features, take the mode or first value if multiple modes
                    mode_values = location_features[col].mode()
                    if len(mode_values) > 0:
                        location_agg[f"location_{col}_mode"] = mode_values[0]
                    else:
                        # Fallback to first value if no mode
                        location_agg[f"location_{col}_mode"] = (
                            location_features[col].iloc[0]
                            if len(location_features) > 0
                            else 0
                        )

            # Combine all features
            policy_features = {
                "policy_duration_days": (
                    policy.expiration_date - policy.effective_date
                ).days,
                "num_drivers": len(policy.drivers),
                "num_vehicles": len(policy.vehicles),
                "num_locations": len(policy.locations),
                "vehicle_per_driver_ratio": len(policy.vehicles)
                / max(1, len(policy.drivers)),
                **driver_agg,
                **vehicle_agg,
                **history_features.iloc[0].to_dict(),
                **location_agg,
            }

            # Add pricing factors if available
            if policy.pricing_factors:
                if policy.pricing_factors.credit_score:
                    policy_features["credit_score"] = (
                        policy.pricing_factors.credit_score
                    )
                if policy.pricing_factors.insurance_score:
                    policy_features["insurance_score"] = (
                        policy.pricing_factors.insurance_score
                    )

            all_features.append(policy_features)

        return pd.DataFrame(all_features)


class FeatureProcessor:
    """Process features for the ML model."""

    def __init__(self):
        self.policy_extractor = PolicyFeatureExtractor()
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, policies: List[Policy], y=None):
        """Fit the feature processor to the data."""
        # Extract features
        X = self.policy_extractor.fit_transform(policies)

        # Save feature names
        self.feature_names = X.columns.tolist()

        # Fit the scaler
        self.scaler.fit(X)

        return self

    def transform(self, policies: List[Policy]) -> np.ndarray:
        """Transform policies into features for the ML model."""
        # Extract features
        X = self.policy_extractor.transform(policies)

        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        # Reorder columns to match the order used during fitting
        X = X[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit_transform(self, policies: List[Policy], y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(policies)
        return self.transform(policies)
