"""
Enhanced feature engineering for insurance pricing models.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler

from src.ml.features import DriverFeatureExtractor, FeatureProcessor, LocationFeatureExtractor, VehicleFeatureExtractor
from src.ml.models import Driver, DrivingHistory, Location, Policy, Vehicle

logger = logging.getLogger(__name__)


class AdvancedDriverFeatureExtractor(DriverFeatureExtractor):
    """Enhanced feature extraction for driver information."""
    
    def __init__(self):
        super().__init__()
        # Additional risk factors
        self.age_risk_curve = self._create_age_risk_curve()
        self.experience_risk_curve = self._create_experience_risk_curve()
    
    def _create_age_risk_curve(self) -> Dict[int, float]:
        """Create a risk curve based on driver age.
        
        This implements a more sophisticated U-shaped risk curve where
        very young and very old drivers have higher risk.
        """
        # Create a dictionary mapping age to risk factor
        risk_curve = {}
        
        # Young drivers (high risk)
        for age in range(16, 25):
            risk_curve[age] = 1.5 - 0.05 * (age - 16)  # Decreasing from 1.5 to 1.05
        
        # Middle-aged drivers (low risk)
        for age in range(25, 65):
            # Slight U-shape with minimum at age 45
            risk_factor = 1.0 + 0.005 * min(age - 25, 65 - age)
            risk_curve[age] = risk_factor
        
        # Older drivers (increasing risk)
        for age in range(65, 100):
            risk_curve[age] = 1.0 + 0.02 * (age - 65)  # Increasing from 1.0 to 1.7
        
        return risk_curve
    
    def _create_experience_risk_curve(self) -> Dict[int, float]:
        """Create a risk curve based on driving experience."""
        # Create a dictionary mapping years of experience to risk factor
        risk_curve = {}
        
        # New drivers (high risk)
        for years in range(0, 3):
            risk_curve[years] = 1.5 - 0.1 * years  # Decreasing from 1.5 to 1.3
        
        # Moderate experience (medium risk)
        for years in range(3, 10):
            risk_curve[years] = 1.3 - 0.04 * (years - 3)  # Decreasing from 1.3 to 1.02
        
        # Experienced drivers (low risk)
        for years in range(10, 60):
            risk_curve[years] = 1.0  # Constant low risk
        
        return risk_curve
    
    def transform(self, X: List[Driver]) -> pd.DataFrame:
        """Transform driver information into features with enhanced risk factors."""
        # Get basic features from parent class
        features_df = super().transform(X)
        
        # Add enhanced risk factors
        for i, driver in enumerate(X):
            # Age risk factor (more sophisticated U-shaped curve)
            age = driver.age
            age_risk = self.age_risk_curve.get(age, 1.5)  # Default to high risk if age is out of range
            features_df.at[i, 'age_risk_factor'] = age_risk
            
            # Experience risk factor (exponential decay)
            experience = driver.driving_experience
            exp_risk = self.experience_risk_curve.get(experience, 1.5)  # Default to high risk if experience is out of range
            features_df.at[i, 'experience_risk_factor'] = exp_risk
            
            # License recency factor (how recently they got their license, regardless of age)
            license_recency = (driver.age - driver.driving_experience) / driver.age if driver.age > 0 else 1.0
            features_df.at[i, 'license_recency_factor'] = license_recency
            
            # Interaction features
            features_df.at[i, 'age_experience_ratio'] = driver.age / max(1, driver.driving_experience)
        
        return features_df


class AdvancedVehicleFeatureExtractor(VehicleFeatureExtractor):
    """Enhanced feature extraction for vehicle information."""
    
    def __init__(self):
        super().__init__()
        # Additional risk factors
        self.value_depreciation_curve = self._create_value_depreciation_curve()
        self.mileage_risk_curve = self._create_mileage_risk_curve()
        
        # Enhanced make/model risk map with more granular categories
        self.vehicle_category_map = self._create_vehicle_category_map()
    
    def _create_value_depreciation_curve(self) -> Dict[int, float]:
        """Create a depreciation curve based on vehicle age."""
        # Create a dictionary mapping vehicle age to depreciation factor
        depreciation_curve = {}
        
        # New vehicles (rapid depreciation)
        depreciation_curve[0] = 1.0  # No depreciation for brand new
        depreciation_curve[1] = 0.8  # 20% depreciation after first year
        depreciation_curve[2] = 0.7  # 30% depreciation after second year
        depreciation_curve[3] = 0.65  # 35% depreciation after third year
        
        # Middle-aged vehicles (moderate depreciation)
        for age in range(4, 10):
            depreciation_curve[age] = 0.65 - 0.025 * (age - 3)  # Decreasing from 0.65 to 0.475
        
        # Older vehicles (slow depreciation)
        for age in range(10, 30):
            depreciation_curve[age] = 0.45 - 0.01 * (age - 10)  # Decreasing from 0.45 to 0.25
        
        return depreciation_curve
    
    def _create_mileage_risk_curve(self) -> Dict[int, float]:
        """Create a risk curve based on annual mileage."""
        # Create a dictionary mapping annual mileage to risk factor
        risk_curve = {}
        
        # Low mileage (low risk)
        for miles in range(0, 5001, 1000):
            risk_curve[miles] = 0.8 + 0.04 * (miles / 1000)  # Increasing from 0.8 to 1.0
        
        # Average mileage (medium risk)
        for miles in range(5001, 15001, 1000):
            risk_curve[miles] = 1.0 + 0.02 * ((miles - 5000) / 1000)  # Increasing from 1.0 to 1.2
        
        # High mileage (high risk)
        for miles in range(15001, 50001, 1000):
            risk_curve[miles] = 1.2 + 0.01 * ((miles - 15000) / 1000)  # Increasing from 1.2 to 1.55
        
        return risk_curve
    
    def _create_vehicle_category_map(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """Create a more detailed vehicle category map."""
        # This would typically be loaded from a database with comprehensive vehicle data
        # For now, we'll create a simplified version with common makes/models
        
        # Categories: economy, standard, luxury, sports, suv, truck, van, electric
        category_map = {}
        
        # Toyota
        category_map["toyota_corolla"] = {"category": "economy", "risk_factor": 0.9, "theft_risk": 0.8}
        category_map["toyota_camry"] = {"category": "standard", "risk_factor": 0.95, "theft_risk": 0.85}
        category_map["toyota_rav4"] = {"category": "suv", "risk_factor": 1.0, "theft_risk": 0.9}
        category_map["toyota_highlander"] = {"category": "suv", "risk_factor": 1.05, "theft_risk": 0.9}
        category_map["toyota_tacoma"] = {"category": "truck", "risk_factor": 1.1, "theft_risk": 1.0}
        
        # Honda
        category_map["honda_civic"] = {"category": "economy", "risk_factor": 0.9, "theft_risk": 1.0}
        category_map["honda_accord"] = {"category": "standard", "risk_factor": 0.95, "theft_risk": 0.9}
        category_map["honda_cr-v"] = {"category": "suv", "risk_factor": 1.0, "theft_risk": 0.85}
        category_map["honda_pilot"] = {"category": "suv", "risk_factor": 1.05, "theft_risk": 0.9}
        category_map["honda_odyssey"] = {"category": "van", "risk_factor": 1.0, "theft_risk": 0.8}
        
        # Ford
        category_map["ford_focus"] = {"category": "economy", "risk_factor": 0.95, "theft_risk": 0.9}
        category_map["ford_fusion"] = {"category": "standard", "risk_factor": 1.0, "theft_risk": 0.9}
        category_map["ford_escape"] = {"category": "suv", "risk_factor": 1.05, "theft_risk": 0.95}
        category_map["ford_explorer"] = {"category": "suv", "risk_factor": 1.1, "theft_risk": 1.0}
        category_map["ford_f-150"] = {"category": "truck", "risk_factor": 1.15, "theft_risk": 1.1}
        category_map["ford_mustang"] = {"category": "sports", "risk_factor": 1.4, "theft_risk": 1.2}
        
        # Chevrolet
        category_map["chevrolet_spark"] = {"category": "economy", "risk_factor": 0.95, "theft_risk": 0.85}
        category_map["chevrolet_malibu"] = {"category": "standard", "risk_factor": 1.0, "theft_risk": 0.9}
        category_map["chevrolet_equinox"] = {"category": "suv", "risk_factor": 1.05, "theft_risk": 0.95}
        category_map["chevrolet_tahoe"] = {"category": "suv", "risk_factor": 1.15, "theft_risk": 1.05}
        category_map["chevrolet_silverado"] = {"category": "truck", "risk_factor": 1.2, "theft_risk": 1.1}
        category_map["chevrolet_corvette"] = {"category": "sports", "risk_factor": 1.5, "theft_risk": 1.3}
        
        # Luxury brands
        category_map["bmw_3series"] = {"category": "luxury", "risk_factor": 1.2, "theft_risk": 1.1}
        category_map["bmw_5series"] = {"category": "luxury", "risk_factor": 1.3, "theft_risk": 1.15}
        category_map["bmw_x3"] = {"category": "luxury_suv", "risk_factor": 1.25, "theft_risk": 1.1}
        category_map["bmw_x5"] = {"category": "luxury_suv", "risk_factor": 1.35, "theft_risk": 1.2}
        
        category_map["mercedes_cclass"] = {"category": "luxury", "risk_factor": 1.25, "theft_risk": 1.15}
        category_map["mercedes_eclass"] = {"category": "luxury", "risk_factor": 1.35, "theft_risk": 1.2}
        category_map["mercedes_glc"] = {"category": "luxury_suv", "risk_factor": 1.3, "theft_risk": 1.15}
        category_map["mercedes_gle"] = {"category": "luxury_suv", "risk_factor": 1.4, "theft_risk": 1.25}
        
        # Electric vehicles
        category_map["tesla_model3"] = {"category": "electric", "risk_factor": 1.15, "theft_risk": 1.0}
        category_map["tesla_modely"] = {"category": "electric_suv", "risk_factor": 1.2, "theft_risk": 1.05}
        category_map["tesla_models"] = {"category": "luxury_electric", "risk_factor": 1.3, "theft_risk": 1.1}
        category_map["tesla_modelx"] = {"category": "luxury_electric_suv", "risk_factor": 1.35, "theft_risk": 1.15}
        
        # Default for unknown vehicles
        category_map["default"] = {"category": "standard", "risk_factor": 1.0, "theft_risk": 1.0}
        
        return category_map
    
    def transform(self, X: List[Vehicle]) -> pd.DataFrame:
        """Transform vehicle information into features with enhanced risk factors."""
        # Get basic features from parent class
        features_df = super().transform(X)
        
        # Add enhanced features
        for i, vehicle in enumerate(X):
            # Vehicle category and risk factors
            make_model_key = f"{vehicle.make.lower()}_{vehicle.model.lower()}"
            vehicle_info = self.vehicle_category_map.get(make_model_key, self.vehicle_category_map["default"])
            
            features_df.at[i, 'vehicle_category'] = vehicle_info["category"]
            features_df.at[i, 'category_risk_factor'] = vehicle_info["risk_factor"]
            features_df.at[i, 'theft_risk_factor'] = vehicle_info["theft_risk"] * (0.7 if vehicle.anti_theft_device else 1.0)
            
            # Value depreciation
            depreciation_factor = self.value_depreciation_curve.get(
                vehicle.vehicle_age, 0.25  # Default to 25% of original value for very old vehicles
            )
            features_df.at[i, 'depreciated_value'] = vehicle.value * depreciation_factor
            
            # Mileage risk
            # Round to nearest 1000 for lookup
            mileage_key = (vehicle.annual_mileage // 1000) * 1000
            mileage_risk = self.mileage_risk_curve.get(
                mileage_key, 1.5  # Default to high risk for very high mileage
            )
            features_df.at[i, 'mileage_risk_factor'] = mileage_risk
            
            # Value-to-age ratio (higher value cars that are older might indicate luxury vehicles)
            features_df.at[i, 'value_age_ratio'] = vehicle.value / max(1, vehicle.vehicle_age)
            
            # Interaction features
            features_df.at[i, 'value_mileage_factor'] = (vehicle.value / 10000) * (vehicle.annual_mileage / 10000)
        
        return features_df


class AdvancedLocationFeatureExtractor(LocationFeatureExtractor):
    """Enhanced feature extraction for location information."""
    
    def __init__(self):
        super().__init__()
        # Enhanced territory risk map with more factors
        self.enhanced_territory_risk_map = self._create_enhanced_territory_risk_map()
        
        # State risk factors based on historical data
        self.state_risk_factors = self._create_state_risk_factors()
    
    def _create_enhanced_territory_risk_map(self) -> Dict[str, Dict[str, float]]:
        """Create an enhanced territory risk map with more factors."""
        # In a real implementation, this would be based on actual data by zip code
        # For now, we'll create a more detailed version of the existing map
        return {
            'urban_high_density': {
                'crime_rate': 1.3,
                'weather_risk': 1.0,
                'traffic_density': 1.4,
                'parking_risk': 1.3,
                'vandalism_risk': 1.25,
                'flood_risk': 1.1,
            },
            'urban_medium_density': {
                'crime_rate': 1.2,
                'weather_risk': 1.0,
                'traffic_density': 1.3,
                'parking_risk': 1.2,
                'vandalism_risk': 1.15,
                'flood_risk': 1.05,
            },
            'urban_low_density': {
                'crime_rate': 1.1,
                'weather_risk': 1.0,
                'traffic_density': 1.2,
                'parking_risk': 1.1,
                'vandalism_risk': 1.1,
                'flood_risk': 1.0,
            },
            'suburban_high_income': {
                'crime_rate': 0.8,
                'weather_risk': 1.0,
                'traffic_density': 1.0,
                'parking_risk': 0.9,
                'vandalism_risk': 0.85,
                'flood_risk': 0.95,
            },
            'suburban_medium_income': {
                'crime_rate': 0.9,
                'weather_risk': 1.0,
                'traffic_density': 1.0,
                'parking_risk': 0.95,
                'vandalism_risk': 0.9,
                'flood_risk': 1.0,
            },
            'suburban_low_income': {
                'crime_rate': 1.0,
                'weather_risk': 1.0,
                'traffic_density': 1.0,
                'parking_risk': 1.0,
                'vandalism_risk': 1.0,
                'flood_risk': 1.0,
            },
            'rural_remote': {
                'crime_rate': 0.6,
                'weather_risk': 1.2,
                'traffic_density': 0.7,
                'parking_risk': 0.7,
                'vandalism_risk': 0.7,
                'flood_risk': 1.1,
            },
            'rural_town': {
                'crime_rate': 0.7,
                'weather_risk': 1.1,
                'traffic_density': 0.8,
                'parking_risk': 0.8,
                'vandalism_risk': 0.8,
                'flood_risk': 1.05,
            },
            'coastal': {
                'crime_rate': 1.0,
                'weather_risk': 1.2,
                'traffic_density': 1.0,
                'parking_risk': 1.0,
                'vandalism_risk': 1.0,
                'flood_risk': 1.3,
            },
            'mountain': {
                'crime_rate': 0.8,
                'weather_risk': 1.3,
                'traffic_density': 0.9,
                'parking_risk': 0.9,
                'vandalism_risk': 0.8,
                'flood_risk': 1.1,
            },
            # Default value for unknown territories
            'default': {
                'crime_rate': 1.0,
                'weather_risk': 1.0,
                'traffic_density': 1.0,
                'parking_risk': 1.0,
                'vandalism_risk': 1.0,
                'flood_risk': 1.0,
            }
        }
    
    def _create_state_risk_factors(self) -> Dict[str, float]:
        """Create state risk factors based on historical data."""
        # In a real implementation, this would be based on actual historical data
        # For now, we'll create a simplified version with some example states
        return {
            'CA': 1.2,  # California - higher risk due to traffic, theft
            'NY': 1.15,  # New York - higher risk due to urban density
            'FL': 1.1,  # Florida - higher risk due to weather events
            'TX': 1.05,  # Texas - slightly higher risk
            'MI': 1.1,  # Michigan - higher risk due to weather, road conditions
            'LA': 1.15,  # Louisiana - higher risk due to weather events
            'NJ': 1.1,  # New Jersey - higher risk due to traffic density
            'IL': 1.05,  # Illinois - slightly higher risk
            'GA': 1.05,  # Georgia - slightly higher risk
            'PA': 1.0,  # Pennsylvania - average risk
            'OH': 1.0,  # Ohio - average risk
            'NC': 1.0,  # North Carolina - average risk
            'VA': 0.95,  # Virginia - slightly lower risk
            'WA': 1.0,  # Washington - average risk
            'MA': 1.05,  # Massachusetts - slightly higher risk
            'AZ': 0.95,  # Arizona - slightly lower risk
            'IN': 0.95,  # Indiana - slightly lower risk
            'TN': 1.0,  # Tennessee - average risk
            'MO': 0.95,  # Missouri - slightly lower risk
            'MD': 1.05,  # Maryland - slightly higher risk
            'WI': 1.0,  # Wisconsin - average risk
            'MN': 1.0,  # Minnesota - average risk
            'CO': 1.0,  # Colorado - average risk
            'AL': 1.0,  # Alabama - average risk
            'SC': 1.0,  # South Carolina - average risk
            'LA': 1.1,  # Louisiana - higher risk due to weather events
            'KY': 0.95,  # Kentucky - slightly lower risk
            'OR': 0.95,  # Oregon - slightly lower risk
            'OK': 1.0,  # Oklahoma - average risk
            'CT': 1.05,  # Connecticut - slightly higher risk
            'IA': 0.9,  # Iowa - lower risk
            'MS': 1.0,  # Mississippi - average risk
            'AR': 0.95,  # Arkansas - slightly lower risk
            'KS': 0.9,  # Kansas - lower risk
            'UT': 0.9,  # Utah - lower risk
            'NV': 1.0,  # Nevada - average risk
            'NM': 0.95,  # New Mexico - slightly lower risk
            'NE': 0.9,  # Nebraska - lower risk
            'WV': 0.95,  # West Virginia - slightly lower risk
            'ID': 0.9,  # Idaho - lower risk
            'HI': 1.0,  # Hawaii - average risk
            'ME': 0.95,  # Maine - slightly lower risk
            'NH': 0.95,  # New Hampshire - slightly lower risk
            'RI': 1.05,  # Rhode Island - slightly higher risk
            'MT': 0.9,  # Montana - lower risk
            'DE': 1.0,  # Delaware - average risk
            'SD': 0.9,  # South Dakota - lower risk
            'ND': 0.9,  # North Dakota - lower risk
            'AK': 1.05,  # Alaska - slightly higher risk due to weather
            'VT': 0.95,  # Vermont - slightly lower risk
            'WY': 0.9,  # Wyoming - lower risk
            'DC': 1.2,  # District of Columbia - higher risk due to urban density
        }
    
    def _classify_enhanced_territory(self, zip_code: str, state: str) -> str:
        """Classify territory with more granular categories."""
        # In a real implementation, this would use a comprehensive database of zip codes
        # For now, we'll use a more sophisticated rule-based approach
        
        # First digit of zip code gives a rough geographic region
        first_digit = zip_code[0]
        
        # Second digit gives a more specific area within the region
        second_digit = zip_code[1]
        
        # Coastal states
        coastal_states = ['CA', 'OR', 'WA', 'TX', 'LA', 'MS', 'AL', 'FL', 'GA', 'SC', 
                         'NC', 'VA', 'MD', 'DE', 'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'ME', 'HI', 'AK']
        
        # Mountain states
        mountain_states = ['CO', 'WY', 'MT', 'ID', 'UT', 'NV', 'AZ', 'NM']
        
        # Check for coastal areas
        if state in coastal_states and (second_digit in ['0', '1', '2']):
            return 'coastal'
        
        # Check for mountain areas
        if state in mountain_states:
            return 'mountain'
        
        # Urban classification
        if first_digit in ['0', '1', '2', '3']:
            # Further classify urban areas by density
            if second_digit in ['0', '1']:
                return 'urban_high_density'
            elif second_digit in ['2', '3', '4']:
                return 'urban_medium_density'
            else:
                return 'urban_low_density'
        
        # Suburban classification
        elif first_digit in ['4', '5', '6']:
            # Further classify suburban areas by income level (simplified)
            if second_digit in ['0', '1', '2']:
                return 'suburban_high_income'
            elif second_digit in ['3', '4', '5', '6']:
                return 'suburban_medium_income'
            else:
                return 'suburban_low_income'
        
        # Rural classification
        elif first_digit in ['7', '8', '9']:
            # Further classify rural areas
            if second_digit in ['0', '1', '2', '3']:
                return 'rural_town'
            else:
                return 'rural_remote'
        
        # Default
        return 'default'
    
    def transform(self, X: List[Location]) -> pd.DataFrame:
        """Transform location information into features with enhanced risk factors."""
        # Get basic features from parent class
        features_df = super().transform(X)
        
        # Add enhanced features
        for i, location in enumerate(X):
            # Enhanced territory classification
            enhanced_territory = self._classify_enhanced_territory(location.zip_code, location.state)
            features_df.at[i, 'enhanced_territory'] = enhanced_territory
            
            # Get enhanced territory risk factors
            territory_risks = self.enhanced_territory_risk_map.get(
                enhanced_territory, self.enhanced_territory_risk_map['default']
            )
            
            # Add all risk factors
            for factor, value in territory_risks.items():
                features_df.at[i, f'territory_{factor}'] = value
            
            # State risk factor
            state_risk = self.state_risk_factors.get(location.state, 1.0)
            features_df.at[i, 'state_risk_factor'] = state_risk
            
            # Combined risk score
            combined_risk = (
                territory_risks['crime_rate'] * 0.3 +
                territory_risks['weather_risk'] * 0.2 +
                territory_risks['traffic_density'] * 0.2 +
                territory_risks['parking_risk'] * 0.1 +
                territory_risks['vandalism_risk'] * 0.1 +
                territory_risks['flood_risk'] * 0.1
            ) * state_risk
            
            features_df.at[i, 'location_combined_risk'] = combined_risk
        
        return features_df


class EnhancedFeatureProcessor(FeatureProcessor):
    """Enhanced feature processor with advanced feature engineering."""
    
    def __init__(self, use_pca: bool = False, n_components: int = 20):
        """Initialize the enhanced feature processor.
        
        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components to use
        """
        # Use enhanced feature extractors
        self.policy_extractor = self._create_enhanced_policy_extractor()
        
        # Feature scaling and dimensionality reduction
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(n_components=n_components) if use_pca else None
        
        # Feature names
        self.feature_names = None
        self.original_feature_names = None
    
    def _create_enhanced_policy_extractor(self) -> BaseEstimator:
        """Create an enhanced policy feature extractor."""
        # Create a custom policy extractor with enhanced components
        class EnhancedPolicyExtractor(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.driver_extractor = AdvancedDriverFeatureExtractor()
                self.vehicle_extractor = AdvancedVehicleFeatureExtractor()
                self.history_extractor = super(EnhancedFeatureProcessor, self)._create_policy_extractor().history_extractor
                self.location_extractor = AdvancedLocationFeatureExtractor()
            
            def fit(self, X, y=None):
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
            
            def transform(self, X):
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
                            driver_agg[f'driver_{col}_mean'] = driver_features[col].mean()
                            driver_agg[f'driver_{col}_max'] = driver_features[col].max()
                            driver_agg[f'driver_{col}_min'] = driver_features[col].min()
                        else:
                            # For categorical features, take the mode
                            driver_agg[f'driver_{col}_mode'] = driver_features[col].mode()[0]
                    
                    # Aggregate vehicle features
                    vehicle_agg = {}
