"""
Integration tests for the insurance pricing API.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient

# Try different import approaches to handle both local and CI environments
try:
    from src.data.generator import InsuranceDataGenerator
    from src.ml.app import app
    from src.ml.model import InsurancePricingModel, PricingService
    from src.ml.models import (
        Driver,
        DrivingHistory,
        Gender,
        IncidentSeverity,
        IncidentType,
        Location,
        MaritalStatus,
        Policy,
        PricingFactors,
        Vehicle,
        VehicleUse,
    )
except ImportError:
    # If the above imports fail, try relative imports
    from data.generator import InsuranceDataGenerator
    from ml.app import app
    from ml.model import InsurancePricingModel, PricingService
    from ml.models import (
        Driver,
        DrivingHistory,
        Gender,
        IncidentSeverity,
        IncidentType,
        Location,
        MaritalStatus,
        Policy,
        PricingFactors,
        Vehicle,
        VehicleUse,
    )


class TestAPI(unittest.TestCase):
    """Integration tests for the insurance pricing API."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a temporary directory for models
        cls.temp_dir = tempfile.TemporaryDirectory()

        # Generate synthetic data
        data_generator = InsuranceDataGenerator(seed=42)
        policies, premiums = data_generator.generate_dataset(100)

        # Create and train model
        cls.model = InsurancePricingModel()
        cls.model.train(policies, premiums)

        # Save model
        cls.model_path = cls.model.save(cls.temp_dir.name)

        # Set environment variable for model path
        os.environ["MODEL_PATH"] = cls.temp_dir.name

        # Create test client
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove temporary directory
        cls.temp_dir.cleanup()

        # Unset environment variable
        del os.environ["MODEL_PATH"]

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("timestamp", data)

    def test_model_info(self):
        """Test model info endpoint."""
        response = self.client.get("/model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_id", data)
        self.assertIn("model_name", data)
        self.assertIn("model_version", data)
        self.assertIn("is_active", data)
        self.assertIn("metrics", data)
        self.assertIn("created_at", data)

    def test_pricing(self):
        """Test pricing endpoint."""
        # Create a sample policy request
        policy_request = {
            "effective_date": date.today().isoformat(),
            "expiration_date": (date.today() + timedelta(days=365)).isoformat(),
            "drivers": [
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "date_of_birth": "1990-01-01",
                    "license_number": "ABC123",
                    "license_issue_date": "2010-01-01",
                    "license_state": "CA",
                    "gender": "male",
                    "marital_status": "single",
                    "occupation": "Engineer",
                }
            ],
            "vehicles": [
                {
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2020,
                    "vin": "1234567890",
                    "value": 25000.0,
                    "primary_use": "commute",
                    "annual_mileage": 12000,
                    "anti_theft_device": True,
                }
            ],
            "locations": [
                {
                    "address_line1": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94105",
                }
            ],
            "driving_history": [
                {
                    "incident_type": "violation",
                    "incident_date": "2022-01-01",
                    "severity": "minor",
                }
            ],
            "pricing_factors": {
                "credit_score": 750.0,
                "insurance_score": 85.0,
            },
        }

        # Send request
        response = self.client.post("/pricing", json=policy_request)

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("policy_id", data)
        self.assertIn("base_premium", data)
        self.assertIn("final_premium", data)
        self.assertIn("factors", data)
        self.assertIn("model_id", data)
        self.assertIn("model_version", data)
        self.assertIn("created_at", data)

        # Check factors
        factors = data["factors"]
        self.assertIn("driver_factor", factors)
        self.assertIn("vehicle_factor", factors)
        self.assertIn("history_factor", factors)
        self.assertIn("location_factor", factors)
        self.assertIn("credit_factor", factors)
        self.assertIn("insurance_factor", factors)

    def test_pricing_missing_fields(self):
        """Test pricing endpoint with missing fields."""
        # Create a sample policy request with missing fields
        policy_request = {
            "effective_date": date.today().isoformat(),
            "expiration_date": (date.today() + timedelta(days=365)).isoformat(),
            "drivers": [],  # Missing drivers
            "vehicles": [
                {
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2020,
                    "vin": "1234567890",
                    "value": 25000.0,
                    "primary_use": "commute",
                    "annual_mileage": 12000,
                    "anti_theft_device": True,
                }
            ],
            "locations": [
                {
                    "address_line1": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94105",
                }
            ],
        }

        # Send request
        response = self.client.post("/pricing", json=policy_request)

        # Check response
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_pricing_invalid_fields(self):
        """Test pricing endpoint with invalid fields."""
        # Create a sample policy request with invalid fields
        policy_request = {
            "effective_date": date.today().isoformat(),
            "expiration_date": (date.today() + timedelta(days=365)).isoformat(),
            "drivers": [
                {
                    "first_name": "John",
                    "last_name": "Doe",
                    "date_of_birth": "1990-01-01",
                    "license_number": "ABC123",
                    "license_issue_date": "2010-01-01",
                    "license_state": "CA",
                    "gender": "invalid",  # Invalid gender
                    "marital_status": "single",
                    "occupation": "Engineer",
                }
            ],
            "vehicles": [
                {
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2020,
                    "vin": "1234567890",
                    "value": 25000.0,
                    "primary_use": "invalid",  # Invalid primary use
                    "annual_mileage": 12000,
                    "anti_theft_device": True,
                }
            ],
            "locations": [
                {
                    "address_line1": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94105",
                }
            ],
        }

        # Send request
        response = self.client.post("/pricing", json=policy_request)

        # Check response
        self.assertEqual(
            response.status_code, 200
        )  # Should still work with invalid fields
        data = response.json()
        self.assertIn("policy_id", data)
        self.assertIn("base_premium", data)
        self.assertIn("final_premium", data)
        self.assertIn("factors", data)
        self.assertIn("model_id", data)
        self.assertIn("model_version", data)
        self.assertIn("created_at", data)


if __name__ == "__main__":
    unittest.main()
