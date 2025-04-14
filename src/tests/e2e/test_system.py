"""
End-to-end tests for the insurance pricing system.
"""

import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

# Try different import approaches to handle both local and CI environments
try:
    from src.data.generator import InsuranceDataGenerator
    from src.ml.app import app
    from src.ml.model import InsurancePricingModel
except ImportError:
    # If the above imports fail, try relative imports
    from data.generator import InsuranceDataGenerator
    from ml.app import app
    from ml.model import InsurancePricingModel


# Custom JSON encoder to handle Decimal values
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class TestSystem(unittest.TestCase):
    """End-to-end tests for the insurance pricing system."""

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

        # Use FastAPI TestClient instead of starting a real server
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

    def test_end_to_end_flow(self):
        """Test the end-to-end flow."""
        # Generate a random policy
        data_generator = InsuranceDataGenerator(seed=43)
        policy = data_generator.generate_policy()

        # Create policy request
        policy_request = {
            "effective_date": policy.effective_date.isoformat(),
            "expiration_date": policy.expiration_date.isoformat(),
            "drivers": [
                {
                    "first_name": driver.first_name,
                    "last_name": driver.last_name,
                    "date_of_birth": driver.date_of_birth.isoformat(),
                    "license_number": driver.license_number,
                    "license_issue_date": driver.license_issue_date.isoformat(),
                    "license_state": driver.license_state,
                    "gender": driver.gender.value if driver.gender else None,
                    "marital_status": (
                        driver.marital_status.value if driver.marital_status else None
                    ),
                    "occupation": driver.occupation,
                }
                for driver in policy.drivers
            ],
            "vehicles": [
                {
                    "make": vehicle.make,
                    "model": vehicle.model,
                    "year": vehicle.year,
                    "vin": vehicle.vin,
                    "value": vehicle.value,
                    "primary_use": vehicle.primary_use.value,
                    "annual_mileage": vehicle.annual_mileage,
                    "anti_theft_device": vehicle.anti_theft_device,
                }
                for vehicle in policy.vehicles
            ],
            "locations": [
                {
                    "address_line1": location.address_line1,
                    "city": location.city,
                    "state": location.state,
                    "zip_code": location.zip_code,
                    "address_line2": location.address_line2,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
                for location in policy.locations
            ],
            "driving_history": [
                {
                    "incident_type": history.incident_type.value,
                    "incident_date": history.incident_date.isoformat(),
                    "severity": history.severity.value,
                    "claim_amount": history.claim_amount,
                    "at_fault": history.at_fault,
                }
                for history in policy.driving_history
            ],
        }

        if policy.pricing_factors:
            policy_request["pricing_factors"] = {
                "credit_score": policy.pricing_factors.credit_score,
                "insurance_score": policy.pricing_factors.insurance_score,
                "territory_code": policy.pricing_factors.territory_code,
                "territory_factor": policy.pricing_factors.territory_factor,
                "driver_factor": policy.pricing_factors.driver_factor,
                "vehicle_factor": policy.pricing_factors.vehicle_factor,
                "history_factor": policy.pricing_factors.history_factor,
            }

        # Send request
        # Convert Decimal values to float for JSON serialization
        policy_request_str = json.dumps(policy_request, cls=DecimalEncoder)
        policy_request = json.loads(policy_request_str)

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

        # Calculate expected premium using the model directly
        expected_premium, expected_factors = self.model.predict_with_factors([policy])
        expected_premium = expected_premium[0] * 1000.0  # Base premium is 1000.0

        # Check that the premium is within a reasonable range
        self.assertAlmostEqual(
            data["final_premium"], expected_premium, delta=expected_premium * 0.2
        )


if __name__ == "__main__":
    unittest.main()
