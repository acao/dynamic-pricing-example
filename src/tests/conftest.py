"""
Pytest configuration and fixtures for tests.
"""
import os
import tempfile
from datetime import date, timedelta
from typing import Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from src.data.generator import InsuranceDataGenerator
from src.ml.app import app
from src.ml.model import InsurancePricingModel, PricingService
from src.ml.models import (
    Driver, DrivingHistory, Gender, IncidentSeverity, IncidentType,
    Location, MaritalStatus, Policy, PricingFactors, Vehicle, VehicleUse
)


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def sample_policy():
    """Create a sample policy for testing."""
    return Policy(
        effective_date=date.today(),
        expiration_date=date.today() + timedelta(days=365),
        drivers=[
            Driver(
                first_name="John",
                last_name="Doe",
                date_of_birth=date(1990, 1, 1),
                license_number="ABC123",
                license_issue_date=date(2010, 1, 1),
                license_state="CA",
                gender=Gender.MALE,
                marital_status=MaritalStatus.SINGLE,
                occupation="Engineer",
            )
        ],
        vehicles=[
            Vehicle(
                make="Toyota",
                model="Camry",
                year=2020,
                vin="1234567890",
                value=25000.0,
                primary_use=VehicleUse.COMMUTE,
                annual_mileage=12000,
                anti_theft_device=True,
            )
        ],
        locations=[
            Location(
                address_line1="123 Main St",
                city="San Francisco",
                state="CA",
                zip_code="94105",
            )
        ],
        driving_history=[
            DrivingHistory(
                incident_type=IncidentType.VIOLATION,
                incident_date=date(2022, 1, 1),
                severity=IncidentSeverity.MINOR,
            )
        ],
        pricing_factors=PricingFactors(
            credit_score=750.0,
            insurance_score=85.0,
        ),
    )


@pytest.fixture(scope="session")
def synthetic_data() -> Tuple[List[Policy], List[float]]:
    """Generate synthetic data for testing."""
    data_generator = InsuranceDataGenerator(seed=42)
    return data_generator.generate_dataset(100)


@pytest.fixture(scope="session")
def trained_model(synthetic_data):
    """Create and train a model for testing."""
    policies, premiums = synthetic_data
    model = InsurancePricingModel()
    model.train(policies, premiums)
    return model


@pytest.fixture(scope="session")
def model_path(trained_model, temp_dir):
    """Save the trained model and return the path."""
    return trained_model.save(temp_dir)


@pytest.fixture(scope="session")
def pricing_service(trained_model):
    """Create a pricing service for testing."""
    return PricingService(model=trained_model)


@pytest.fixture(scope="function")
def test_client(model_path):
    """Create a test client for the FastAPI app."""
    # Set environment variable for model path
    old_model_path = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = os.path.dirname(model_path)
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Restore environment variable
    if old_model_path is not None:
        os.environ["MODEL_PATH"] = old_model_path
    else:
        del os.environ["MODEL_PATH"]
