"""
FastAPI application for the ML service.
"""

import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    PricingRequest,
    PricingResponse,
    Vehicle,
    VehicleUse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Insurance Pricing API",
    description="API for insurance pricing using ML models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pricing service
MODEL_PATH = os.environ.get("MODEL_PATH", "./models")
DEFAULT_MODEL = os.path.join(MODEL_PATH, "random_forest_20250413224919.joblib")

# Check if model exists, otherwise create a new one
if os.path.exists(DEFAULT_MODEL):
    pricing_service = PricingService(model_path=DEFAULT_MODEL)
    logger.info(f"Loaded model from {DEFAULT_MODEL}")
else:
    pricing_service = PricingService()
    logger.info("Created new pricing service with default model")


# Pydantic models for API
class DriverRequest(BaseModel):
    """Driver information for API requests."""

    first_name: str
    last_name: str
    date_of_birth: str
    license_number: str
    license_issue_date: str
    license_state: str
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None


class VehicleRequest(BaseModel):
    """Vehicle information for API requests."""

    make: str
    model: str
    year: int
    vin: str
    value: float
    primary_use: str
    annual_mileage: int
    anti_theft_device: bool = False


class DrivingHistoryRequest(BaseModel):
    """Driving history information for API requests."""

    incident_type: str
    incident_date: str
    severity: str
    claim_amount: Optional[float] = None
    at_fault: bool = False


class LocationRequest(BaseModel):
    """Location information for API requests."""

    address_line1: str
    city: str
    state: str
    zip_code: str
    address_line2: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PricingFactorsRequest(BaseModel):
    """Pricing factors for API requests."""

    credit_score: Optional[float] = None
    insurance_score: Optional[float] = None
    territory_code: Optional[str] = None
    territory_factor: Optional[float] = None
    driver_factor: Optional[float] = None
    vehicle_factor: Optional[float] = None
    history_factor: Optional[float] = None


class PolicyRequest(BaseModel):
    """Policy information for API requests."""

    effective_date: str
    expiration_date: str
    drivers: List[DriverRequest] = Field(
        ..., min_length=1, description="At least one driver is required"
    )
    vehicles: List[VehicleRequest] = Field(
        ..., min_length=1, description="At least one vehicle is required"
    )
    locations: List[LocationRequest] = Field(
        ..., min_length=1, description="At least one location is required"
    )
    driving_history: List[DrivingHistoryRequest] = []
    pricing_factors: Optional[PricingFactorsRequest] = None


class PricingResponseModel(BaseModel):
    """Response model for pricing API."""

    policy_id: str
    base_premium: float
    final_premium: float
    factors: Dict[str, float]
    model_id: str
    model_version: str
    created_at: str


class ModelInfoResponse(BaseModel):
    """Response model for model info API."""

    model_id: str
    model_name: str
    model_version: str
    is_active: bool
    metrics: Optional[Dict[str, float]] = None
    created_at: str


class HealthResponse(BaseModel):
    """Response model for health check API."""

    status: str
    timestamp: str


# Helper functions
def _parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _convert_driver(driver_req: DriverRequest) -> Driver:
    """Convert driver request to driver object."""
    gender_val = None
    if driver_req.gender:
        try:
            gender_val = Gender(driver_req.gender.lower())
        except ValueError:
            gender_val = None

    marital_status_val = None
    if driver_req.marital_status:
        try:
            marital_status_val = MaritalStatus(driver_req.marital_status.lower())
        except ValueError:
            marital_status_val = None

    return Driver(
        first_name=driver_req.first_name,
        last_name=driver_req.last_name,
        date_of_birth=_parse_date(driver_req.date_of_birth),
        license_number=driver_req.license_number,
        license_issue_date=_parse_date(driver_req.license_issue_date),
        license_state=driver_req.license_state,
        gender=gender_val,
        marital_status=marital_status_val,
        occupation=driver_req.occupation,
    )


def _convert_vehicle(vehicle_req: VehicleRequest) -> Vehicle:
    """Convert vehicle request to vehicle object."""
    try:
        primary_use_val = VehicleUse(vehicle_req.primary_use.lower())
    except ValueError:
        primary_use_val = VehicleUse.PERSONAL

    return Vehicle(
        make=vehicle_req.make,
        model=vehicle_req.model,
        year=vehicle_req.year,
        vin=vehicle_req.vin,
        value=vehicle_req.value,
        primary_use=primary_use_val,
        annual_mileage=vehicle_req.annual_mileage,
        anti_theft_device=vehicle_req.anti_theft_device,
    )


def _convert_driving_history(history_req: DrivingHistoryRequest) -> DrivingHistory:
    """Convert driving history request to driving history object."""
    try:
        incident_type_val = IncidentType(history_req.incident_type.lower())
    except ValueError:
        incident_type_val = IncidentType.ACCIDENT

    try:
        severity_val = IncidentSeverity(history_req.severity.lower())
    except ValueError:
        severity_val = IncidentSeverity.MINOR

    return DrivingHistory(
        incident_type=incident_type_val,
        incident_date=_parse_date(history_req.incident_date),
        severity=severity_val,
        claim_amount=history_req.claim_amount,
        at_fault=history_req.at_fault,
    )


def _convert_location(location_req: LocationRequest) -> Location:
    """Convert location request to location object."""
    return Location(
        address_line1=location_req.address_line1,
        city=location_req.city,
        state=location_req.state,
        zip_code=location_req.zip_code,
        address_line2=location_req.address_line2,
        latitude=location_req.latitude,
        longitude=location_req.longitude,
    )


def _convert_pricing_factors(
    factors_req: Optional[PricingFactorsRequest],
) -> Optional[PricingFactors]:
    """Convert pricing factors request to pricing factors object."""
    if not factors_req:
        return None

    return PricingFactors(
        credit_score=factors_req.credit_score,
        insurance_score=factors_req.insurance_score,
        territory_code=factors_req.territory_code,
        territory_factor=factors_req.territory_factor,
        driver_factor=factors_req.driver_factor,
        vehicle_factor=factors_req.vehicle_factor,
        history_factor=factors_req.history_factor,
    )


def _convert_policy(policy_req: PolicyRequest) -> Policy:
    """Convert policy request to policy object."""
    drivers = [_convert_driver(driver) for driver in policy_req.drivers]
    vehicles = [_convert_vehicle(vehicle) for vehicle in policy_req.vehicles]
    locations = [_convert_location(location) for location in policy_req.locations]
    driving_history = [
        _convert_driving_history(history) for history in policy_req.driving_history
    ]
    pricing_factors = _convert_pricing_factors(policy_req.pricing_factors)

    return Policy(
        effective_date=_parse_date(policy_req.effective_date),
        expiration_date=_parse_date(policy_req.expiration_date),
        drivers=drivers,
        vehicles=vehicles,
        locations=locations,
        driving_history=driving_history,
        pricing_factors=pricing_factors,
    )


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model."""
    if not pricing_service.model.model_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model loaded",
        )

    model_version = pricing_service.model.model_version

    return {
        "model_id": str(model_version.model_id),
        "model_name": model_version.model_name,
        "model_version": model_version.model_version,
        "is_active": model_version.is_active,
        "metrics": model_version.metrics,
        "created_at": model_version.created_at.isoformat(),
    }


@app.post("/pricing", response_model=PricingResponseModel)
async def calculate_pricing(policy_req: PolicyRequest):
    """Calculate pricing for a policy."""
    try:
        # Convert request to policy object
        policy = _convert_policy(policy_req)

        # Calculate premium
        final_premium, factors = pricing_service.calculate_premium(policy)

        # Create response
        response = PricingResponse(
            policy_id=policy.policy_id,
            base_premium=pricing_service.base_premium,
            final_premium=final_premium,
            factors=factors,
            model_id=pricing_service.model.model_version.model_id,
            model_version=pricing_service.model.model_version.model_version,
        )

        return {
            "policy_id": str(response.policy_id),
            "base_premium": response.base_premium,
            "final_premium": response.final_premium,
            "factors": response.factors,
            "model_id": str(response.model_id),
            "model_version": response.model_version,
            "created_at": response.created_at.isoformat(),
        }
    except Exception as e:
        logger.exception("Error calculating pricing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


if __name__ == "__main__":
    # Run the application
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("src.ml.app:app", host="0.0.0.0", port=port, reload=True)
