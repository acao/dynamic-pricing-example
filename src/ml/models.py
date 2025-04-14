"""
Data models for the ML service.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class MaritalStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


class VehicleUse(str, Enum):
    PERSONAL = "personal"
    COMMUTE = "commute"
    BUSINESS = "business"


class IncidentType(str, Enum):
    ACCIDENT = "accident"
    VIOLATION = "violation"
    CLAIM = "claim"


class IncidentSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class PolicyStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Driver:
    """Driver information for insurance pricing."""

    first_name: str
    last_name: str
    date_of_birth: date
    license_number: str
    license_issue_date: date
    license_state: str
    gender: Optional[Gender] = None
    marital_status: Optional[MaritalStatus] = None
    occupation: Optional[str] = None
    driver_id: UUID = field(default_factory=uuid4)

    @property
    def age(self) -> int:
        """Calculate the driver's age."""
        today = date.today()
        return (
            today.year
            - self.date_of_birth.year
            - (
                (today.month, today.day)
                < (self.date_of_birth.month, self.date_of_birth.day)
            )
        )

    @property
    def driving_experience(self) -> int:
        """Calculate the driver's driving experience in years."""
        today = date.today()
        return (
            today.year
            - self.license_issue_date.year
            - (
                (today.month, today.day)
                < (self.license_issue_date.month, self.license_issue_date.day)
            )
        )


@dataclass
class Vehicle:
    """Vehicle information for insurance pricing."""

    make: str
    model: str
    year: int
    vin: str
    value: float
    primary_use: VehicleUse
    annual_mileage: int
    anti_theft_device: bool = False
    vehicle_id: UUID = field(default_factory=uuid4)

    @property
    def vehicle_age(self) -> int:
        """Calculate the vehicle's age."""
        return datetime.now().year - self.year


@dataclass
class DrivingHistory:
    """Driving history information for insurance pricing."""

    incident_type: IncidentType
    incident_date: date
    severity: IncidentSeverity
    claim_amount: Optional[float] = None
    at_fault: bool = False
    history_id: UUID = field(default_factory=uuid4)


@dataclass
class Location:
    """Location information for insurance pricing."""

    address_line1: str
    city: str
    state: str
    zip_code: str
    address_line2: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_id: UUID = field(default_factory=uuid4)


@dataclass
class PricingFactors:
    """Pricing factors for insurance pricing."""

    credit_score: Optional[float] = None
    insurance_score: Optional[float] = None
    territory_code: Optional[str] = None
    territory_factor: Optional[float] = None
    driver_factor: Optional[float] = None
    vehicle_factor: Optional[float] = None
    history_factor: Optional[float] = None
    factor_id: UUID = field(default_factory=uuid4)


@dataclass
class Policy:
    """Policy information for insurance pricing."""

    effective_date: date
    expiration_date: date
    drivers: List[Driver]
    vehicles: List[Vehicle]
    locations: List[Location]
    driving_history: List[DrivingHistory]
    pricing_factors: Optional[PricingFactors] = None
    base_premium: Optional[float] = None
    final_premium: Optional[float] = None
    status: PolicyStatus = PolicyStatus.PENDING
    policy_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelVersion:
    """Model version information."""

    model_name: str
    model_version: str
    model_path: str
    is_active: bool = False
    metrics: Optional[Dict[str, float]] = None
    model_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PricingRequest:
    """Request for insurance pricing."""

    drivers: List[Dict]
    vehicles: List[Dict]
    locations: List[Dict]
    driving_history: List[Dict]
    effective_date: str
    expiration_date: str
    additional_factors: Optional[Dict] = None


@dataclass
class PricingResponse:
    """Response for insurance pricing."""

    policy_id: UUID
    base_premium: float
    final_premium: float
    factors: Dict[str, float]
    model_id: UUID
    model_version: str
    created_at: datetime = field(default_factory=datetime.now)
