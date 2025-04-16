"""
Data generator for creating synthetic insurance data.
"""

import logging
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from faker import Faker

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

logger = logging.getLogger(__name__)
fake = Faker()


class InsuranceDataGenerator:
    """Generate synthetic insurance data for training and testing."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            fake.seed_instance(seed)

        # Load reference data
        self.vehicle_makes_models = self._load_vehicle_makes_models()
        self.occupation_list = self._load_occupations()
        self.state_zip_codes = self._load_state_zip_codes()

    def _load_vehicle_makes_models(self) -> Dict[str, List[str]]:
        """Load vehicle makes and models."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple dictionary
        return {
            "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"],
            "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
            "Ford": ["F-150", "Escape", "Explorer", "Mustang", "Edge"],
            "Chevrolet": ["Silverado", "Equinox", "Malibu", "Tahoe", "Corvette"],
            "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "Frontier"],
            "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
            "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
            "Audi": ["A4", "A6", "Q5", "Q7", "A3"],
            "Tesla": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck"],
            "Subaru": ["Outback", "Forester", "Crosstrek", "Impreza", "Legacy"],
        }

    def _load_occupations(self) -> List[str]:
        """Load occupations."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple list
        return [
            "Accountant",
            "Actor",
            "Architect",
            "Artist",
            "Attorney",
            "Baker",
            "Barber",
            "Carpenter",
            "Chef",
            "Chemist",
            "Dentist",
            "Designer",
            "Doctor",
            "Driver",
            "Electrician",
            "Engineer",
            "Farmer",
            "Firefighter",
            "Journalist",
            "Lawyer",
            "Mechanic",
            "Nurse",
            "Pharmacist",
            "Pilot",
            "Plumber",
            "Police Officer",
            "Professor",
            "Programmer",
            "Salesperson",
            "Scientist",
            "Student",
            "Teacher",
            "Technician",
            "Veterinarian",
            "Writer",
            "Retired",
            "Unemployed",
        ]

    def _load_state_zip_codes(self) -> Dict[str, List[str]]:
        """Load state zip codes."""
        # In a real implementation, this would load from a database or file
        # For now, we'll use a simple dictionary with some example zip codes
        return {
            "AL": ["35004", "35005", "35006", "35007", "35010"],
            "AK": ["99501", "99502", "99503", "99504", "99505"],
            "AZ": ["85001", "85002", "85003", "85004", "85005"],
            "AR": ["71601", "71602", "71603", "71611", "71612"],
            "CA": ["90001", "90002", "90003", "90004", "90005"],
            "CO": ["80001", "80002", "80003", "80004", "80005"],
            "CT": ["06001", "06002", "06003", "06006", "06010"],
            "DE": ["19701", "19702", "19703", "19706", "19707"],
            "FL": ["32003", "32004", "32006", "32007", "32008"],
            "GA": ["30002", "30004", "30005", "30008", "30009"],
            "HI": ["96701", "96703", "96704", "96705", "96706"],
            "ID": ["83201", "83202", "83203", "83204", "83205"],
            "IL": ["60001", "60002", "60004", "60005", "60006"],
            "IN": ["46001", "46011", "46012", "46013", "46014"],
            "IA": ["50001", "50002", "50003", "50005", "50006"],
            "KS": ["66002", "66006", "66007", "66008", "66010"],
            "KY": ["40003", "40004", "40006", "40007", "40008"],
            "LA": ["70001", "70002", "70003", "70004", "70005"],
            "ME": ["03901", "03902", "03903", "03904", "03905"],
            "MD": ["20601", "20602", "20603", "20606", "20607"],
            "MA": ["01001", "01002", "01003", "01004", "01005"],
            "MI": ["48001", "48002", "48003", "48004", "48005"],
            "MN": ["55001", "55002", "55003", "55005", "55006"],
            "MS": ["38601", "38603", "38606", "38610", "38611"],
            "MO": ["63001", "63005", "63006", "63010", "63011"],
            "MT": ["59001", "59002", "59003", "59006", "59007"],
            "NE": ["68001", "68002", "68003", "68004", "68005"],
            "NV": ["89001", "89002", "89003", "89004", "89005"],
            "NH": ["03031", "03032", "03033", "03034", "03036"],
            "NJ": ["07001", "07002", "07003", "07004", "07005"],
            "NM": ["87001", "87002", "87004", "87005", "87006"],
            "NY": ["10001", "10002", "10003", "10004", "10005"],
            "NC": ["27006", "27007", "27009", "27010", "27011"],
            "ND": ["58001", "58002", "58004", "58005", "58006"],
            "OH": ["43001", "43002", "43003", "43004", "43005"],
            "OK": ["73001", "73002", "73003", "73004", "73005"],
            "OR": ["97001", "97002", "97003", "97004", "97005"],
            "PA": ["15001", "15003", "15005", "15006", "15007"],
            "RI": ["02801", "02802", "02804", "02806", "02807"],
            "SC": ["29001", "29002", "29003", "29006", "29009"],
            "SD": ["57001", "57002", "57003", "57004", "57005"],
            "TN": ["37010", "37011", "37012", "37013", "37014"],
            "TX": ["75001", "75002", "75006", "75007", "75009"],
            "UT": ["84001", "84002", "84003", "84004", "84005"],
            "VT": ["05001", "05009", "05030", "05031", "05032"],
            "VA": ["20101", "20102", "20103", "20104", "20105"],
            "WA": ["98001", "98002", "98003", "98004", "98005"],
            "WV": ["24701", "24712", "24714", "24715", "24716"],
            "WI": ["53001", "53002", "53003", "53004", "53005"],
            "WY": ["82001", "82002", "82003", "82005", "82006"],
        }

    def generate_driver(self) -> Driver:
        """Generate a random driver."""
        # Generate basic driver information
        first_name = fake.first_name()
        last_name = fake.last_name()

        # Generate date of birth (between 18 and 80 years ago)
        min_age = 18
        max_age = 80
        days_in_year = 365.25
        min_days = int(min_age * days_in_year)
        max_days = int(max_age * days_in_year)
        dob = fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)

        # Generate license issue date (at least 16 years after birth, but not before 16 years ago)
        min_license_age = 16
        min_license_date = date(dob.year + min_license_age, dob.month, dob.day)
        max_license_date = date.today()

        if min_license_date > max_license_date:
            # This would happen if the person is younger than the minimum license age
            min_license_date = max_license_date

        license_issue_date = fake.date_between(min_license_date, max_license_date)

        # Generate license state
        license_state = random.choice(list(self.state_zip_codes.keys()))

        # Generate license number
        license_number = fake.bothify(
            text="?######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        # Generate gender
        gender = random.choice(list(Gender))

        # Generate marital status
        marital_status = random.choice(list(MaritalStatus))

        # Generate occupation
        occupation = random.choice(self.occupation_list)

        return Driver(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=dob,
            license_number=license_number,
            license_issue_date=license_issue_date,
            license_state=license_state,
            gender=gender,
            marital_status=marital_status,
            occupation=occupation,
        )

    def generate_vehicle(self) -> Vehicle:
        """Generate a random vehicle."""
        # Generate make and model
        make = random.choice(list(self.vehicle_makes_models.keys()))
        model = random.choice(self.vehicle_makes_models[make])

        # Generate year (between 1 and 15 years old)
        current_year = date.today().year
        year = random.randint(current_year - 15, current_year)

        # Generate VIN
        vin = fake.bothify(text="1??????????????#", letters="ABCDEFGHJKLMNPRSTUVWXYZ")

        # Generate value (between $5,000 and $50,000)
        value = random.uniform(5000, 50000)

        # Generate primary use
        primary_use = random.choice(list(VehicleUse))

        # Generate annual mileage (between 5,000 and 25,000)
        annual_mileage = random.randint(5000, 25000)

        # Generate anti-theft device
        anti_theft_device = (
            random.random() < 0.3
        )  # 30% chance of having an anti-theft device

        return Vehicle(
            make=make,
            model=model,
            year=year,
            vin=vin,
            value=value,
            primary_use=primary_use,
            annual_mileage=annual_mileage,
            anti_theft_device=anti_theft_device,
        )

    def generate_driving_history(self, driver: Driver) -> List[DrivingHistory]:
        """Generate random driving history for a driver."""
        # Determine number of incidents (more likely for younger drivers)
        age = driver.age
        if age < 25:
            max_incidents = 3
            incident_probability = 0.6
        elif age < 35:
            max_incidents = 2
            incident_probability = 0.4
        elif age < 50:
            max_incidents = 2
            incident_probability = 0.3
        else:
            max_incidents = 1
            incident_probability = 0.2

        # Determine if driver has incidents
        if random.random() > incident_probability:
            return []  # No incidents

        # Generate random number of incidents
        num_incidents = random.randint(1, max_incidents)

        # Generate incidents
        incidents = []
        for _ in range(num_incidents):
            # Generate incident type
            incident_type = random.choice(list(IncidentType))

            # Generate incident date (within the last 5 years)
            incident_date = fake.date_between(
                date.today() - timedelta(days=5 * 365), date.today()
            )

            # Generate severity
            severity = random.choice(list(IncidentSeverity))

            # Generate claim amount (if applicable)
            claim_amount = None
            if (
                incident_type == IncidentType.CLAIM
                or incident_type == IncidentType.ACCIDENT
            ):
                if severity == IncidentSeverity.MINOR:
                    claim_amount = random.uniform(500, 2000)
                elif severity == IncidentSeverity.MODERATE:
                    claim_amount = random.uniform(2000, 5000)
                else:  # MAJOR
                    claim_amount = random.uniform(5000, 15000)

            # Generate at-fault status
            at_fault = random.random() < 0.7  # 70% chance of being at fault

            incidents.append(
                DrivingHistory(
                    incident_type=incident_type,
                    incident_date=incident_date,
                    severity=severity,
                    claim_amount=claim_amount,
                    at_fault=at_fault,
                )
            )

        return incidents

    def generate_location(self) -> Location:
        """Generate a random location."""
        # Generate state
        state = random.choice(list(self.state_zip_codes.keys()))

        # Generate zip code
        zip_code = random.choice(self.state_zip_codes[state])

        # Generate address
        address_line1 = fake.street_address()
        city = fake.city()

        # Generate optional address line 2
        address_line2 = None
        if random.random() < 0.3:  # 30% chance of having address line 2
            address_line2 = fake.secondary_address()

        # Generate coordinates
        latitude = fake.latitude()
        longitude = fake.longitude()

        return Location(
            address_line1=address_line1,
            city=city,
            state=state,
            zip_code=zip_code,
            address_line2=address_line2,
            latitude=latitude,
            longitude=longitude,
        )

    def generate_pricing_factors(self) -> PricingFactors:
        """Generate random pricing factors."""
        # Generate credit score (between 300 and 850)
        credit_score = random.uniform(300, 850)

        # Generate insurance score (between 0 and 100)
        insurance_score = random.uniform(0, 100)

        # Generate territory code
        territory_code = fake.bothify(text="??##", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Generate territory factor (between 0.8 and 1.5)
        territory_factor = random.uniform(0.8, 1.5)

        # Generate driver factor (between 0.8 and 1.5)
        driver_factor = random.uniform(0.8, 1.5)

        # Generate vehicle factor (between 0.8 and 1.5)
        vehicle_factor = random.uniform(0.8, 1.5)

        # Generate history factor (between 0.8 and 1.5)
        history_factor = random.uniform(0.8, 1.5)

        return PricingFactors(
            credit_score=credit_score,
            insurance_score=insurance_score,
            territory_code=territory_code,
            territory_factor=territory_factor,
            driver_factor=driver_factor,
            vehicle_factor=vehicle_factor,
            history_factor=history_factor,
        )

    def generate_policy(
        self,
        num_drivers: Optional[int] = None,
        num_vehicles: Optional[int] = None,
        num_locations: Optional[int] = None,
        include_pricing_factors: bool = True,
    ) -> Policy:
        """Generate a random policy."""
        # Generate number of drivers, vehicles, and locations
        if num_drivers is None:
            num_drivers = random.randint(1, 3)

        if num_vehicles is None:
            num_vehicles = random.randint(1, 3)

        if num_locations is None:
            num_locations = random.randint(1, 2)

        # Generate effective date (within the last year)
        effective_date = fake.date_between(
            date.today() - timedelta(days=365), date.today()
        )

        # Generate expiration date (1 year after effective date)
        expiration_date = date(
            effective_date.year + 1, effective_date.month, effective_date.day
        )

        # Generate drivers
        drivers = [self.generate_driver() for _ in range(num_drivers)]

        # Generate vehicles
        vehicles = [self.generate_vehicle() for _ in range(num_vehicles)]

        # Generate locations
        locations = [self.generate_location() for _ in range(num_locations)]

        # Generate driving history for each driver
        driving_history = []
        for driver in drivers:
            driving_history.extend(self.generate_driving_history(driver))

        # Generate pricing factors
        pricing_factors = (
            self.generate_pricing_factors() if include_pricing_factors else None
        )

        return Policy(
            effective_date=effective_date,
            expiration_date=expiration_date,
            drivers=drivers,
            vehicles=vehicles,
            locations=locations,
            driving_history=driving_history,
            pricing_factors=pricing_factors,
        )

    def generate_policies(
        self,
        num_policies: int,
        include_pricing_factors: bool = True,
    ) -> List[Policy]:
        """Generate multiple random policies."""
        return [
            self.generate_policy(include_pricing_factors=include_pricing_factors)
            for _ in range(num_policies)
        ]

    def generate_premium(self, policy: Policy) -> float:
        """Generate a premium for a policy based on its characteristics."""
        # Base premium
        base_premium = 1000.0

        # Driver factors
        driver_age_factor = 1.0
        driver_experience_factor = 1.0
        driver_occupation_factor = 1.0

        for driver in policy.drivers:
            # Age factor
            if driver.age < 25:
                driver_age_factor *= 1.5
            elif driver.age < 35:
                driver_age_factor *= 1.2
            elif driver.age > 65:
                driver_age_factor *= 1.3

            # Experience factor
            if driver.driving_experience < 3:
                driver_experience_factor *= 1.4
            elif driver.driving_experience < 10:
                driver_experience_factor *= 1.1

            # Occupation factor
            if driver.occupation in ["Driver", "Pilot"]:
                driver_occupation_factor *= 1.2
            elif driver.occupation in ["Student", "Unemployed"]:
                driver_occupation_factor *= 1.1

        # Vehicle factors
        vehicle_age_factor = 1.0
        vehicle_value_factor = 1.0
        vehicle_use_factor = 1.0

        for vehicle in policy.vehicles:
            # Age factor
            if vehicle.vehicle_age < 3:
                vehicle_age_factor *= 1.2
            elif vehicle.vehicle_age > 10:
                vehicle_age_factor *= 1.1

            # Value factor
            if vehicle.value > 30000:
                vehicle_value_factor *= 1.3
            elif vehicle.value > 20000:
                vehicle_value_factor *= 1.2
            elif vehicle.value > 10000:
                vehicle_value_factor *= 1.1

            # Use factor
            if vehicle.primary_use == VehicleUse.BUSINESS:
                vehicle_use_factor *= 1.3
            elif vehicle.primary_use == VehicleUse.COMMUTE:
                vehicle_use_factor *= 1.1

        # Driving history factors
        accident_factor = 1.0
        violation_factor = 1.0
        claim_factor = 1.0

        for history in policy.driving_history:
            if history.incident_type == IncidentType.ACCIDENT:
                if history.severity == IncidentSeverity.MAJOR:
                    accident_factor *= 1.5
                elif history.severity == IncidentSeverity.MODERATE:
                    accident_factor *= 1.3
                else:
                    accident_factor *= 1.1

            elif history.incident_type == IncidentType.VIOLATION:
                if history.severity == IncidentSeverity.MAJOR:
                    violation_factor *= 1.4
                elif history.severity == IncidentSeverity.MODERATE:
                    violation_factor *= 1.2
                else:
                    violation_factor *= 1.1

            elif history.incident_type == IncidentType.CLAIM:
                if history.claim_amount and history.claim_amount > 5000:
                    claim_factor *= 1.4
                elif history.claim_amount and history.claim_amount > 2000:
                    claim_factor *= 1.2
                else:
                    claim_factor *= 1.1

        # Location factors
        location_factor = 1.0

        for location in policy.locations:
            # Urban/suburban/rural factor based on zip code
            first_digit = location.zip_code[0]
            if first_digit in ["1", "2", "3"]:  # Urban
                location_factor *= 1.2
            elif first_digit in ["4", "5", "6"]:  # Suburban
                location_factor *= 1.1

        # Calculate final premium
        premium = base_premium
        premium *= driver_age_factor
        premium *= driver_experience_factor
        premium *= driver_occupation_factor
        premium *= vehicle_age_factor
        premium *= vehicle_value_factor
        premium *= vehicle_use_factor
        premium *= accident_factor
        premium *= violation_factor
        premium *= claim_factor
        premium *= location_factor

        # Add some random noise (±10%)
        noise_factor = random.uniform(0.9, 1.1)
        premium *= noise_factor

        return premium

    def generate_dataset(
        self,
        num_policies: int,
        include_pricing_factors: bool = True,
    ) -> Tuple[List[Policy], List[float]]:
        """Generate a dataset of policies and premiums."""
        policies = self.generate_policies(num_policies, include_pricing_factors)
        premiums = [self.generate_premium(policy) for policy in policies]

        return policies, premiums
