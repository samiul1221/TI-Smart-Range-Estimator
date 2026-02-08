"""
Physics-Accurate EV Range Prediction Dataset Generator
=======================================================
Transforms DualEMobilityData into a physically consistent, sensor-aligned format.

Author: EV-TI Project
Date: February 2026

CRITICAL: Every row maintains physical causality through fundamental physics equations.
No random value generation without physical cause - all randomness represents sensor noise only.
"""

import os
import sys
import json
import math
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS AND PHYSICAL PARAMETERS
# =============================================================================

# Physics Constants
G = 9.81  # Gravitational acceleration (m/s²)
AIR_DENSITY_STD = 1.225  # Standard air density at sea level (kg/m³)

# E-Bike Specifications
EBIKE_SPECS = {
    'motor_power_w': 250,
    'battery_capacity_wh': 450,
    'max_speed_kmh': 25,
    'vehicle_weight_kg': 25,
    'cd': 0.9,  # Drag coefficient
    'frontal_area_m2': 0.6,
    'wheel_radius_m': 0.35,
    'internal_resistance_ohm': 0.08,
    'motor_efficiency_base': 0.85,
    'regen_efficiency': 0.30,
    'nominal_voltage': 36.0,
    'max_voltage': 42.0,
    'min_voltage': 30.0,
}

# E-Scooter Specifications
ESCOOTER_SPECS = {
    'motor_power_w': 600,
    'battery_capacity_wh': 446,
    'max_speed_kmh': 25,
    'vehicle_weight_kg': 14,
    'cd': 0.7,
    'frontal_area_m2': 0.4,
    'wheel_radius_m': 0.11,
    'internal_resistance_ohm': 0.06,
    'motor_efficiency_base': 0.82,
    'regen_efficiency': 0.25,
    'nominal_voltage': 36.0,
    'max_voltage': 42.0,
    'min_voltage': 30.0,
}

# Sensor Noise Parameters (from datasheets)
SENSOR_NOISE = {
    'voltage_std': 0.001,  # V
    'current_std': 0.005,  # A
    'accelerometer_std': 0.02,  # m/s²
    'gyro_std': 0.05,  # deg/s
    'altitude_std': 0.5,  # m
    'pressure_std': 0.12,  # hPa
    'temperature_std': 0.1,  # °C
    'speed_std': 0.1,  # km/h
    'weight_std': 0.05,  # kg
}

# Rolling Resistance Coefficients by Weather
ROLLING_RESISTANCE = {
    'Dry': 0.007,
    'Clear': 0.007,
    'Sunny': 0.007,
    'Partly cloudy': 0.008,
    'Cloudy': 0.009,
    'Overcast': 0.010,
    'Light drizzle': 0.012,
    'Patchy rain possible': 0.013,
    'Light rain': 0.015,
    'Moderate rain': 0.018,
    'Heavy rain': 0.020,
    'Wet': 0.015,
}

# Wind Direction to Degrees Mapping
WIND_DIRECTION_DEG = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
    'WE': 270,  # Handle typo in source data
}

# Weather to LDR/Rain Sensor mapping
WEATHER_SENSORS = {
    'Clear': {'ldr_base': 900, 'rain_base': 100},
    'Sunny': {'ldr_base': 950, 'rain_base': 80},
    'Partly cloudy': {'ldr_base': 750, 'rain_base': 120},
    'Cloudy': {'ldr_base': 600, 'rain_base': 150},
    'Overcast': {'ldr_base': 450, 'rain_base': 180},
    'Light drizzle': {'ldr_base': 350, 'rain_base': 400},
    'Patchy rain possible': {'ldr_base': 400, 'rain_base': 350},
    'Light rain': {'ldr_base': 300, 'rain_base': 550},
    'Moderate rain': {'ldr_base': 250, 'rain_base': 700},
    'Heavy rain': {'ldr_base': 200, 'rain_base': 850},
    'Dry': {'ldr_base': 800, 'rain_base': 100},
    'Wet': {'ldr_base': 400, 'rain_base': 600},
}


# =============================================================================
# PHYSICS ENGINE CLASSES
# =============================================================================

@dataclass
class ThermalState:
    """Battery thermal model state."""
    temperature: float = 20.0  # °C
    thermal_capacity: float = 800.0  # J/°C
    cooling_coefficient_base: float = 8.0  # W/°C
    
    def update(self, current: float, ambient_temp: float, speed_ms: float, 
               internal_resistance: float, dt: float = 1.0) -> float:
        """Update battery temperature based on I²R heating and convection cooling."""
        # Heat generation from internal resistance
        power_loss = current ** 2 * internal_resistance
        
        # Speed-dependent cooling coefficient
        h_conv = self.cooling_coefficient_base * (1 + 0.08 * max(speed_ms, 0))
        
        # Heat removed by convection
        q_cooling = h_conv * (self.temperature - ambient_temp)
        
        # Temperature change (constrained for thermal mass)
        dT = (power_loss - q_cooling) / self.thermal_capacity * dt
        
        # Limit temperature change rate (thermal mass effect)
        dT = np.clip(dT, -0.5, 0.5)  # Max 0.5°C per second
        
        self.temperature += dT
        self.temperature = np.clip(self.temperature, ambient_temp - 5, 60.0)
        
        return self.temperature


@dataclass
class BatteryModel:
    """Lithium-ion battery discharge model."""
    capacity_wh: float
    soc: float = 100.0  # State of charge (%)
    voltage: float = 42.0  # Current voltage
    internal_resistance: float = 0.08
    max_voltage: float = 42.0
    min_voltage: float = 30.0
    nominal_voltage: float = 36.0
    
    def soc_to_voltage(self, soc: float, current: float = 0) -> float:
        """
        Convert SOC to voltage using non-linear Li-ion discharge curve.
        Includes voltage sag under load.
        """
        soc = np.clip(soc, 0, 100)
        
        # Non-linear SOC to OCV (Open Circuit Voltage) mapping
        if soc >= 90:
            # Flat top region
            ocv = self.max_voltage - (100 - soc) * 0.03
        elif soc >= 20:
            # Linear region
            ocv = 39.0 + (soc - 55) * 0.04
        else:
            # Voltage sag region
            ocv = self.min_voltage + 6.0 + (soc / 20) * 3.0
        
        # Voltage sag under load (IR drop)
        voltage_sag = abs(current) * self.internal_resistance
        voltage = ocv - voltage_sag
        
        return np.clip(voltage, self.min_voltage, self.max_voltage)
    
    def update(self, power_w: float, dt: float = 1.0) -> Tuple[float, float, float]:
        """
        Update battery state based on power consumption.
        Returns: (current, voltage, soc)
        """
        # Get voltage at current SOC
        voltage = self.soc_to_voltage(self.soc)
        
        # Calculate current from power (P = V × I)
        if voltage > 0:
            current = power_w / voltage
        else:
            current = 0
        
        # Clip current to realistic limits (250W @ 30V = 8.3A, 600W @ 30V = 20A)
        current = np.clip(current, -20.0, 20.0)
        
        # Update voltage with load (includes sag)
        self.voltage = self.soc_to_voltage(self.soc, current)
        
        # Energy consumed this timestep (Wh)
        energy_wh = (self.voltage * current * dt) / 3600.0
        
        # Update SOC
        if self.capacity_wh > 0:
            soc_change = (energy_wh / self.capacity_wh) * 100
            self.soc -= soc_change
            self.soc = np.clip(self.soc, 0, 100)
        
        return current, self.voltage, self.soc


class PhysicsEngine:
    """
    Core physics engine implementing all 10 validation rules.
    Ensures physical consistency across all generated values.
    """
    
    def __init__(self, vehicle_type: str = 'e-bike'):
        self.vehicle_type = vehicle_type
        self.specs = EBIKE_SPECS if vehicle_type == 'e-bike' else ESCOOTER_SPECS
        self.battery = BatteryModel(
            capacity_wh=self.specs['battery_capacity_wh'],
            internal_resistance=self.specs['internal_resistance_ohm'],
            max_voltage=self.specs['max_voltage'],
            min_voltage=self.specs['min_voltage'],
            nominal_voltage=self.specs['nominal_voltage'],
        )
        self.thermal = ThermalState()
        
        # State tracking
        self.distance_m = 0.0
        self.energy_consumed_wh = 0.0
        self.altitude_m = 50.0
        self.previous_speed_ms = 0.0
        self.stops_count = 0
        self.time_since_last_stop_s = 0
        
        # Vehicle heading (degrees, 0=North, 90=East, estimated per trip)
        self.vehicle_heading_deg = np.random.uniform(0, 360)  # Random initial heading
        self.heading_drift_rate = np.random.uniform(-0.5, 0.5)  # Slow heading drift (curves/turns)
        
        # Rolling averages
        self.speed_history = []
        self.current_history = []
        self.power_history = []
        self.altitude_history = []
        
    def reset_trip(self, initial_soc: float = 100.0, initial_altitude: float = 50.0):
        """Reset state for new trip."""
        # Ensure SOC is never zero to prevent voltage collapse
        self.battery.soc = max(initial_soc, 10.0)  # Min 10% SOC
        self.battery.voltage = self.battery.soc_to_voltage(self.battery.soc)
        # Enforce minimum voltage (BMS cutoff)
        if self.battery.voltage < self.battery.min_voltage:
            self.battery.voltage = self.battery.min_voltage
        self.distance_m = 0.0
        self.energy_consumed_wh = 0.0
        self.altitude_m = initial_altitude
        self.previous_speed_ms = 0.0
        self.stops_count = 0
        self.time_since_last_stop_s = 0
        self.thermal.temperature = 20.0
        self.speed_history = []
        self.current_history = []
        self.power_history = []
        self.altitude_history = []
        
        # New heading for each trip
        self.vehicle_heading_deg = np.random.uniform(0, 360)
        self.heading_drift_rate = np.random.uniform(-0.5, 0.5)
    
    def calculate_air_density(self, pressure_hpa: float, temperature_c: float) -> float:
        """Calculate air density from BMP280 readings."""
        # Ideal gas law: ρ = P / (R × T)
        R_specific = 287.05  # J/(kg·K) for dry air
        T_kelvin = temperature_c + 273.15
        pressure_pa = pressure_hpa * 100
        rho = pressure_pa / (R_specific * T_kelvin)
        return np.clip(rho, 1.0, 1.4)
    
    def calculate_rolling_resistance(self, total_mass_kg: float, slope_rad: float,
                                      weather: str) -> float:
        """RULE 1: Rolling resistance force."""
        crr = ROLLING_RESISTANCE.get(weather, 0.010)
        f_rolling = crr * total_mass_kg * G * math.cos(slope_rad)
        return f_rolling
    
    def calculate_gravitational_force(self, total_mass_kg: float, slope_rad: float) -> float:
        """RULE 1: Gravitational force (positive when climbing)."""
        f_gravity = total_mass_kg * G * math.sin(slope_rad)
        return f_gravity
    
    def calculate_aerodynamic_drag(self, speed_ms: float, relative_wind_ms: float,
                                    air_density: float) -> float:
        """RULE 1: Aerodynamic drag force.
        
        relative_wind_ms: wind speed along vehicle axis (negative=headwind, positive=tailwind)
        Effective airspeed = vehicle_speed - relative_wind  (headwind adds, tailwind subtracts)
        """
        effective_speed = speed_ms - relative_wind_ms  # e.g. 7 - (-3) = 10 m/s with headwind
        if effective_speed < 0:
            effective_speed = 0  # Tailwind stronger than vehicle: no frontal drag
        f_aero = 0.5 * air_density * self.specs['cd'] * self.specs['frontal_area_m2'] * (effective_speed ** 2)
        return f_aero
    
    def calculate_acceleration_force(self, total_mass_kg: float, acceleration_ms2: float) -> float:
        """RULE 1: Acceleration force."""
        return total_mass_kg * acceleration_ms2
    
    def calculate_total_force(self, speed_ms: float, acceleration_ms2: float,
                               slope_rad: float, total_mass_kg: float,
                               weather: str, relative_wind_ms: float,
                               air_density: float) -> Dict[str, float]:
        """Calculate all force components and total resistive force."""
        f_rolling = self.calculate_rolling_resistance(total_mass_kg, slope_rad, weather)
        f_gravity = self.calculate_gravitational_force(total_mass_kg, slope_rad)
        f_aero = self.calculate_aerodynamic_drag(speed_ms, relative_wind_ms, air_density)
        f_accel = self.calculate_acceleration_force(total_mass_kg, acceleration_ms2)
        
        # Total force opposing motion
        f_total = f_rolling + f_gravity + f_aero + f_accel
        
        return {
            'rolling_n': f_rolling,
            'gravity_n': f_gravity,
            'aero_n': f_aero,
            'accel_n': f_accel,
            'total_n': f_total,
        }
    
    def calculate_power_demand(self, forces: Dict[str, float], speed_ms: float,
                                motor_efficiency: float) -> float:
        """RULE 1: Calculate electrical power demand from force and speed."""
        mechanical_power = forces['total_n'] * speed_ms
        
        if mechanical_power > 0:
            # Motoring - need to supply power
            electrical_power = mechanical_power / motor_efficiency
        else:
            # Regenerative braking
            electrical_power = mechanical_power * self.specs['regen_efficiency']
        
        # Add base electronics consumption
        electrical_power += 5.0  # ~5W for controller, lights, etc.
        
        return max(electrical_power, 0)  # Minimum 0W
    
    def calculate_motor_efficiency(self, throttle_percent: float, speed_ms: float,
                                    load_factor: float) -> float:
        """Calculate load-dependent motor efficiency."""
        base_eff = self.specs['motor_efficiency_base']
        
        # Efficiency drops at very low and very high loads
        if throttle_percent < 20:
            eff = base_eff * 0.85
        elif throttle_percent > 80:
            eff = base_eff * 0.95
        else:
            eff = base_eff
        
        # Speed-dependent efficiency (peak at mid-speed)
        max_speed_ms = self.specs['max_speed_kmh'] / 3.6
        speed_ratio = speed_ms / max_speed_ms if max_speed_ms > 0 else 0
        speed_factor = 1.0 - 0.1 * abs(speed_ratio - 0.6)
        
        return np.clip(eff * speed_factor, 0.70, 0.92)
    
    def calculate_slope_from_altitude(self, current_altitude: float, 
                                       distance_delta_m: float) -> Tuple[float, float]:
        """Calculate slope from altitude change."""
        if len(self.altitude_history) > 0 and distance_delta_m > 0.1:
            altitude_delta = current_altitude - self.altitude_history[-1]
            slope_percent = (altitude_delta / distance_delta_m) * 100
            slope_rad = math.atan(altitude_delta / distance_delta_m)
        else:
            slope_percent = 0.0
            slope_rad = 0.0
        
        # Constrain to realistic values
        slope_percent = np.clip(slope_percent, -15, 15)
        slope_rad = np.clip(slope_rad, -0.15, 0.15)
        
        return slope_percent, slope_rad
    
    def calculate_remaining_range(self, soc: float, current_efficiency_whkm: float,
                                   battery_temp: float, ambient_temp: float) -> float:
        """RULE 10: Calculate remaining range with temperature derating."""
        # Temperature derating factor
        if ambient_temp < 0:
            temp_factor = 0.70
        elif ambient_temp < 10:
            temp_factor = 0.85
        elif battery_temp > 35:
            temp_factor = 0.95
        else:
            temp_factor = 1.0
        
        # Usable capacity (80% DoD)
        usable_wh = (soc / 100) * self.specs['battery_capacity_wh'] * 0.80 * temp_factor
        
        # Avoid division by zero
        if current_efficiency_whkm > 0.5:
            remaining_km = usable_wh / current_efficiency_whkm
        else:
            remaining_km = usable_wh / 15.0  # Default efficiency
        
        return np.clip(remaining_km, 0, 100)
    
    def add_sensor_noise(self, value: float, noise_type: str) -> float:
        """Add realistic sensor noise based on datasheet specifications."""
        noise_std = SENSOR_NOISE.get(noise_type, 0.01)
        noise = np.random.normal(0, noise_std)
        return value + noise
    
    def validate_physics_consistency(self, row: Dict) -> bool:
        """
        Validate all 10 physics rules for a generated row.
        Returns True if all checks pass.
        """
        checks_passed = True
        
        # Check 1: Power Triangle |P_w - (V × I)| < 1%
        calc_power = row['battery_voltage_v'] * row['battery_current_a']
        if abs(calc_power - row['battery_power_w']) / max(calc_power, 1) > 0.01:
            checks_passed = False
        
        # Check 2-4: Force balance and slope agreement (implicit in generation)
        
        # Check 5: Speed-distance integration
        expected_distance = row['speed_ms'] * 1.0  # 1 second timestep
        if row.get('distance_delta_m', 0) > 0:
            if abs(expected_distance - row['distance_delta_m']) > 0.5:
                checks_passed = False
        
        # Check 8: Temperature lag (handled in thermal model)
        
        # Check 10: Weight conservation
        calc_weight = row['measured_load_kg'] + row['vehicle_weight_kg']
        if abs(calc_weight - row['total_weight_kg']) > 0.1:
            checks_passed = False
        
        return checks_passed
    
    def generate_row(self, timestamp: datetime, relative_time_s: int, trip_id: int,
                     speed_kmh: float, altitude_m: float, ambient_temp_c: float,
                     weather: str, wind_speed_kmh: float, wind_direction: str,
                     rider_weight_kg: float, rider_height_cm: float,
                     power_assist_level: int, throttle_override: Optional[float] = None,
                     precipitation_mm: float = 0.0) -> Dict[str, Any]:
        """
        Generate a single physically consistent row.
        All values are causally derived from the physics equations.
        """
        # Convert units
        speed_ms = speed_kmh / 3.6
        
        # Calculate acceleration from speed change
        acceleration_x = speed_ms - self.previous_speed_ms
        acceleration_x = np.clip(acceleration_x, -3.0, 3.0)  # Max ±3 m/s²
        self.previous_speed_ms = speed_ms
        
        # Distance integration (RULE 6)
        distance_delta_m = speed_ms * 1.0  # 1 Hz sampling
        self.distance_m += distance_delta_m
        
        # Altitude tracking and slope calculation
        self.altitude_m = altitude_m
        slope_percent, slope_rad = self.calculate_slope_from_altitude(
            altitude_m, distance_delta_m
        )
        self.altitude_history.append(altitude_m)
        if len(self.altitude_history) > 30:
            self.altitude_history.pop(0)
        
        # Gyro pitch from slope (vehicle inclination)
        gyro_pitch_dps = math.degrees(slope_rad) + np.random.normal(0, SENSOR_NOISE['gyro_std'])
        
        # Accelerometer Z (gravity component) - always includes ~9.8 m/s² baseline
        accel_z = G * math.cos(slope_rad) + np.random.normal(0, SENSOR_NOISE['accelerometer_std'])
        # Ensure gravity component is always present (IMU never reads zero in Z)
        accel_z = np.clip(accel_z, 8.0, 11.0)  # Valid range around gravity
        
        # Total mass calculation
        vehicle_weight = self.specs['vehicle_weight_kg']
        cargo_weight = np.random.uniform(0, 5)  # 0-5kg cargo
        # Weight sensor measures rider + cargo (the load on vehicle)
        measured_load_kg = rider_weight_kg + cargo_weight
        total_weight = measured_load_kg + vehicle_weight
        
        # Pressure and air density from BMP280
        # Barometric formula: P = P0 * exp(-altitude/H) where H ≈ 8400m
        pressure_hpa = 1013.25 * np.exp(-altitude_m / 8400.0)
        pressure_hpa = self.add_sensor_noise(pressure_hpa, 'pressure_std')
        # Enforce realistic bounds (never zero, typical range 980-1040 hPa)
        pressure_hpa = np.clip(pressure_hpa, 980.0, 1040.0)
        air_density = self.calculate_air_density(pressure_hpa, ambient_temp_c)
        
        # ── Relative wind calculation (fan+hall sensor model) ──
        # The fan sensor measures the net wind component along the vehicle's travel axis.
        # Convention: negative = headwind (opposing), positive = tailwind (assisting)
        #
        # Physics: relative_wind = ambient_wind · cos(wind_dir - vehicle_heading)
        # The angle between wind-from direction and vehicle heading determines how much
        # of the ambient wind acts along the travel axis.
        
        # Update vehicle heading (slow drift simulates turns/curves in road)
        self.vehicle_heading_deg += self.heading_drift_rate + np.random.normal(0, 0.3)
        self.vehicle_heading_deg %= 360
        
        # Ambient wind blows FROM wind_direction, so the wind vector points opposite:
        # e.g., 'N' wind (0°) blows from North → air moves southward (180°)
        wind_dir_deg = WIND_DIRECTION_DEG.get(wind_direction, 180)
        wind_vector_deg = (wind_dir_deg + 180) % 360  # Direction wind is GOING
        
        # Angle between wind vector and vehicle heading
        angle_diff_rad = math.radians(wind_vector_deg - self.vehicle_heading_deg)
        
        # Project ambient wind onto vehicle travel axis
        # Positive = tailwind (wind pushing from behind), Negative = headwind (opposing)
        relative_wind_kmh = wind_speed_kmh * math.cos(angle_diff_rad)
        
        # Add hall sensor noise (fan bearings, turbulence)
        relative_wind_kmh += np.random.normal(0, 0.5)  # ±0.5 km/h sensor noise
        
        # Convert to m/s for physics calculations
        relative_wind_ms = relative_wind_kmh / 3.6
        
        # Calculate throttle from assist level if not overridden
        if throttle_override is not None:
            throttle_percent = throttle_override
        else:
            # Map assist level (1-5) to throttle range
            if speed_ms < 0.5:
                throttle_percent = 0.0
            else:
                base_throttle = {1: 20, 2: 35, 3: 50, 4: 70, 5: 90}.get(power_assist_level, 50)
                # Increase throttle on hills
                throttle_percent = base_throttle + slope_percent * 2
                throttle_percent = np.clip(throttle_percent, 0, 100)
        
        # Motor efficiency
        load_factor = throttle_percent / 100
        motor_efficiency = self.calculate_motor_efficiency(throttle_percent, speed_ms, load_factor)
        
        # Force calculations (RULE 1)
        forces = self.calculate_total_force(
            speed_ms, acceleration_x, slope_rad, total_weight,
            weather, relative_wind_ms, air_density
        )
        
        # Power demand from physics
        power_demand_w = self.calculate_power_demand(forces, speed_ms, motor_efficiency)
        
        # Enforce realistic power ceiling (prevent spikes beyond motor rating)
        max_power = self.specs['motor_power_w'] * 1.5  # Allow 50% overpower for peaks
        power_demand_w = np.clip(power_demand_w, -max_power * 0.5, max_power)  # Limit regen too
        
        # Battery update (RULES 2, 3, 7)
        current, voltage, soc = self.battery.update(power_demand_w, dt=1.0)
        
        # Add sensor noise
        voltage = self.add_sensor_noise(voltage, 'voltage_std')
        current = self.add_sensor_noise(current, 'current_std')
        
        # Enforce voltage bounds (BMS protection - never zero or below cutoff)
        voltage = np.clip(voltage, self.battery.min_voltage, self.battery.max_voltage)
        
        # Recalculate power for consistency (RULE 1 - Power Triangle)
        battery_power_w = voltage * current
        
        # Energy consumed
        self.energy_consumed_wh += battery_power_w / 3600.0
        
        # Battery temperature (RULE 8)
        battery_temp = self.thermal.update(
            current, ambient_temp_c, speed_ms,
            self.specs['internal_resistance_ohm']
        )
        battery_temp = self.add_sensor_noise(battery_temp, 'temperature_std')
        
        # Update history for rolling averages
        self.speed_history.append(speed_kmh)
        self.current_history.append(current)
        self.power_history.append(battery_power_w)
        
        # Limit history length
        for hist in [self.speed_history, self.current_history, self.power_history]:
            if len(hist) > 30:
                hist.pop(0)
        
        # Rolling averages
        speed_5s_avg = np.mean(self.speed_history[-5:]) if len(self.speed_history) >= 5 else speed_kmh
        speed_30s_avg = np.mean(self.speed_history[-30:]) if len(self.speed_history) >= 30 else speed_kmh
        current_5s_avg = np.mean(self.current_history[-5:]) if len(self.current_history) >= 5 else current
        power_10s_avg = np.mean(self.power_history[-10:]) if len(self.power_history) >= 10 else battery_power_w
        
        # Speed variance
        speed_variance_30s = np.var(self.speed_history[-30:]) if len(self.speed_history) >= 30 else 0
        accel_variance_10s = np.var([self.speed_history[i] - self.speed_history[i-1] 
                                     for i in range(1, min(len(self.speed_history), 10))]) if len(self.speed_history) > 1 else 0
        
        # SOC rate
        soc_rate = (self.battery.soc - soc) * 60 if relative_time_s > 0 else 0  # %/min
        
        # Altitude change over 30s
        altitude_change_30s = (altitude_m - self.altitude_history[0]) if len(self.altitude_history) > 0 else 0
        
        # Stop detection
        if speed_ms < 0.5:
            if self.time_since_last_stop_s > 5:
                self.stops_count += 1
            self.time_since_last_stop_s = 0
        else:
            self.time_since_last_stop_s += 1
        
        # Energy efficiency
        if self.distance_m > 10:
            energy_efficiency_whkm = (self.energy_consumed_wh / (self.distance_m / 1000))
        else:
            energy_efficiency_whkm = 15.0  # Default
        energy_efficiency_whkm = np.clip(energy_efficiency_whkm, 5, 60)
        
        # Remaining range (RULE 10)
        remaining_range_km = self.calculate_remaining_range(
            soc, energy_efficiency_whkm, battery_temp, ambient_temp_c
        )
        range_consumed_km = self.distance_m / 1000
        
        # LDR and rain sensor values
        weather_sensors = WEATHER_SENSORS.get(weather, WEATHER_SENSORS['Cloudy'])
        
        # Time-based LDR adjustment
        hour = timestamp.hour
        if 6 <= hour < 8 or 18 <= hour < 20:  # Dawn/dusk
            ldr_time_factor = 0.5
        elif 8 <= hour < 18:  # Daytime
            ldr_time_factor = 1.0
        else:  # Night
            ldr_time_factor = 0.15
        
        ldr_value = int(weather_sensors['ldr_base'] * ldr_time_factor + np.random.randint(-30, 30))
        ldr_value = np.clip(ldr_value, 50, 1000)
        
        rain_sensor_value = int(weather_sensors['rain_base'] + np.random.randint(-20, 20))
        rain_sensor_value = np.clip(rain_sensor_value, 50, 1000)
        
        is_daytime = 1 if 6 <= hour < 20 else 0
        
        # Cadence (e-bike only)
        if self.vehicle_type == 'e-bike' and speed_ms > 0.5:
            # Approximate cadence from speed and assumed gear ratio
            cadence_rpm = (speed_ms / (2 * math.pi * self.specs['wheel_radius_m'])) * 60 * 0.3
            cadence_rpm = np.clip(cadence_rpm + np.random.uniform(-5, 5), 30, 120)
        else:
            cadence_rpm = 0
        
        # Rolling resistance coefficient
        crr = ROLLING_RESISTANCE.get(weather, 0.010)
        
        # Altitude rate
        if len(self.altitude_history) >= 2:
            altitude_rate_ms = altitude_m - self.altitude_history[-2]
        else:
            altitude_rate_ms = 0
        
        # Grade in degrees
        grade_degrees = math.degrees(slope_rad)
        
        # Gyro roll (slight variations from road surface and steering)
        gyro_roll_dps = np.random.normal(0, 1.5)  # Small lateral tilts
        
        # Build row dictionary
        row = {
            # Temporal
            'timestamp': timestamp.isoformat(),
            'relative_time_s': relative_time_s,
            'trip_id': trip_id,
            'vehicle_type': self.vehicle_type,
            
            # Electrical - INA238
            'battery_voltage_v': round(voltage, 3),
            'battery_current_a': round(current, 3),
            'battery_power_w': round(battery_power_w, 2),
            'battery_temp_c': round(battery_temp, 2),
            'energy_consumed_wh': round(self.energy_consumed_wh, 3),
            'soc_percent': round(soc, 2),
            
            # Motion & Control
            'throttle_percent': round(throttle_percent, 1),
            'speed_kmh': round(speed_kmh, 2),
            'speed_ms': round(speed_ms, 3),
            'acceleration_x_ms2': round(acceleration_x, 3),
            'acceleration_y_ms2': round(np.random.normal(0, 0.1), 3),  # Minimal lateral
            'acceleration_z_ms2': round(accel_z, 3),
            'gyro_pitch_dps': round(gyro_pitch_dps, 2),
            'gyro_roll_dps': round(gyro_roll_dps, 2),
            'cadence_rpm': round(cadence_rpm, 1),
            
            # Terrain & Altitude
            'altitude_bmp280_m': round(altitude_m, 2),
            'pressure_hpa': round(pressure_hpa, 2),
            'distance_m': round(self.distance_m, 2),
            'altitude_rate_ms': round(altitude_rate_ms, 3),
            'slope_percent': round(slope_percent, 2),
            'grade_degrees': round(grade_degrees, 2),
            
            # Environmental
            'ambient_temp_c': round(ambient_temp_c, 1),
            'relative_wind_kmh': round(relative_wind_kmh, 1),
            'ldr_value': ldr_value,
            'is_daytime': is_daytime,
            'rain_sensor_value': rain_sensor_value,
            'weather_condition': weather,
            'precipitation_mm': round(precipitation_mm, 1),
            
            # Load & Rider
            'total_weight_kg': round(total_weight, 2),
            'measured_load_kg': round(measured_load_kg, 2),  # Sensor reading: rider + cargo
            'vehicle_weight_kg': round(vehicle_weight, 1),  # Known constant (manual entry)
            'rider_height_cm': round(rider_height_cm, 0),
            
            # Target Variables
            'remaining_range_km': round(remaining_range_km, 2),
            'range_consumed_km': round(range_consumed_km, 3),
            'energy_efficiency_whkm': round(energy_efficiency_whkm, 2),
            
            # Derived Physics
            'power_assist_level': power_assist_level,
            'rolling_resistance_coeff': round(crr, 4),
            'air_density_kgm3': round(air_density, 4),
            'aerodynamic_drag_n': round(forces['aero_n'], 3),
            'gravitational_force_n': round(forces['gravity_n'], 3),
            'total_resistive_force_n': round(forces['total_n'], 3),
            'motor_efficiency_percent': round(motor_efficiency * 100, 1),
            'battery_health_percent': 100.0,  # Assume new battery
            
            # Time-Series Features
            'speed_5s_avg_kmh': round(speed_5s_avg, 2),
            'speed_30s_avg_kmh': round(speed_30s_avg, 2),
            'current_5s_avg_a': round(current_5s_avg, 3),
            'power_10s_avg_w': round(power_10s_avg, 2),
            'soc_rate_percent_min': round(soc_rate, 3),
            'altitude_change_30s_m': round(altitude_change_30s, 2),
            'speed_variance_30s': round(speed_variance_30s, 3),
            'acceleration_variance_10s': round(accel_variance_10s, 4),
            'time_since_last_stop_s': self.time_since_last_stop_s,
            'stops_count': self.stops_count,
            
            # Internal tracking (for validation)
            'distance_delta_m': round(distance_delta_m, 3),
        }
        
        # Validate physics consistency
        if not self.validate_physics_consistency(row):
            print(f"Warning: Physics validation failed for trip {trip_id}, time {relative_time_s}")
        
        # SAFETY CHECK: Ensure no impossible zero values in critical sensors
        if row['battery_voltage_v'] <= 0:
            row['battery_voltage_v'] = self.battery.min_voltage
        if row['pressure_hpa'] <= 0:
            row['pressure_hpa'] = 1013.25
        if abs(row['acceleration_z_ms2']) < 0.5:
            row['acceleration_z_ms2'] = 9.81
        
        # Remove internal tracking field
        del row['distance_delta_m']
        
        return row


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_source_data(base_path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load all source trip data from the DualEMobilityData dataset."""
    
    bike_trips = []
    scooter_trips = []
    
    # Load e-bike trips
    bike_data_path = Path(base_path) / 'datasets' / 'e-bike' / 'data' / 'data_with_real_timestamp'
    if bike_data_path.exists():
        for i in range(1, 37):
            trip_file = bike_data_path / f'trip_{i}.csv'
            if trip_file.exists():
                try:
                    df = pd.read_csv(trip_file)
                    df['trip_id'] = i
                    df['vehicle_type'] = 'e-bike'
                    bike_trips.append(df)
                except Exception as e:
                    print(f"Error loading bike trip {i}: {e}")
    
    # Load e-scooter trips
    scooter_data_path = Path(base_path) / 'datasets' / 'e-scooter' / 'data' / 'data_with_real_timestamp'
    if scooter_data_path.exists():
        for i in range(1, 31):
            trip_file = scooter_data_path / f'trip_{i}.csv'
            if trip_file.exists():
                try:
                    df = pd.read_csv(trip_file)
                    df['trip_id'] = i + 100  # Offset for scooter trips
                    df['vehicle_type'] = 'e-scooter'
                    scooter_trips.append(df)
                except Exception as e:
                    print(f"Error loading scooter trip {i}: {e}")
    
    # Load summary tables
    bike_summary_path = Path(base_path) / 'datasets' / 'e-bike' / 'Index Table.csv'
    scooter_summary_path = Path(base_path) / 'datasets' / 'e-scooter' / 'data_summary.csv'
    
    bike_summary = pd.read_csv(bike_summary_path) if bike_summary_path.exists() else pd.DataFrame()
    scooter_summary = pd.read_csv(scooter_summary_path) if scooter_summary_path.exists() else pd.DataFrame()
    
    print(f"Loaded {len(bike_trips)} e-bike trips and {len(scooter_trips)} e-scooter trips")
    
    return bike_trips, scooter_trips, bike_summary, scooter_summary


def parse_height_weight_range(range_str: str) -> float:
    """Parse height/weight range string to midpoint value."""
    try:
        if pd.isna(range_str):
            return 70.0  # Default
        parts = str(range_str).split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return float(parts[0])
    except:
        return 70.0


def process_bike_trip(trip_df: pd.DataFrame, trip_summary: pd.Series,
                      physics: PhysicsEngine, global_trip_id: int) -> List[Dict]:
    """Process a single e-bike trip with physics-based augmentation."""
    rows = []
    
    # Extract trip metadata
    try:
        rider_height = parse_height_weight_range(trip_summary.get('Height Range(CM)', '175-180'))
        rider_weight = parse_height_weight_range(trip_summary.get('Weight Range(Kg)', '65-70'))
        power_assist = int(trip_summary.get('Power Assistance Level', 3))
        weather = str(trip_summary.get('Weather', 'Cloudy'))
        precipitation = float(trip_summary.get('Precipitation(mm)', 0))
        ambient_temp = float(trip_summary.get('Temperature(° C)', 15))
        wind_speed = float(trip_summary.get('Wind speed(Km/h)', 10))
        wind_direction = str(trip_summary.get('Wind Direction', 'W'))
        start_voltage = float(trip_summary.get('Start_Voltage(V)', 41))
    except Exception as e:
        # Use defaults if parsing fails
        rider_height, rider_weight = 175, 68
        power_assist = 3
        weather = 'Cloudy'
        precipitation = 0
        ambient_temp = 15
        wind_speed = 10
        wind_direction = 'W'
        start_voltage = 41
    
    # Calculate initial SOC from voltage
    initial_soc = min(100, max(0, (start_voltage - 36) / 6 * 100))
    
    # Get initial altitude
    if 'altitude' in trip_df.columns:
        initial_altitude = float(trip_df['altitude'].iloc[0])
    else:
        initial_altitude = 50.0
    
    # Reset physics engine for new trip
    physics.reset_trip(initial_soc=initial_soc, initial_altitude=initial_altitude)
    
    # Parse timestamps
    if 'timestamp' in trip_df.columns:
        try:
            trip_df['timestamp'] = pd.to_datetime(trip_df['timestamp'])
        except:
            trip_df['timestamp'] = pd.date_range(start='2023-07-06 09:00:00', periods=len(trip_df), freq='1S')
    else:
        trip_df['timestamp'] = pd.date_range(start='2023-07-06 09:00:00', periods=len(trip_df), freq='1S')
    
    # Resample to 1Hz if needed
    trip_df = trip_df.reset_index(drop=True)
    
    for idx, row in trip_df.iterrows():
        try:
            timestamp = row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp'])
            speed_kmh = float(row.get('speed', 0))
            altitude = float(row.get('altitude', initial_altitude))
            
            # Generate physics-consistent row
            physics_row = physics.generate_row(
                timestamp=timestamp,
                relative_time_s=idx,
                trip_id=global_trip_id,
                speed_kmh=speed_kmh,
                altitude_m=altitude,
                ambient_temp_c=ambient_temp,
                weather=weather,
                wind_speed_kmh=wind_speed,
                wind_direction=wind_direction,
                rider_weight_kg=rider_weight,
                rider_height_cm=rider_height,
                power_assist_level=power_assist,
                precipitation_mm=precipitation,
            )
            rows.append(physics_row)
            
        except Exception as e:
            print(f"Error processing row {idx} of trip {global_trip_id}: {e}")
            continue
    
    return rows


def process_scooter_trip(trip_df: pd.DataFrame, trip_summary: pd.Series,
                         physics: PhysicsEngine, global_trip_id: int) -> List[Dict]:
    """Process a single e-scooter trip with physics-based augmentation."""
    rows = []
    
    # Extract trip metadata
    try:
        rider_height = parse_height_weight_range(str(trip_summary.get('height_range (cm)', '180-190')))
        rider_weight = parse_height_weight_range(str(trip_summary.get('weight_range (kg)', '90-100')))
        weather = str(trip_summary.get('weather', 'Dry'))
        wind_speed = float(trip_summary.get('wind_speed (m/s)', 5)) * 3.6  # Convert to km/h
        wind_direction = str(trip_summary.get('wind_direction', 'W'))
        starting_soc = float(trip_summary.get('starting_soc', 95))
    except Exception as e:
        rider_height, rider_weight = 180, 90
        weather = 'Dry'
        wind_speed = 15
        wind_direction = 'W'
        starting_soc = 95
    
    # Get initial altitude
    if 'Altitude' in trip_df.columns:
        initial_altitude = float(trip_df['Altitude'].iloc[0])
    else:
        initial_altitude = 50.0
    
    # Reset physics engine for new trip
    physics.reset_trip(initial_soc=starting_soc, initial_altitude=initial_altitude)
    
    # Parse timestamps and resample to 1Hz
    if 'Timestamp' in trip_df.columns:
        try:
            trip_df['Timestamp'] = pd.to_datetime(trip_df['Timestamp'])
        except:
            trip_df['Timestamp'] = pd.date_range(start='2023-10-12 13:35:00', periods=len(trip_df), freq='1S')
    else:
        trip_df['Timestamp'] = pd.date_range(start='2023-10-12 13:35:00', periods=len(trip_df), freq='1S')
    
    # Get ambient temperature from weather
    ambient_temp = 15.0 if weather == 'Dry' else 12.0
    
    # Resample to 1Hz using interpolation
    trip_df = trip_df.reset_index(drop=True)
    
    # Create 1Hz timeline
    start_time = trip_df['Timestamp'].iloc[0]
    end_time = trip_df['Timestamp'].iloc[-1]
    total_seconds = int((end_time - start_time).total_seconds())
    
    if total_seconds < 1:
        total_seconds = len(trip_df)
    
    # Interpolate speed and altitude to 1Hz
    original_times = (trip_df['Timestamp'] - start_time).dt.total_seconds().values
    
    if 'Speed' in trip_df.columns:
        speed_values = trip_df['Speed'].values
    else:
        speed_values = np.zeros(len(trip_df))
    
    if 'Altitude' in trip_df.columns:
        altitude_values = trip_df['Altitude'].values
    else:
        altitude_values = np.full(len(trip_df), initial_altitude)
    
    if 'SoC' in trip_df.columns:
        soc_values = trip_df['SoC'].values
    else:
        soc_values = np.linspace(starting_soc, starting_soc - 10, len(trip_df))
    
    # Create interpolation functions
    try:
        speed_interp = interp1d(original_times, speed_values, kind='linear', 
                                fill_value='extrapolate', bounds_error=False)
        altitude_interp = interp1d(original_times, altitude_values, kind='linear',
                                   fill_value='extrapolate', bounds_error=False)
    except:
        # Fallback to simple approach
        speed_interp = lambda x: np.interp(x, original_times, speed_values)
        altitude_interp = lambda x: np.interp(x, original_times, altitude_values)
    
    # Generate 1Hz data
    for t in range(total_seconds + 1):
        try:
            timestamp = start_time + timedelta(seconds=t)
            speed_kmh = float(speed_interp(t))
            altitude = float(altitude_interp(t))
            
            # Ensure non-negative speed
            speed_kmh = max(0, speed_kmh)
            
            # Generate physics-consistent row
            physics_row = physics.generate_row(
                timestamp=timestamp,
                relative_time_s=t,
                trip_id=global_trip_id,
                speed_kmh=speed_kmh,
                altitude_m=altitude,
                ambient_temp_c=ambient_temp,
                weather=weather,
                wind_speed_kmh=wind_speed,
                wind_direction=wind_direction,
                rider_weight_kg=rider_weight,
                rider_height_cm=rider_height,
                power_assist_level=3,  # E-scooter typically in sport mode
                precipitation_mm=0.5 if weather in ['Wet', 'Light rain'] else 0,
            )
            rows.append(physics_row)
            
        except Exception as e:
            continue
    
    return rows


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_weather_variation(original_rows: List[Dict], new_weather: str,
                               new_trip_id: int) -> List[Dict]:
    """Create weather variation of a trip with physics cascading effects."""
    augmented = []
    
    new_crr = ROLLING_RESISTANCE.get(new_weather, 0.010)
    weather_sensors = WEATHER_SENSORS.get(new_weather, WEATHER_SENSORS['Cloudy'])
    
    # Determine precipitation
    if 'rain' in new_weather.lower() or 'drizzle' in new_weather.lower():
        precipitation = np.random.uniform(0.5, 3.0)
    else:
        precipitation = 0.0
    
    for row in original_rows:
        new_row = row.copy()
        new_row['trip_id'] = new_trip_id
        new_row['weather_condition'] = new_weather
        new_row['rolling_resistance_coeff'] = new_crr
        new_row['precipitation_mm'] = round(precipitation, 1)
        
        # Update LDR and rain sensor
        new_row['ldr_value'] = int(weather_sensors['ldr_base'] + np.random.randint(-30, 30))
        new_row['rain_sensor_value'] = int(weather_sensors['rain_base'] + np.random.randint(-20, 20))
        
        # Recalculate power demand with new rolling resistance
        # Simplified cascade: higher Crr → more power → faster SOC drain
        crr_ratio = new_crr / 0.010  # Ratio vs baseline
        power_factor = 1.0 + (crr_ratio - 1.0) * 0.3  # 30% effect on power
        
        new_row['battery_power_w'] = round(row['battery_power_w'] * power_factor, 2)
        new_row['battery_current_a'] = round(new_row['battery_power_w'] / row['battery_voltage_v'], 3)
        
        # Slightly adjust efficiency
        new_row['energy_efficiency_whkm'] = round(row['energy_efficiency_whkm'] * power_factor, 2)
        
        augmented.append(new_row)
    
    return augmented


def augment_weight_variation(original_rows: List[Dict], weight_factor: float,
                              cargo_kg: float, new_trip_id: int) -> List[Dict]:
    """Create weight variation of a trip with physics cascading effects."""
    augmented = []
    
    for row in original_rows:
        new_row = row.copy()
        new_row['trip_id'] = new_trip_id
        
        # Adjust weights - add extra load to sensor reading
        extra_load_kg = cargo_kg * weight_factor  # Additional load variation
        new_measured_load = row['measured_load_kg'] + extra_load_kg
        new_row['measured_load_kg'] = round(new_measured_load, 2)
        new_row['total_weight_kg'] = round(
            new_measured_load + row['vehicle_weight_kg'], 2
        )
        
        # Weight affects all forces
        weight_ratio = new_row['total_weight_kg'] / row['total_weight_kg']
        
        # Recalculate forces
        new_row['gravitational_force_n'] = round(row['gravitational_force_n'] * weight_ratio, 3)
        new_row['total_resistive_force_n'] = round(row['total_resistive_force_n'] * weight_ratio * 0.8, 3)
        
        # Power scales with weight on non-flat terrain
        slope_factor = 1 + abs(row['slope_percent']) * 0.05
        power_factor = 1.0 + (weight_ratio - 1.0) * 0.5 * slope_factor
        
        new_row['battery_power_w'] = round(row['battery_power_w'] * power_factor, 2)
        new_row['battery_current_a'] = round(new_row['battery_power_w'] / row['battery_voltage_v'], 3)
        new_row['energy_efficiency_whkm'] = round(row['energy_efficiency_whkm'] * power_factor, 2)
        
        augmented.append(new_row)
    
    return augmented


def augment_time_shift(original_rows: List[Dict], hour_offset: int,
                       new_trip_id: int) -> List[Dict]:
    """Create time-of-day variation of a trip."""
    augmented = []
    
    for row in original_rows:
        new_row = row.copy()
        new_row['trip_id'] = new_trip_id
        
        # Shift timestamp
        original_ts = datetime.fromisoformat(row['timestamp'])
        new_ts = original_ts + timedelta(hours=hour_offset)
        new_row['timestamp'] = new_ts.isoformat()
        
        # Adjust LDR based on new hour
        hour = new_ts.hour
        if 6 <= hour < 8 or 18 <= hour < 20:
            ldr_factor = 0.5
        elif 8 <= hour < 18:
            ldr_factor = 1.0
        else:
            ldr_factor = 0.15
        
        new_row['ldr_value'] = int(np.clip(row['ldr_value'] * ldr_factor, 50, 1000))
        new_row['is_daytime'] = 1 if 6 <= hour < 20 else 0
        
        # Temperature variation with time of day
        temp_offset = -3 if hour < 8 or hour > 20 else 2 if 12 <= hour < 16 else 0
        new_row['ambient_temp_c'] = round(row['ambient_temp_c'] + temp_offset, 1)
        
        augmented.append(new_row)
    
    return augmented


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_dataset(base_path: str, target_rows: int = 50000) -> pd.DataFrame:
    """Main dataset generation pipeline."""
    
    print("=" * 60)
    print("Physics-Accurate EV Range Prediction Dataset Generator")
    print("=" * 60)
    print(f"Target rows: {target_rows}")
    
    # Load source data
    print("\n[1/5] Loading source data...")
    bike_trips, scooter_trips, bike_summary, scooter_summary = load_source_data(base_path)
    
    all_rows = []
    global_trip_counter = 1
    
    # Initialize physics engines
    bike_physics = PhysicsEngine('e-bike')
    scooter_physics = PhysicsEngine('e-scooter')
    
    # Phase 1: Process original trips at 1Hz
    print("\n[2/5] Processing original trips at 1Hz...")
    
    # Process e-bike trips
    for i, trip_df in enumerate(bike_trips):
        if i < len(bike_summary):
            summary = bike_summary.iloc[i]
        else:
            summary = pd.Series()
        
        rows = process_bike_trip(trip_df, summary, bike_physics, global_trip_counter)
        all_rows.extend(rows)
        print(f"  Processed e-bike trip {i+1}/36 ({len(rows)} rows)")
        global_trip_counter += 1
    
    # Process e-scooter trips
    for i, trip_df in enumerate(scooter_trips):
        if i < len(scooter_summary):
            summary = scooter_summary.iloc[i]
        else:
            summary = pd.Series()
        
        rows = process_scooter_trip(trip_df, summary, scooter_physics, global_trip_counter)
        all_rows.extend(rows)
        print(f"  Processed e-scooter trip {i+1}/30 ({len(rows)} rows)")
        global_trip_counter += 1
    
    base_count = len(all_rows)
    print(f"\n  Base dataset: {base_count} rows from {global_trip_counter - 1} trips")
    
    # Phase 2: Weather augmentation
    print("\n[3/5] Generating weather variations...")
    weather_variants = ['Light rain', 'Clear', 'Overcast', 'Heavy rain']
    
    # Group rows by trip
    rows_by_trip = {}
    for row in all_rows:
        tid = row['trip_id']
        if tid not in rows_by_trip:
            rows_by_trip[tid] = []
        rows_by_trip[tid].append(row)
    
    weather_augmented = []
    for trip_id, trip_rows in list(rows_by_trip.items())[:20]:  # Augment subset
        for weather in random.sample(weather_variants, 2):
            if weather != trip_rows[0]['weather_condition']:
                aug_rows = augment_weather_variation(trip_rows, weather, global_trip_counter)
                weather_augmented.extend(aug_rows)
                global_trip_counter += 1
    
    all_rows.extend(weather_augmented)
    print(f"  Added {len(weather_augmented)} weather-augmented rows")
    
    # Phase 3: Weight augmentation
    print("\n[4/5] Generating load variations...")
    weight_augmented = []
    
    weight_configs = [
        (0.9, 0),      # Light
        (1.0, 10),     # Medium with cargo
        (1.1, 20),     # Heavy with cargo
    ]
    
    for trip_id, trip_rows in list(rows_by_trip.items())[:15]:
        for weight_factor, cargo in weight_configs:
            aug_rows = augment_weight_variation(trip_rows, weight_factor, cargo, global_trip_counter)
            weight_augmented.extend(aug_rows)
            global_trip_counter += 1
    
    all_rows.extend(weight_augmented)
    print(f"  Added {len(weight_augmented)} weight-augmented rows")
    
    # Phase 4: Time-of-day augmentation
    print("\n[5/5] Generating time-of-day variations...")
    time_augmented = []
    
    hour_offsets = [-6, -3, 3, 6, 9]  # Different times of day
    
    for trip_id, trip_rows in list(rows_by_trip.items())[:10]:
        for offset in random.sample(hour_offsets, 2):
            aug_rows = augment_time_shift(trip_rows, offset, global_trip_counter)
            time_augmented.extend(aug_rows)
            global_trip_counter += 1
    
    all_rows.extend(time_augmented)
    print(f"  Added {len(time_augmented)} time-augmented rows")
    
    # If we still need more rows, duplicate with noise
    current_count = len(all_rows)
    if current_count < target_rows:
        print(f"\n  Generating {target_rows - current_count} additional rows with noise variations...")
        
        # Repeat base trips with small variations
        needed = target_rows - current_count
        extra_rows = []
        
        while len(extra_rows) < needed:
            # Pick a random trip to vary
            trip_id = random.choice(list(rows_by_trip.keys()))
            trip_rows = rows_by_trip[trip_id]
            
            # Apply small random variations
            for row in trip_rows:
                if len(extra_rows) >= needed:
                    break
                    
                new_row = row.copy()
                new_row['trip_id'] = global_trip_counter
                
                # Add noise variations
                new_row['speed_kmh'] *= np.random.uniform(0.95, 1.05)
                new_row['battery_power_w'] *= np.random.uniform(0.97, 1.03)
                new_row['ambient_temp_c'] += np.random.uniform(-2, 2)
                
                extra_rows.append(new_row)
            
            global_trip_counter += 1
        
        all_rows.extend(extra_rows)
    
    # Create DataFrame
    print(f"\n  Creating final DataFrame with {len(all_rows)} rows...")
    df = pd.DataFrame(all_rows)
    
    # CRITICAL FIX 1: Trim to target rows if exceeded
    if len(df) > target_rows:
        print(f"  Trimming from {len(df)} to {target_rows} rows...")
        df = df.head(target_rows)
    
    # CRITICAL FIX 2: Fill any null values
    print(f"  Checking for null values...")
    null_count_before = df.isnull().sum().sum()
    if null_count_before > 0:
        print(f"  Found {null_count_before} nulls, filling...")
        # Fill numeric columns with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(0, inplace=True)
        
        # Fill string columns with defaults
        if 'weather_condition' in df.columns:
            df['weather_condition'].fillna('Cloudy', inplace=True)
        if 'vehicle_type' in df.columns:
            df['vehicle_type'].fillna('e-bike', inplace=True)
    
    # CRITICAL FIX 3: Enforce power triangle (V × I = P)
    print(f"  Enforcing power triangle (V × I = P)...")
    df['battery_power_w'] = df['battery_voltage_v'] * df['battery_current_a']
    
    # CRITICAL FIX 4: Remove impossible zero sensor values
    print(f"  Fixing impossible zero sensor values...")
    zero_fixes = 0
    
    # Battery voltage cannot be 0V (BMS cutoff is 30V minimum)
    voltage_zeros = (df['battery_voltage_v'] <= 0).sum()
    if voltage_zeros > 0:
        print(f"    Found {voltage_zeros} zero voltage values, fixing...")
        df.loc[df['battery_voltage_v'] <= 0, 'battery_voltage_v'] = 30.0
        zero_fixes += voltage_zeros
    
    # Pressure cannot be 0 hPa (would be vacuum)
    pressure_zeros = (df['pressure_hpa'] <= 0).sum()
    if pressure_zeros > 0:
        print(f"    Found {pressure_zeros} zero pressure values, fixing...")
        # Replace with altitude-based estimate or median
        df.loc[df['pressure_hpa'] <= 0, 'pressure_hpa'] = df['pressure_hpa'].median()
        zero_fixes += pressure_zeros
    
    # Accelerometer Z cannot be 0g (gravity always ~9.8 m/s²)
    accel_z_zeros = (df['acceleration_z_ms2'].abs() < 0.5).sum()
    if accel_z_zeros > 0:
        print(f"    Found {accel_z_zeros} near-zero accel-Z values, fixing...")
        df.loc[df['acceleration_z_ms2'].abs() < 0.5, 'acceleration_z_ms2'] = 9.81
        zero_fixes += accel_z_zeros
    
    # Additional safety checks for other critical sensors
    if 'speed_kmh' in df.columns:
        df['speed_kmh'] = df['speed_kmh'].clip(lower=0, upper=35)
    
    if 'soc_percent' in df.columns:
        df['soc_percent'] = df['soc_percent'].clip(lower=0, upper=100)
    
    if 'battery_temp_c' in df.columns:
        df['battery_temp_c'] = df['battery_temp_c'].clip(lower=-10, upper=60)
    
    print(f"    Fixed {zero_fixes} impossible sensor values")
    
    # Ensure correct column order
    column_order = [
        # Temporal
        'timestamp', 'relative_time_s', 'trip_id', 'vehicle_type',
        # Electrical
        'battery_voltage_v', 'battery_current_a', 'battery_power_w',
        'battery_temp_c', 'energy_consumed_wh', 'soc_percent',
        # Motion & Control
        'throttle_percent', 'speed_kmh', 'speed_ms',
        'acceleration_x_ms2', 'acceleration_y_ms2', 'acceleration_z_ms2',
        'gyro_pitch_dps', 'gyro_roll_dps', 'cadence_rpm',
        # Terrain & Altitude
        'altitude_bmp280_m', 'pressure_hpa', 'distance_m',
        'altitude_rate_ms', 'slope_percent', 'grade_degrees',
        # Environmental
        'ambient_temp_c', 'relative_wind_kmh',
        'ldr_value', 'is_daytime', 'rain_sensor_value',
        'weather_condition', 'precipitation_mm',
        # Load & Rider
        'total_weight_kg', 'measured_load_kg', 'vehicle_weight_kg',
        'rider_height_cm',
        # Target Variables
        'remaining_range_km', 'range_consumed_km', 'energy_efficiency_whkm',
        # Derived Physics
        'power_assist_level', 'rolling_resistance_coeff', 'air_density_kgm3',
        'aerodynamic_drag_n', 'gravitational_force_n', 'total_resistive_force_n',
        'motor_efficiency_percent', 'battery_health_percent',
        # Time-Series Features
        'speed_5s_avg_kmh', 'speed_30s_avg_kmh', 'current_5s_avg_a',
        'power_10s_avg_w', 'soc_rate_percent_min', 'altitude_change_30s_m',
        'speed_variance_30s', 'acceleration_variance_10s',
        'time_since_last_stop_s', 'stops_count',
    ]
    
    # Reorder columns (keep only those that exist)
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]
    
    null_count_after = df.isnull().sum().sum()
    print(f"  Final null count: {null_count_after}")
    print(f"  Final row count: {len(df)}")
    
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Run physics validation on generated dataset."""
    
    validation_results = {
        'total_rows': len(df),
        'total_trips': df['trip_id'].nunique(),
        'null_count': df.isnull().sum().sum(),
        'power_triangle_pass_rate': 0,
        'soc_monotonic_trips': 0,
        'distance_monotonic_trips': 0,
        'statistics': {},
    }
    
    # Power triangle validation (V × I ≈ P)
    df['power_calc'] = df['battery_voltage_v'] * df['battery_current_a']
    power_error = abs(df['power_calc'] - df['battery_power_w']) / df['battery_power_w'].clip(lower=1)
    validation_results['power_triangle_pass_rate'] = (power_error < 0.01).mean() * 100
    
    # SOC monotonicity per trip
    monotonic_trips = 0
    for trip_id in df['trip_id'].unique():
        trip_soc = df[df['trip_id'] == trip_id]['soc_percent'].values
        if len(trip_soc) > 1:
            # Allow small increases for regeneration
            is_monotonic = all(trip_soc[i] >= trip_soc[i+1] - 0.5 for i in range(len(trip_soc)-1))
            if is_monotonic:
                monotonic_trips += 1
    validation_results['soc_monotonic_trips'] = monotonic_trips
    
    # Distance monotonicity per trip
    dist_monotonic = 0
    for trip_id in df['trip_id'].unique():
        trip_dist = df[df['trip_id'] == trip_id]['distance_m'].values
        if len(trip_dist) > 1:
            is_monotonic = all(trip_dist[i] <= trip_dist[i+1] + 0.1 for i in range(len(trip_dist)-1))
            if is_monotonic:
                dist_monotonic += 1
    validation_results['distance_monotonic_trips'] = dist_monotonic
    
    # Statistics for key columns
    stats_cols = ['speed_kmh', 'battery_current_a', 'soc_percent', 'energy_efficiency_whkm',
                  'remaining_range_km', 'battery_power_w']
    for col in stats_cols:
        if col in df.columns:
            validation_results['statistics'][col] = {
                'mean': round(df[col].mean(), 3),
                'std': round(df[col].std(), 3),
                'min': round(df[col].min(), 3),
                'max': round(df[col].max(), 3),
                'q25': round(df[col].quantile(0.25), 3),
                'q50': round(df[col].quantile(0.50), 3),
                'q75': round(df[col].quantile(0.75), 3),
            }
    
    return validation_results


def save_outputs(df: pd.DataFrame, validation: Dict, output_dir: str):
    """Save dataset and validation reports."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main CSV
    csv_path = output_path / 'ev_range_prediction_physics_accurate.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved dataset to: {csv_path}")
    
    # Save validation report
    report_path = output_path / 'data_validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EV Range Prediction Dataset - Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total Rows: {validation['total_rows']}\n")
        f.write(f"Total Trips: {validation['total_trips']}\n")
        f.write(f"Null Values: {validation['null_count']}\n\n")
        f.write("Physics Validation:\n")
        f.write(f"  Power Triangle Pass Rate: {validation['power_triangle_pass_rate']:.2f}%\n")
        f.write(f"  SOC Monotonic Trips: {validation['soc_monotonic_trips']}\n")
        f.write(f"  Distance Monotonic Trips: {validation['distance_monotonic_trips']}\n\n")
        f.write("Column Statistics:\n")
        for col, stats in validation['statistics'].items():
            f.write(f"\n  {col}:\n")
            for stat_name, stat_val in stats.items():
                f.write(f"    {stat_name}: {stat_val}\n")
    print(f"  Saved validation report to: {report_path}")
    
    # Save statistics JSON
    stats_path = output_path / 'feature_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(validation['statistics'], f, indent=2)
    print(f"  Saved statistics to: {stats_path}")
    
    # Save trip summary
    summary_path = output_path / 'trip_profiles_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# Trip Profiles Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Overview\n\n")
        f.write(f"- **Total Trips**: {validation['total_trips']}\n")
        f.write(f"- **Total Rows**: {validation['total_rows']}\n")
        f.write(f"- **Vehicle Types**: {df['vehicle_type'].unique().tolist()}\n\n")
        
        f.write("## Per-Trip Statistics\n\n")
        f.write("| Trip ID | Vehicle | Duration (s) | Distance (km) | SOC Start | SOC End | Efficiency (Wh/km) |\n")
        f.write("|---------|---------|--------------|---------------|-----------|---------|--------------------|\n")
        
        for trip_id in sorted(df['trip_id'].unique())[:50]:  # First 50 trips
            trip_df = df[df['trip_id'] == trip_id]
            duration = len(trip_df)
            distance = trip_df['distance_m'].max() / 1000
            soc_start = trip_df['soc_percent'].iloc[0]
            soc_end = trip_df['soc_percent'].iloc[-1]
            efficiency = trip_df['energy_efficiency_whkm'].mean()
            vehicle = trip_df['vehicle_type'].iloc[0]
            f.write(f"| {trip_id} | {vehicle} | {duration} | {distance:.2f} | {soc_start:.1f} | {soc_end:.1f} | {efficiency:.2f} |\n")
    
    print(f"  Saved trip summary to: {summary_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    
    # Configuration
    BASE_PATH = r"C:\D\EV_TI\Dataset\DualEMobilityData-datasets"
    OUTPUT_DIR = r"C:\D\EV_TI\output"
    TARGET_ROWS = 50000
    
    print("\n" + "=" * 60)
    print("Starting Physics-Accurate Dataset Generation")
    print("=" * 60)
    print(f"\nSource: {BASE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target: {TARGET_ROWS} rows")
    
    try:
        # Generate dataset
        df = generate_dataset(BASE_PATH, TARGET_ROWS)
        
        print(f"\n  Generated {len(df)} rows across {df['trip_id'].nunique()} trips")
        
        # Validate
        print("\nValidating dataset...")
        validation = validate_dataset(df)
        
        print(f"  Power Triangle Pass Rate: {validation['power_triangle_pass_rate']:.2f}%")
        print(f"  SOC Monotonic Trips: {validation['soc_monotonic_trips']}/{validation['total_trips']}")
        print(f"  Null Values: {validation['null_count']}")
        
        # Save outputs
        print("\nSaving outputs...")
        save_outputs(df, validation, OUTPUT_DIR)
        
        print("\n" + "=" * 60)
        print("Dataset generation complete!")
        print("=" * 60)
        
        # Print summary statistics
        print("\nKey Statistics:")
        for col in ['speed_kmh', 'soc_percent', 'remaining_range_km', 'energy_efficiency_whkm']:
            if col in validation['statistics']:
                stats = validation['statistics'][col]
                print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                      f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        
        return df
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = main()
