"""
Atmospheric refraction model with temperature and pressure correction.
"""

STANDARD_PRESSURE_HPA = 1013.25
STANDARD_LAPSE_RATE = 0.0065  # °C per meter


def estimate_pressure_from_elevation(
    elevation_m: float,
    sea_level_pressure: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = 15.0,
) -> float:
    """
    Estimate atmospheric pressure at a given elevation using the
    barometric formula (standard lapse rate).

    Parameters:
        elevation_m: Elevation above sea level in meters
        sea_level_pressure: Sea-level pressure in hPa
        temperature_c: Temperature at sea level in °C

    Returns:
        Estimated pressure in hPa
    """
    t_k = temperature_c + 273.15
    exponent = 5.2558  # g / (R * L) for dry air
    return sea_level_pressure * (
        1 - STANDARD_LAPSE_RATE * elevation_m / t_k
    ) ** exponent


def refraction_at_horizon(
    temperature_c: float = 10.0,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
) -> float:
    """
    Compute atmospheric refraction at the horizon (altitude = 0 deg)
    using a temperature- and pressure-corrected model.

    Based on Saemundsson's formula with Meeus correction factors.

    Parameters:
        temperature_c: Air temperature in °C
        pressure_hpa: Atmospheric pressure in hPa (mbar)

    Returns:
        Refraction angle in degrees
    """
    r0_arcmin = 34.5  # arcminutes at standard conditions (10°C, 1013.25 hPa)

    pressure_factor = pressure_hpa / STANDARD_PRESSURE_HPA
    temperature_factor = 283.0 / (273.0 + temperature_c)

    r_arcmin = r0_arcmin * pressure_factor * temperature_factor
    return r_arcmin / 60.0  # arcminutes -> degrees
