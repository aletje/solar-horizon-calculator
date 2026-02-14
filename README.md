# Solar Horizon Calculator

A Python tool for computing sunrise and sunset times accounting for local terrain (DTM-based horizon profiles).

## Features

- Computes local horizon profile from DTM raster data
- Calculates sunrise/sunset times using Skyfield ephemeris
- Accounts for observer elevation, atmospheric refraction, and Earth curvature
- Temperature-dependent refraction from local sensor readings
- Handles polar day / polar night conditions
- Exports results to CSV and PDF

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python solar_calculator.py
```

Edit the `__main__` block in `solar_calculator.py` or create a `SolarCalcConfig` to customize parameters.

### Temperature Data

Place your daily sensor readings in a CSV file (semicolon-delimited):

```csv
date;temperature_c
2025-01-01;-12.3
2025-01-02;-10.8
2025-01-03;-15.1
```

Configure the path in `SolarCalcConfig.temperature_file`.
