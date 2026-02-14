"""
Local temperature data reader for temperature-dependent refraction correction.

Reads daily temperature measurements from a local CSV file (e.g., from a
home weather sensor). Falls back to a standard temperature if data is missing.
"""

import csv
from datetime import date
from pathlib import Path


FALLBACK_TEMP_C = 10.0  # Standard atmosphere assumption


class TemperatureDataError(Exception):
    """Raised when temperature data cannot be loaded."""
    pass


def load_temperatures_from_csv(
    filepath: str | Path,
    date_column: str = "date",
    temp_column: str = "temperature_c",
    delimiter: str = ";",
    fallback_temp_c: float = FALLBACK_TEMP_C,
) -> dict[date, float]:
    """
    Load daily temperature readings from a local CSV file.

    Expected CSV format (semicolon-delimited by default):
        date;temperature_c
        2025-01-01;-12.3
        2025-01-02;-10.8

    Parameters:
        filepath: Path to the CSV file with sensor readings
        date_column: Name of the column containing dates (ISO format)
        temp_column: Name of the column containing temperature in °C
        delimiter: CSV delimiter character
        fallback_temp_c: Default temperature if a value is missing or invalid

    Returns:
        Dict mapping date -> temperature in °C
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise TemperatureDataError(
            f"Temperature file not found: {filepath}"
        )

    temperatures = {}

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Validate that expected columns exist
        if reader.fieldnames is None:
            raise TemperatureDataError(f"CSV file is empty: {filepath}")

        missing_cols = []
        if date_column not in reader.fieldnames:
            missing_cols.append(date_column)
        if temp_column not in reader.fieldnames:
            missing_cols.append(temp_column)
        if missing_cols:
            raise TemperatureDataError(
                f"CSV missing expected columns {missing_cols}. "
                f"Found: {reader.fieldnames}"
            )

        for line_num, row in enumerate(reader, start=2):
            date_str = row.get(date_column, "").strip()
            temp_str = row.get(temp_column, "").strip()

            if not date_str:
                continue

            try:
                d = date.fromisoformat(date_str)
            except ValueError:
                print(f"Warning: Skipping invalid date on line {line_num}: "
                      f"'{date_str}'")
                continue

            try:
                temp = float(temp_str)
            except (ValueError, TypeError):
                print(f"Warning: Invalid temperature on line {line_num} "
                      f"({date_str}): '{temp_str}'. "
                      f"Using fallback {fallback_temp_c}°C.")
                temp = fallback_temp_c

            temperatures[d] = temp

    if not temperatures:
        print(f"Warning: No valid temperature records found in {filepath}.")

    return temperatures


def get_temperature(
    daily_temps: dict[date, float],
    target_date: date,
    fallback_temp_c: float = FALLBACK_TEMP_C,
) -> float:
    """
    Look up the temperature for a specific date, with fallback.

    Parameters:
        daily_temps: Dict from load_temperatures_from_csv()
        target_date: The date to look up
        fallback_temp_c: Value to return if the date has no reading

    Returns:
        Temperature in °C
    """
    temp = daily_temps.get(target_date)
    if temp is None:
        print(f"Warning: No sensor reading for {target_date.isoformat()}. "
              f"Using fallback {fallback_temp_c}°C.")
        return fallback_temp_c
    return temp
