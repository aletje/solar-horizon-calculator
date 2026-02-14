"""
Solar Horizon Calculator

Computes sunrise and sunset times for a given observer location,
accounting for local terrain via DTM-based horizon profiles,
atmospheric refraction, Earth curvature, and sun disk geometry.
"""

import math
from datetime import datetime, timezone, timedelta, date
import csv
import zoneinfo
from dataclasses import dataclass, field

import numpy as np
import rasterio
from pyproj import Transformer
from skyfield.api import load, Topos

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0
SUN_RADIUS_DEG = 0.2665       # Angular radius of the sun (~0.533° diameter / 2)
LAT_METERS_PER_DEG = 111_320.0  # Approximate meters per degree of latitude


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SolarCalcConfig:
    """Configuration for the solar horizon calculator."""
    dtm_file: str
    observer_lat: float
    observer_lon: float
    observer_elev: float | None = None
    obs_offset: float = 1.7
    azimuth_step: float = 1.0
    max_distance: float = 50_000.0
    timezone: str = "Europe/Oslo"
    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))
    temperature_file: str | None = None


# ---------------------------------------------------------------------------
# Ephemeris caching (lazy singleton)
# ---------------------------------------------------------------------------

_ts = None
_eph = None


def _get_ephemeris():
    """Load and cache Skyfield timescale and ephemeris data."""
    global _ts, _eph
    if _ts is None:
        _ts = load.timescale()
        _eph = load('de421.bsp')
    return _ts, _eph


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def curvature_drop(distance_m: float) -> float:
    """Approximate drop in apparent height due to Earth's curvature."""
    return distance_m ** 2 / (2 * EARTH_RADIUS_M)


def ray_pixels(r0, c0, azimuth_rad, length, height, width):
    """
    Vectorized ray-marching: generate all pixel coordinates along a ray
    using numpy, replacing the pixel-by-pixel Bresenham approach.

    Parameters:
        r0, c0: Observer row/col in local grid
        azimuth_rad: Azimuth in radians (0=N, clockwise)
        length: Ray length in pixels
        height, width: Grid dimensions

    Returns:
        (rows, cols) arrays of valid pixel indices along the ray
    """
    sin_a = math.sin(azimuth_rad)
    cos_a = math.cos(azimuth_rad)
    num_steps = int(length) + 1
    t = np.arange(num_steps, dtype=float)
    cols = np.round(c0 + sin_a * t).astype(int)
    rows = np.round(r0 - cos_a * t).astype(int)
    # Clip to valid bounds
    mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows[mask], cols[mask]


# ---------------------------------------------------------------------------
# Horizon profile computation
# ---------------------------------------------------------------------------

def compute_horizon_profile(
    dtm_file: str,
    observer_lat: float,
    observer_lon: float,
    observer_elev: float = None,
    obs_offset: float = 0.0,
    azimuth_step: float = 1.0,
    max_distance: float = None
):
    """
    Computes the local horizon profile (maximum elevation angle) around
    an observer, accounting for Earth curvature and geographic CRS.

    Returns:
        (azimuths, horizon_angles) — arrays of azimuth values and
        corresponding maximum elevation angles in degrees.
    """
    with rasterio.open(dtm_file) as ds:
        dtm_crs = ds.crs
        is_geographic = bool(dtm_crs and dtm_crs.is_geographic)
        obs_x, obs_y = observer_lon, observer_lat

        if not is_geographic and dtm_crs:
            transformer = Transformer.from_crs("EPSG:4326", dtm_crs, always_xy=True)
            obs_x, obs_y = transformer.transform(observer_lon, observer_lat)

        try:
            obs_row, obs_col = ds.index(obs_x, obs_y)
        except Exception as e:
            raise RuntimeError(
                f"Observer position ({observer_lat}, {observer_lon}) is outside DTM: {e}"
            )

        try:
            obs_dem_value = next(ds.sample([(obs_x, obs_y)]))[0]
        except Exception:
            obs_dem_value = ds.read(
                1, window=rasterio.windows.Window(obs_col, obs_row, 1, 1)
            )[0, 0]

        if math.isnan(obs_dem_value):
            raise ValueError("DTM contains no valid elevation for the observer position.")

        obs_elev = observer_elev if observer_elev is not None else obs_dem_value + obs_offset

        # Determine sub-window bounds
        if max_distance is not None:
            pixel_size_x = abs(ds.transform.a)
            pixel_size_y = abs(ds.transform.e) if ds.transform.e != 0 else abs(ds.transform.a)

            if is_geographic:
                deg_per_meter_lat = 1.0 / LAT_METERS_PER_DEG
                deg_per_meter_lon = 1.0 / (
                    LAT_METERS_PER_DEG * math.cos(math.radians(observer_lat))
                )
                pixel_count_lat = int((max_distance * deg_per_meter_lat) / pixel_size_y) + 1
                pixel_count_lon = int((max_distance * deg_per_meter_lon) / pixel_size_x) + 1
            else:
                pixel_count_lat = int(max_distance / pixel_size_y) + 1
                pixel_count_lon = int(max_distance / pixel_size_x) + 1

            pixel_count_lat = min(pixel_count_lat, ds.height - 1)
            pixel_count_lon = min(pixel_count_lon, ds.width - 1)
            row_min = max(obs_row - pixel_count_lat, 0)
            row_max = min(obs_row + pixel_count_lat, ds.height - 1)
            col_min = max(obs_col - pixel_count_lon, 0)
            col_max = min(obs_col + pixel_count_lon, ds.width - 1)
        else:
            row_min, col_min = 0, 0
            row_max, col_max = ds.height - 1, ds.width - 1
            pixel_size_x = abs(ds.transform.a)
            pixel_size_y = abs(ds.transform.e) if ds.transform.e != 0 else abs(ds.transform.a)

        if row_max < row_min or col_max < col_min:
            raise ValueError(
                f"Invalid sub-window: row_min={row_min}, row_max={row_max}, "
                f"col_min={col_min}, col_max={col_max}."
            )

        height = row_max - row_min + 1
        width = col_max - col_min + 1
        if height < 1 or width < 1:
            raise ValueError("Sub-window has non-positive size.")

        window = rasterio.windows.Window(col_min, row_min, width, height)
        dem_data = ds.read(1, window=window)
        transform = ds.transform * rasterio.Affine.translation(-col_min, -row_min)

    # Local observer coordinates within the sub-window
    obs_row_local = obs_row - row_min
    obs_col_local = obs_col - col_min

    obs_x_coord = transform.c + obs_col_local * transform.a + obs_row_local * transform.b
    obs_y_coord = transform.f + obs_col_local * transform.d + obs_row_local * transform.e

    azimuths = np.arange(0, 360, azimuth_step, dtype=float)
    horizon_angles = np.empty_like(azimuths)

    px_size = max(pixel_size_x, pixel_size_y)

    for i, az in enumerate(azimuths):
        az_rad = math.radians(az)
        sin_a = math.sin(az_rad)
        cos_a = math.cos(az_rad)

        # Compute maximum ray length in pixels
        if abs(sin_a) < 1e-14:
            t_col = float('inf')
        else:
            col_dist = (width - 1 - obs_col_local) if sin_a > 0 else obs_col_local
            t_col = col_dist / abs(sin_a) if col_dist > 0 else 0.0

        if abs(cos_a) < 1e-14:
            t_row = float('inf')
        else:
            row_dist = obs_row_local if cos_a > 0 else (height - 1 - obs_row_local)
            t_row = row_dist / abs(cos_a) if row_dist > 0 else 0.0

        t_max_edge = min(v for v in [t_col, t_row] if v > 0) if any(
            v > 0 for v in [t_col, t_row]
        ) else 0.0

        if t_max_edge <= 0:
            horizon_angles[i] = -90.0
            continue

        if max_distance is not None and px_size > 0:
            t_dist = max_distance / px_size
            t_use = min(t_max_edge, t_dist)
        else:
            t_use = t_max_edge

        if t_use <= 0:
            horizon_angles[i] = -90.0
            continue

        # Use vectorized ray instead of Bresenham
        rows, cols = ray_pixels(obs_row_local, obs_col_local, az_rad, t_use, height, width)

        if len(rows) < 2:
            horizon_angles[i] = -90.0
            continue

        elev_profile = dem_data[rows, cols]

        x_coords = transform.c + cols * transform.a + rows * transform.b
        y_coords = transform.f + cols * transform.d + rows * transform.e

        dx = x_coords - obs_x_coord
        dy = y_coords - obs_y_coord

        # Convert to meters if CRS is geographic
        if is_geographic:
            lon_scale = LAT_METERS_PER_DEG * math.cos(math.radians(observer_lat))
            dx_m = dx * lon_scale
            dy_m = dy * LAT_METERS_PER_DEG
            distances = np.sqrt(dx_m ** 2 + dy_m ** 2)
        else:
            distances = np.sqrt(dx ** 2 + dy ** 2)

        # Apply Earth curvature correction
        height_diffs = elev_profile - obs_elev - curvature_drop(distances)
        alt_angles = np.degrees(np.arctan2(height_diffs, distances))

        horizon_angles[i] = alt_angles[1:].max() if len(alt_angles) > 1 else -90.0

    return azimuths, horizon_angles


# ---------------------------------------------------------------------------
# Bisection helper
# ---------------------------------------------------------------------------

def bisect_crossing(observer, sun, t_before, t_after, horizon_az_ext,
                    horizon_alt_ext, refraction_deg, rising=True, iterations=20):
    """
    Binary search for the exact moment the sun crosses the local horizon.

    Parameters:
        observer: Skyfield observer
        sun: Skyfield sun object
        t_before, t_after: Skyfield Time bounds
        horizon_az_ext, horizon_alt_ext: Extended horizon profile arrays
        refraction_deg: Atmospheric refraction angle in degrees
        rising: True for sunrise (negative->positive), False for sunset
        iterations: Number of bisection iterations

    Returns:
        datetime in UTC
    """
    for _ in range(iterations):
        t_mid = t_before + (t_after - t_before) / 2
        alt_mid, az_mid, _ = observer.at(t_mid).observe(sun).apparent().altaz()
        hor_alt_mid = np.interp(az_mid.degrees, horizon_az_ext, horizon_alt_ext)
        diff_mid = alt_mid.degrees + SUN_RADIUS_DEG + refraction_deg - hor_alt_mid
        if abs(diff_mid) < 1e-4:
            break
        if rising:
            if diff_mid < 0:
                t_before = t_mid
            else:
                t_after = t_mid
        else:
            if diff_mid > 0:
                t_before = t_mid
            else:
                t_after = t_mid
    return t_mid.utc_datetime()


# ---------------------------------------------------------------------------
# Sunrise / sunset computation
# ---------------------------------------------------------------------------

def find_sunrise_sunset(
    observer_lat: float,
    observer_lon: float,
    observer_elev: float,
    obs_offset: float,
    horizon_az: np.ndarray,
    horizon_alt: np.ndarray,
    target_date: date,
    refraction_deg: float = 0.575
):
    """
    Computes local sunrise and sunset times given a horizon profile.

    Parameters:
        observer_lat: Latitude in degrees
        observer_lon: Longitude in degrees
        observer_elev: Observer elevation in meters
        obs_offset: Observer height offset in meters (e.g., eye level)
        horizon_az: Array of azimuth angles in degrees
        horizon_alt: Array of horizon altitude angles in degrees
        target_date: Date for which to compute sunrise/sunset
        refraction_deg: Atmospheric refraction angle in degrees

    Returns:
        (sunrise_utc, sunset_utc) — datetimes in UTC, or
        ("polar_day", "polar_day") if the sun never sets, or
        ("polar_night", "polar_night") if the sun never rises.
    """
    ts, eph = _get_ephemeris()
    sun = eph['Sun']
    earth = eph['Earth']

    observer = earth + Topos(
        latitude_degrees=observer_lat,
        longitude_degrees=observer_lon,
        elevation_m=(observer_elev or 0) + obs_offset
    )

    year, month, day = target_date.year, target_date.month, target_date.day
    t0 = ts.utc(year, month, day, 0, 0, 0)
    t1 = ts.utc(year, month, day, 23, 59, 59)

    minutes = 24 * 60 + 1
    times = ts.linspace(t0, t1, minutes)

    altitudes, azimuts, _ = observer.at(times).observe(sun).apparent().altaz()
    alt_deg = altitudes.degrees
    az_deg = azimuts.degrees

    horizon_az_extended = np.append(horizon_az, 360.0)
    horizon_alt_extended = np.append(horizon_alt, horizon_alt[0])
    horizon_at_sun = np.interp(az_deg, horizon_az_extended, horizon_alt_extended)

    # Include sun radius AND atmospheric refraction
    diff = alt_deg + SUN_RADIUS_DEG + refraction_deg - horizon_at_sun

    sunrise_time = None
    sunset_time = None

    # Find sunrise (first negative->positive crossing)
    for i in range(1, len(diff)):
        if diff[i] >= 0 and diff[i - 1] < 0:
            sunrise_time = bisect_crossing(
                observer, sun, times[i - 1], times[i],
                horizon_az_extended, horizon_alt_extended, refraction_deg, rising=True
            )
            break

    # Find sunset (first positive->negative crossing)
    for i in range(1, len(diff)):
        if diff[i] < 0 and diff[i - 1] >= 0:
            sunset_time = bisect_crossing(
                observer, sun, times[i - 1], times[i],
                horizon_az_extended, horizon_alt_extended, refraction_deg, rising=False
            )
            break

    # Handle polar day / polar night
    if sunrise_time is None and sunset_time is None:
        if diff[0] >= 0:
            return "polar_day", "polar_day"
        else:
            return "polar_night", "polar_night"

    return sunrise_time, sunset_time


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_dst(dt_utc, tz):
    """Reliably check if a UTC datetime falls in DST for the given timezone."""
    if not dt_utc or isinstance(dt_utc, str):
        return False
    local_dt = dt_utc.astimezone(tz)
    return bool(local_dt.dst())


def to_local_str(utc_dt, tz):
    """Convert a UTC datetime to a local time string, or return a label."""
    if not utc_dt:
        return "Ingen"
    if isinstance(utc_dt, str):
        return utc_dt  # e.g. "polar_day" or "polar_night"
    return utc_dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_to_csv(rows, filename):
    """
    Export results to a CSV file.

    Parameters:
        rows: List of dicts with 'date', 'sunrise', 'sunset', 'is_dst'
        filename: Path to the CSV file to write.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Dato", "Soloppgang (lokal)", "Solnedgang (lokal)", "Sommertid?"])
        for row in rows:
            is_dst_str = "Ja" if row['is_dst'] else "Nei"
            writer.writerow([
                row['date'].isoformat(),
                row['sunrise'],
                row['sunset'],
                is_dst_str
            ])


def export_to_pdf(rows, filename, start_date, end_date):
    """
    Export results to a PDF file using reportlab.
    Lines with DST are rendered in bold.
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    x_margin = 50
    y_pos = height - 50
    line_spacing = 14

    # Dynamic title
    if start_date.year == end_date.year:
        title = f"Soltider {start_date.year}"
    else:
        title = f"Soltider {start_date.year}\u2013{end_date.year}"

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y_pos, title)
    y_pos -= 2 * line_spacing

    for row in rows:
        line_text = (
            f"{row['date']} | Opp: {row['sunrise']} | Ned: {row['sunset']} | "
            f"Sommertid: {'Ja' if row['is_dst'] else 'Nei'}"
        )
        if row['is_dst']:
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)

        c.drawString(x_margin, y_pos, line_text)
        y_pos -= line_spacing

        if y_pos < 50:
            c.showPage()
            y_pos = height - 50

    c.showPage()
    c.save()
    print(f"PDF exported to: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from refraction_model import estimate_pressure_from_elevation, refraction_at_horizon
    from weather import load_temperatures_from_csv, get_temperature, FALLBACK_TEMP_C

    config = SolarCalcConfig(
        dtm_file="/Users/aleksandertjernes/downloads/DTM10_UTM32_20250307/6606_4_10m_z32.tif",
        observer_lat=60.2889842362621,
        observer_lon=11.200069130005131,
        obs_offset=1.7,
        azimuth_step=1.0,
        max_distance=50000,
        timezone="Europe/Oslo",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        temperature_file="temperatures.csv",  # Optional: path to sensor CSV
    )

    # 1) Compute horizon profile
    az, hor_alt = compute_horizon_profile(
        dtm_file=config.dtm_file,
        observer_lat=config.observer_lat,
        observer_lon=config.observer_lon,
        obs_offset=config.obs_offset,
        azimuth_step=config.azimuth_step,
        max_distance=config.max_distance,
    )

    # 2) Load temperature data from local sensor CSV (if available)
    daily_temps = {}
    if config.temperature_file:
        try:
            daily_temps = load_temperatures_from_csv(config.temperature_file)
            print(f"Loaded {len(daily_temps)} temperature readings from {config.temperature_file}")
        except Exception as e:
            print(f"Warning: Could not load temperature file: {e}")
            print(f"Using fallback temperature {FALLBACK_TEMP_C}°C for all days.")

    # 3) Estimate atmospheric pressure from observer elevation (once)
    obs_elevation = config.observer_elev if config.observer_elev is not None else 0.0
    pressure_hpa = estimate_pressure_from_elevation(
        elevation_m=obs_elevation,
        temperature_c=15.0  # Standard assumption for pressure estimation
    )
    print(f"Estimated pressure at {obs_elevation}m: {pressure_hpa:.2f} hPa")

    # 4) Timezone
    oslo_tz = zoneinfo.ZoneInfo(config.timezone)

    # 5) Iterate over all days and collect results
    num_days = (config.end_date - config.start_date).days + 1

    results = []
    for d in range(num_days):
        current_date = config.start_date + timedelta(days=d)

        # Get temperature for this day from sensor data (or use fallback)
        temp_c = get_temperature(daily_temps, current_date, FALLBACK_TEMP_C)

        # Compute temperature-dependent refraction for this day
        refraction_deg = refraction_at_horizon(
            temperature_c=temp_c,
            pressure_hpa=pressure_hpa
        )

        sunrise_utc, sunset_utc = find_sunrise_sunset(
            observer_lat=config.observer_lat,
            observer_lon=config.observer_lon,
            observer_elev=config.observer_elev,
            obs_offset=config.obs_offset,
            horizon_az=az,
            horizon_alt=hor_alt,
            target_date=current_date,
            refraction_deg=refraction_deg,
        )

        sunrise_local_str = to_local_str(sunrise_utc, oslo_tz)
        sunset_local_str = to_local_str(sunset_utc, oslo_tz)

        # Robust DST detection
        dst_flag = is_dst(sunrise_utc, oslo_tz) or is_dst(sunset_utc, oslo_tz)

        row_data = {
            'date': current_date,
            'sunrise': sunrise_local_str,
            'sunset': sunset_local_str,
            'is_dst': dst_flag,
        }
        results.append(row_data)

    # 6) Print sample + export
    for row in results[:10]:
        print(row)

    csv_file = "soltider_2026.csv"
    export_to_csv(results, csv_file)
    print(f"Results exported to {csv_file} (CSV).")

    pdf_file = "soltider_2026.pdf"
    export_to_pdf(results, pdf_file, config.start_date, config.end_date)
    print(f"Results exported to {pdf_file} (PDF).")