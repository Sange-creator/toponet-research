"""Data / geometry.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import csv
import json
import math
import numpy as np
from toponet.data.towers import load_cells


def _ecef(latitude, longitude):
    """WGS84 Earth-centred positions at zero ellipsoidal height."""
    lat = np.radians(np.asarray(latitude, dtype=float))
    lon = np.radians(np.asarray(longitude, dtype=float))
    a = 6378137.0
    flattening = 1.0 / 298.257223563
    e2 = flattening * (2.0 - flattening)
    radius = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.stack(
        (
            radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * (1.0 - e2) * np.sin(lat),
        ),
        axis=-1,
    )


def latlon_to_xy(latitude, longitude, reference):
    """Use the SAME reference for cells and routes; return [..., 2] metres.

    East/north tangent-plane coordinates approximate ground distances locally.
    This is not a terrain model. Antenna height is handled separately later.
    """
    lat, lon = np.broadcast_arrays(
        np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Coordinates must be finite.")
    if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
        raise ValueError("Invalid latitude or longitude.")
    lat0 = float(reference["latitude_deg"])
    lon0 = float(reference["longitude_deg"])
    if (
        not math.isfinite(lat0)
        or not math.isfinite(lon0)
        or not -90 <= lat0 <= 90
        or not -180 <= lon0 <= 180
    ):
        raise ValueError("Invalid reference coordinates.")
    delta = _ecef(lat, lon) - _ecef(lat0, lon0)
    phi, lam = math.radians(lat0), math.radians(lon0)
    east_axis = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north_axis = np.array(
        [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)]
    )
    return np.stack((delta @ east_axis, delta @ north_axis), axis=-1)


def build_layout(cells):
    """Choose one origin for this dataset and retain row/node order."""
    if not cells:
        raise ValueError("No cells supplied.")
    reference = {
        "latitude_deg": float(np.mean([c["lat"] for c in cells])),
        "longitude_deg": float(np.mean([c["lon"] for c in cells])),
        "projection": "WGS84 ECEF to local east/north tangent plane",
        "units": "metres",
        "note": "Zero ellipsoidal height assumed for ground positions. "
        "Not measured terrain elevations or antenna heights.",
    }
    xy = latlon_to_xy([c["lat"] for c in cells], [c["lon"] for c in cells], reference)
    if xy.shape != (len(cells), 2) or not np.isfinite(xy).all():
        raise ValueError("Unexpected projected layout.")
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    return reference, xy, distances
