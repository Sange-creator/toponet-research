"""Data / towers.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from pathlib import Path
import math
from openpyxl import load_workbook

NUMERIC_FIELDS = (
    "site_id",
    "enodeb_id",
    "cell_id",
    "pci",
    "earfcn",
    "tac",
    "lat",
    "lon",
    "bandwidth_mhz",
    "prb_capacity",
    "azimuth",
    "num_sectors",
    "height_m",
    "tx_power_dbm",
    "baseline_load_pct",
)
INTEGER_FIELDS = (
    "site_id",
    "enodeb_id",
    "cell_id",
    "pci",
    "earfcn",
    "tac",
    "prb_capacity",
    "num_sectors",
)
REQUIRED_FIELDS = set(NUMERIC_FIELDS) | {"site_name", "bands"}


def load_cells(filename, expected_count=46):
    """Return validated dictionaries; keep supplied values and add derived fields."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        names = [str(v).strip() if v is not None else None for v in header]
        present = [v for v in names if v is not None]
        if len(present) != len(set(present)):
            raise ValueError("Duplicate spreadsheet column names.")
        missing = REQUIRED_FIELDS - set(present)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        cells = []
        for excel_row, values in enumerate(rows, start=2):
            if all(value is None for value in values):
                continue
            cell = {name: value for name, value in zip(names, values) if name}
            for field in NUMERIC_FIELDS:
                try:
                    value = float(cell[field])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Row {excel_row}: {field} must be numeric."
                    ) from None
                if not math.isfinite(value):
                    raise ValueError(f"Row {excel_row}: {field} must be finite.")
                if field in INTEGER_FIELDS:
                    if not value.is_integer():
                        raise ValueError(
                            f"Row {excel_row}: {field} must be an integer."
                        )
                    value = int(value)
                cell[field] = value

            if not isinstance(cell["site_name"], str) or not cell["site_name"].strip():
                raise ValueError(f"Row {excel_row}: missing site name.")
            if not -90 <= cell["lat"] <= 90 or not -180 <= cell["lon"] <= 180:
                raise ValueError(f"Row {excel_row}: invalid geographic coordinates.")
            if not 0 <= cell["pci"] <= 503:
                raise ValueError(f"Row {excel_row}: PCI is outside LTE range.")
            if not 1200 <= cell["earfcn"] <= 1949:
                raise ValueError(
                    f"Row {excel_row}: this loader currently supports Band 3 only."
                )
            bandwidth_to_prbs = {10.0: 50, 15.0: 75, 20.0: 100}
            if bandwidth_to_prbs.get(cell["bandwidth_mhz"]) != cell["prb_capacity"]:
                raise ValueError(
                    f"Row {excel_row}: unsupported or inconsistent bandwidth/PRB values."
                )
            if not 0 <= cell["baseline_load_pct"] <= 1:
                raise ValueError(
                    f"Row {excel_row}: load must be a fraction between 0 and 1."
                )
            if not 0 <= cell["azimuth"] < 360 or cell["height_m"] <= 0:
                raise ValueError(f"Row {excel_row}: invalid azimuth or antenna height.")

            cell["node_index"] = len(
                cells
            )  # Zero-based simulation index, not a real cell ID.
            cell["operating_band"] = 3
            cell["downlink_frequency_mhz"] = 1805.0 + 0.1 * (cell["earfcn"] - 1200)
            cell["assumed_load_fraction"] = cell["baseline_load_pct"]
            # Original 'bands' and 'num_sectors' are retained as metadata, not expanded.
            cells.append(cell)
    finally:
        workbook.close()

    if len(cells) != expected_count:
        raise ValueError(f"Expected {expected_count} records, found {len(cells)}.")
    for field in ("site_id", "cell_id"):
        if len({cell[field] for cell in cells}) != len(cells):
            raise ValueError(f"Duplicate {field} values.")
    return cells
