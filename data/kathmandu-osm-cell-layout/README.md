# Kathmandu 46-site cell-layout workbook

This directory contains the 46-row workbook used as the Kathmandu-area cell-layout input. It has one worksheet and 17 columns. The workbook is copied unchanged from the authors' supplied file.

## Provenance and interpretation

The manuscript reports that the authors obtained 46 mapped cell-site locations from OpenStreetMap (OSM) and checked the layout coordinates against the corresponding OSM-mapped locations. The latitude and longitude are in decimal degrees. The workbook does not record an OSM snapshot date or OSM object identifiers.

The workbook also contains network identifiers and configuration fields (`enodeb_id`, `cell_id`, `pci`, `earfcn`, `tac`, `bands`, `bandwidth_mhz`, `prb_capacity`, `azimuth`, `num_sectors`, `height_m`, `tx_power_dbm`, and `baseline_load_pct`). The workbook does not document independent sources or operator-side validation for these fields. They are included as supplied cell-layout/simulation inputs; they should not be interpreted as verified live-network telemetry or as operator-certified infrastructure data.

`baseline_load_pct` is stored as a decimal fraction in the workbook (for example, `0.85` represents 85%). Other units are indicated by the column names where present.

## OpenStreetMap attribution and data license

The OSM-derived site-location database is attributed to OpenStreetMap contributors and made available under the Open Database License (ODbL) 1.0. See <https://www.openstreetmap.org/copyright> and <https://opendatacommons.org/licenses/odbl/1-0/>. The authors' supplied configuration fields are not represented as OSM observations.

## Limits

This file documents the cell-layout input. It does not include the route splits, trained checkpoints, or complete case-level simulation results needed to reproduce the full evaluation. See [`../../docs/DATA_AND_REPRODUCIBILITY.md`](../../docs/DATA_AND_REPRODUCIBILITY.md) for the remaining release limits.
