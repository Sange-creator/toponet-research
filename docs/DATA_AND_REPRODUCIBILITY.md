# Data and reproducibility

## Public release contents

This repository contains the Python implementation, reference configuration files, code-only numerical checks, a privacy-filtered extract of separate highway dashboard observations, and the 46-site Kathmandu cell-layout workbook. See [`../data/highway-dashboard-observation/README.md`](../data/highway-dashboard-observation/README.md) and [`../data/kathmandu-osm-cell-layout/README.md`](../data/kathmandu-osm-cell-layout/README.md) for their provenance and limits. Route splits, trained checkpoints, per-scenario simulation traces, complete simulation evaluation records, private backup indexes, and generated simulation analysis outputs are not included.

The Kathmandu site-location database is attributed to OpenStreetMap contributors and released under ODbL 1.0. The workbook's additional network-configuration fields are included as supplied layout/simulation attributes; the release does not establish that they are operator records. The remaining simulation artifacts stay outside this release because their redistribution rights and attribution requirements have not been cleared. The highway extract is author-provided dashboard data; it is not operator data and its exact route segment is not encoded. Phone-number and raw-OCR fields and source videos were excluded. Do not treat this extract as validation of the learned controller.

The repository `.gitignore` excludes arbitrary local data, checkpoints, evidence, reports, provenance, and run outputs. Only the documented public datasets under `data/highway-dashboard-observation/` and `data/kathmandu-osm-cell-layout/` are tracked.

## Reproduction limits

The result table in the README summarizes a completed private simulation archive. The archive includes more than the source code: it records the scenario inputs, frozen policy identities, completed cases, pairing information, and aggregate analysis. Those records are not distributed here.

As a result, a clean clone includes the Kathmandu site layout but still cannot independently reproduce the reported evaluation. The route partitions, trained checkpoints, and complete case records are unavailable. Dataset-backed commands require compatible local inputs and checkpoint files. The tested-local dependency inventory is not a validated cross-platform lock file.

Do not publish inferred cell-site information as verified operator infrastructure. The highway extract in the current GitHub repository is covered by the repository's MIT License. The earlier Zenodo `v0.1.0` snapshot remains separately available under CC BY 4.0 at [DOI 10.5281/zenodo.23922329](https://doi.org/10.5281/zenodo.23922329).

## Validation boundaries

The published code should not be interpreted as field validation. The transfer evaluation uses real road geometry and synthetic cell layouts. The same-area route identifiers are held out, but the road segments overlap with the training pool. Conditional confidence intervals describe the sampled scenarios and frozen policies, not the population of cities, networks, or training seeds.
