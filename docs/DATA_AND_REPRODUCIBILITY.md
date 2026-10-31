# Data and reproducibility

## Public release contents

This repository contains the Python implementation, reference configuration files, and code-only numerical checks. It does not contain route geometry, cell-layout inputs, trained checkpoints, per-scenario traces, complete evaluation records, private backup indexes, or generated analysis outputs.

Those artifacts remain outside the public release because redistribution rights and attribution requirements have not been cleared. OpenStreetMap-derived data may carry separate attribution and share-alike obligations. Do not add local research data or model artifacts to a public commit without resolving those requirements.

The repository `.gitignore` excludes local data, checkpoints, evidence, reports, provenance, and run outputs. Keep those folders out of public commits even when present in a local working copy.

## Reproduction limits

The result table in the README summarizes a completed private simulation archive. The archive includes more than the source code: it records the scenario inputs, frozen policy identities, completed cases, pairing information, and aggregate analysis. Those records are not distributed here.

As a result, a clean clone can be used to inspect the implementation and run the code-only checks, but it cannot independently reproduce the reported evaluation. Dataset-backed commands require compatible local inputs and checkpoint files. The tested-local dependency inventory is not a validated cross-platform lock file.

For any future data release, publish a separate data manifest with source, version, collection date, attribution, redistribution license, checksums, and instructions for obtaining the data. Do not publish inferred cell-site information as verified operator infrastructure.

## Validation boundaries

The published code should not be interpreted as field validation. The transfer evaluation uses real road geometry and synthetic cell layouts. The same-area route identifiers are held out, but the road segments overlap with the training pool. Conditional confidence intervals describe the sampled scenarios and frozen policies, not the population of cities, networks, or training seeds.
