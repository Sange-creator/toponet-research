#!/usr/bin/env bash
# Site-specific setup for the HPCOrbit compute nodes.
# Edit this file to load the Python module and activate the environment used by
# the project. Keep cluster credentials out of this file and out of Git.

# Example only. Replace with module names documented by your HPCOrbit admins.
# module purge
# module load python/3.11
# source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
