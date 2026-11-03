# TopoNet Research Code

This repository contains the simulation code for a graph-attention Dueling Double Deep Q-Network (DDQN) study of cellular handover decisions. The implementation models a 46-cell network graph and uses 15 per-cell input features to select a serving cell during simulated movement.

The project is intended for research review and reproducibility work. It is not a production network controller. It has not been validated on an operating cellular network.

The identified manuscript, its PDF, LaTeX source, and figures are in [`paper/`](paper/). The paper describes the method, evaluation, results, and study limits. Its simulation inputs and complete case records are not included, so the published results cannot be reproduced from this repository alone.

## Research scope

The agent combines graph attention with a Dueling Double DQN. Its observation includes simulator-derived radio, rate, load, mobility, and recent handover information. It also uses one-step route lookahead. These inputs are simulation assumptions; their availability to a handset or operator system has not been established.

The code compares three frozen learned policies with a configured A3-style handover controller. The reported evaluation includes 639 same-area held-out route cases and 540 Pokhara simulation groups. The latter use real road geometry with synthetic cell layouts.

## Highway dashboard observation

The paper's Introduction describes a separate iPhone 15 network-dashboard recording from travel along the Prithvi Highway. A privacy-filtered, one-second extract from three recordings is in [`data/highway-dashboard-observation/`](data/highway-dashboard-observation/). It contains displayed cell identifiers and radio readings, with unavailable values left blank. The observations provide field context only; they were not used to train, tune, or evaluate the controller. The phone-number and raw OCR fields, and the original screen recordings, are not included. The exact highway segment and travel direction are not recorded in the workbook. See the folder README for field definitions and limits.

## Reported simulation results

| Evaluation | Learned-policy throughput change | Learned-policy handover-rate change |
| --- | ---: | ---: |
| Same-area reserved routes | +8.86% to +9.99% | -36.08% to -53.00% |
| Pokhara transfer scenarios | +3.66% to +3.85% | -23.68% to -47.10% |

These aggregate results are from the project's private experiment archive. The public repository does not include the raw inputs, complete case records, or trained policies needed to reproduce them. Same-area test routes share road segments with training routes. The results do not establish unseen-road generalization, performance on real networks, or deployment readiness.

The evaluation has material limitations. Same-area short-stay proportions increased for all three learned policies. Only three training seeds were evaluated. No non-GNN ablation results are included yet, so these results do not establish that graph message passing is necessary. The reported comparison is against one configured A3-style baseline. These results should be interpreted as a bounded simulation comparison. A controlled MLP ablation workflow is provided in `hpcorbit/`.

## Repository layout

```text
toponet-research/
├── README.md
├── pyproject.toml
├── configs/
│   ├── reference.json
│   ├── splits.json
│   └── training/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── DATA_AND_REPRODUCIBILITY.md
│   └── RESULTS.md
├── data/
│   └── highway-dashboard-observation/  # Privacy-filtered field-context extract
├── paper/            # Identified IOEGC manuscript, PDF, and LaTeX source
├── src/toponet/
│   ├── agents/       # Double-DQN agent and replay buffer
│   ├── baselines/    # A3-style and load-aware controllers
│   ├── data/         # Geometry, movement, graph, and data readers
│   ├── envs/         # Radio, service, traffic, features, and handover
│   ├── evaluation/   # Replay, diagnostics, metrics, and analysis
│   ├── models/       # Graph-attention Q-network
│   └── training/     # Episode scheduling, sampling, and recovery
├── hpcorbit/         # Graph-free MLP ablation and Slurm workflows
│   ├── README.md
│   ├── nodewise_mlp.py
│   ├── prepare_non_gnn_configs.py
│   ├── train_non_gnn.py
│   ├── evaluate_non_gnn.py
│   └── *.slurm / *.sh
└── tests/
    └── test_core.py
```

The simulation datasets, trained policies, detailed experiment records, generated reports, and backup metadata are excluded because redistribution rights and attribution have not been cleared. The separate highway dashboard extract is described in [data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md); it is not simulation input or model evaluation data.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) describes the runtime flow and package responsibilities.
- [Configuration](docs/CONFIGURATION.md) explains how to create controlled training configurations.
- [Results](docs/RESULTS.md) records the reported outcomes and their interpretation limits.
- [Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) documents the omitted assets and replication boundaries.
- [HPCOrbit ablation](hpcorbit/README.md) documents the matched non-GNN training and evaluation workflow.
- [Paper](paper/README.md) explains the manuscript files and how to build the PDF.

## Requirements

- Python 3.11 or newer
- PyTorch 2.0 or newer
- NumPy 1.24 or newer
- OpenPyXL 3.1 or newer
- Matplotlib 3.6 or newer for analysis plots

Install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis]'
python -m toponet --help
```

The command-line help and core numerical checks do not require the private experiment artifacts. Dataset-backed training, replay, and full analysis commands do require local inputs and checkpoints that are not part of this repository.

## Core checks

Run the code-only numerical self-checks with:

```bash
python -m unittest discover -s tests -p 'test_core.py' -v
```

Other historical integration checks depend on private datasets, model weights, or backup files. They are retained in the owner's research workspace and are not included in this public repository.

## License

No software license has been selected for this release. The repository is public for inspection, but public visibility does not grant permission to reuse or redistribute the code.
