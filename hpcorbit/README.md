# Non-GNN Ablation on HPCOrbit

This directory implements a graph-free control for the TopoNet DDQN handover
study. It replaces the two graph-attention layers with two shared, node-wise
fully connected layers. The adjacency matrix is accepted only to preserve the
agent interface and is not read by the ablation model.

The experiment keeps the DDQN update, dueling heads, state features, action
mask, reward, replay buffer, training schedule, route sampling, and evaluation
cases unchanged. For each trained graph-attention policy, prepare an MLP run
from the exact GAT configuration used for that seed. The preparation script
preserves all settings and records hashes for the GAT config and ablation code.

This comparison estimates the contribution of graph message passing within
this simulator and training setup. It does not establish that GNNs are
generally superior to non-GNN methods.

## Files

```text
hpcorbit/
├── README.md
├── nodewise_mlp.py
├── prepare_non_gnn_configs.py
├── train_non_gnn.py
├── train_non_gnn.slurm
├── submit_non_gnn.sh
├── evaluate_non_gnn.py
├── evaluate_non_gnn.slurm
└── hpcorbit_env.sh
```

## Prerequisites

The private project data, split files, archived evaluation cases, and frozen
GAT checkpoints must already be present on HPCOrbit in the project directory.
They are not included in the public source repository. The project package
must be installed in the Python environment used by compute jobs.

From the project root, create or activate the environment approved for your
HPCOrbit allocation. For a Python environment already provisioned with PyTorch:

```bash
python3 -m pip install -e '.[analysis]'
```

Edit `hpcorbit/hpcorbit_env.sh` for the Python module and environment used on
the compute nodes. The example module commands are intentionally commented out
because module names and environment rules depend on the HPCOrbit installation.

Review the resource requests in `train_non_gnn.slurm` and
`evaluate_non_gnn.slurm`. Add the account and partition directives required by
your allocation. The defaults request CPU jobs; training code uses CPU
inference and optimization.

## Prepare matched configurations

Pass the exact GAT JSON files used for the reported policies, one file per
training seed. Do not create new seeds or route schedules for this ablation.
The script changes no seed, episode count, reward setting, route setting,
reference configuration, or split configuration.

```bash
python hpcorbit/prepare_non_gnn_configs.py \
  /path/to/GAT_seed17.json \
  /path/to/GAT_seed27.json \
  /path/to/GAT_seed37.json \
  --output-dir configs/ablation_non_gnn
```

Use the actual filenames stored with the original experiments. If the original
GAT configurations are unavailable, do not infer their route seeds from the
agent seed. Recover the exact configurations from the experiment archive first.
The preparation script refuses to overwrite generated configuration files.

## Train

Submit one Slurm array task per MLP configuration. The array is limited to one
concurrent task by default to reduce load on the shared cluster.

```bash
PROJECT_ROOT="$PWD" MAX_CONCURRENT=1 bash hpcorbit/submit_non_gnn.sh
```

The script writes logs to `logs/ablation/` and run outputs to
`runs/ablation/non_gnn/`. Each run has a distinct folder and preserves the
runner's boundary recovery files. If a job is interrupted, resubmit only its
array index with `RESUME=1`; do not resume an already completed run. The index
is the position of the sorted JSON file in `configs/ablation_non_gnn/`.

```bash
PROJECT_ROOT="$PWD" CONFIG_DIR="$PWD/configs/ablation_non_gnn" \
OUTPUT_ROOT="$PWD/runs/ablation/non_gnn" RESUME=1 \
sbatch --array=0 hpcorbit/train_non_gnn.slurm
```

Replace `0` with the interrupted task index. The runner checks that the
configuration, project source, and runtime match before it resumes.

## Evaluate

Evaluation selects each checkpoint using development routes only. It then
evaluates the selected MLP policies on the same frozen held-out or Pokhara
scenarios used for the archived GAT and A3-style results. Archived rows are
reused only after their recorded hashes and case pairing are verified.

Submit separate jobs for each evaluation area. Use a new output directory for
each evaluation attempt because completed results are never overwritten.

```bash
PROJECT_ROOT="$PWD" AREA=heldout \
OUTPUT="$PWD/runs/ablation/evaluation/heldout_mlp_v1" \
sbatch hpcorbit/evaluate_non_gnn.slurm

PROJECT_ROOT="$PWD" AREA=pokhara \
OUTPUT="$PWD/runs/ablation/evaluation/pokhara_mlp_v1" \
sbatch hpcorbit/evaluate_non_gnn.slurm
```

The output directory contains:

```text
COMPLETE.json      Output hashes and completion status
per_case.jsonl     Paired archived and MLP results for each scenario
summary.json       Aggregate metrics and route-clustered intervals
```

The summary reports each MLP policy against A3 and its archived GAT policy
with the same training seed. The pairwise intervals use the same scenario and
route-cluster pairing.
The intervals are conditional on the frozen policies and evaluated scenarios.
They do not measure variability across training seeds, cities, or layout
populations. Report all three matched seed results and the pooled scenario
comparison. Do not select a seed or metric after looking at the held-out
results.

## Runtime environment

The site setup file is sourced inside every compute job. Edit it to match your
HPCOrbit modules or environment activation method. Keep account credentials,
tokens, and private data paths out of tracked source files.
