# Architecture

TopoNet simulates cellular handover decisions and evaluates learned policies against baseline controllers. The main runtime is organized as a Python package under `src/toponet`.

## Runtime flow

1. Data readers load route, cell, and simulation configuration inputs.
2. The environment computes radio conditions, user and background traffic, candidate-cell features, legal actions, handover interruption, and reward.
3. A graph-attention encoder processes per-cell features and the cell-neighbor graph.
4. A dueling Double-DQN agent selects a legal serving-cell action and updates its policy from replayed transitions during training.
5. Training utilities schedule complete journeys and save recovery state at defined boundaries.
6. Evaluation utilities replay frozen policies and calculate paired scenario metrics.

## Package areas

| Package | Responsibility |
| --- | --- |
| `agents` | Double-DQN action selection, optimization, and replay memory |
| `models` | Graph-attention encoder and dueling value/advantage heads |
| `envs` | Radio, allocation, background traffic, state features, rewards, and handover transitions |
| `baselines` | Configured A3-style and load-aware controllers |
| `data` | Geometry, route movement, graph construction, and input validation |
| `training` | Journey sampling, schedules, run management, and boundary recovery |
| `evaluation` | Frozen-policy replay, transfer adapters, diagnostics, and analysis |

## Simulation information assumptions

The policy uses modeled radio and load estimates, candidate-rate information, and known-route lookahead. These are simulator inputs. The project does not establish that the same observations are available in a live handset or operator network.

The model and environment are research software. The project does not implement cellular signaling, network handover execution, or a deployment interface.
