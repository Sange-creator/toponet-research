# Making controlled changes

Copy a training JSON to a new filename. Edit `episodes`, `agent_seed`, `route_seed`, `handover_cost`, `threads`, `validation_target_steps`, `agent_overrides` or `network_overrides`. Settings are validated by the canonical classes. The new run records the resolved settings, source/config hashes, runtime and planned journey sequence.

For a separately declared simulator/split experiment, training JSON may specify `reference_config` and `split_config`, relative to the project root or absolute. Defaults are `configs/reference.json` and `configs/splits.json`. Copy them before changing anything. Graph/layout/route hashes and feature/radio semantics are checked; changing one file does not magically make an incompatible dataset valid. Keep the frozen originals for replay and verification.

Feature order is part of checkpoint identity. Existing selected checkpoints require exactly 15 ordered features and their original graph for native inference. Pokhara uses a separate audited inference adapter. Changing feature meaning, layer shapes or graph semantics requires a NEW experiment/checkpoint, not relabeling old results.

Each `runs/<name>` folder contains `run_config.json`, `planned_journeys.json`, `baseline_development.json`, `training.jsonl`, `validation.jsonl`, `progress.json`, policy checkpoint files, `recovery_latest.pt`, `recovery_previous.pt`, `RECOVERY_STATUS.json` and finally `COMPLETE.json`. The short smoke command intentionally writes a smaller subset. Run status and validation metrics are written after committed journey boundaries. Nothing is automatically published or selected from test data.

Policy checkpoints are frozen-inference weights; recovery files contain full CPU training state. Keep both when new training must resume. Avoid copying a live recovery file as if it were a stable committed snapshot; backup checks report changes rather than hiding them.

The default driver reproduces the historical selected-cost schedule but is a NEW orchestration layer, not a bitwise recreation of the original multi-stage reward-search scheduler. Fresh full-budget equality has not been established. Covered core computation, original schedule, checkpoint loading, tiny exact recovery and selected historical frozen actions have been tested.
