# DiagEvidence Patrol

Online multi-robot patrol scheduling, evidence checks and reproducible replay. Development version: **0.3.0.dev0**.

The centralized simulator closes the loop from asynchronous observations to assignment, A* movement, service execution and updated observations. It includes task releases/deadlines/priorities, energy reserves, outages, traffic arbitration and replanning.

## Quick start

Python >=3.10, from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
diag-evidence patrol-demo --scenario mixed --seed 7 --output-dir artifacts/patrol
```

Open the returned `replay_path` in a browser. The portable HTML includes synchronized maps, playback controls, event history and final metrics for direct-message, verified-message and perfect-task-state policies. Each policy uses the same physical mission and allocator. The default mixed scenario completes 4/8 tasks directly and 8/8 after validation.

Choose `--allocator greedy` for the greedy control, or `--config examples/patrol_mission.json` for a custom mission. The exact matching allocator maximizes one-step additive assignment utility with idle actions, bounded to 8 robots and 64 tasks. Paths use four-neighbor A* and static-route caching; execution reserves cells, prevents edge swaps and replans around traffic.

## Evaluation

```bash
diag-evidence patrol-evaluate --split development --output-dir artifacts/patrol-development
diag-evidence patrol-freeze --development-dir artifacts/patrol-development --output artifacts/patrol-freeze.json
diag-evidence patrol-evaluate --split held_out --freeze artifacts/patrol-freeze.json --output-dir artifacts/patrol-held-out
```

Eight scenarios, two allocators and three information policies across 40 seeds produce 1,920 simulation runs. Seeds 0–19 are development layouts; 20–39 are evaluated after source freezing. Results include paired completion rates, energy, travel, reinspections, communication payload and outage reassignment. Local regression: 340 tests, including 32 new patrol tests. In the held-out mixed scenario, mean on-time completion is 49.375% for direct messages and 99.375% for verified messages. The fresh-error control yields equal direct/verified performance. Delayed reports expose repeated-service overhead.

See [design and assumptions](docs/patrol_scheduling.md), [project explanation](docs/patrol_project_story.md) and [full results](docs/results/patrol_20261102.json).

The first version models a known static 2D grid, reliable current robot state and asynchronous task-state observations. Motion and energy use discrete units. Planned adapters target task acknowledgements/idempotency, ROS2 navigation and distributed allocation.

The diagnostic workflow and bearing benchmarks remain available as earlier modules. `diag-evidence demo --scenario all --no-write` runs the diagnostic contract scenarios. The published v0.1.1 tag is preserved. Code: MIT; data terms are recorded separately in [provenance](DATA_PROVENANCE.md).
