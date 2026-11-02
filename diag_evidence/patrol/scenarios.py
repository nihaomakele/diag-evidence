"""Fixed seeded missions and common exogenous messages for all policies."""
import random

from .models import Grid, RobotSpec, TaskSpec, Observation, Scenario

SCENARIOS = ('clean', 'stale', 'replay', 'conflict', 'delayed', 'outage', 'fresh_error', 'mixed')


def make_scenario(name='mixed', seed=7):
    if name not in SCENARIOS or type(seed) is not int or not 0 <= seed < 1000000:
        raise ValueError('Invalid patrol scenario or seed')
    rng = random.Random(seed)
    blocked = frozenset((x, y) for x in (6, 11) for y in range(2, 10) if y not in (4, 5, 8))
    grid = Grid(18, 12, blocked)
    robots = tuple(RobotSpec('R' + str(i), (1 + i, 1), 110) for i in range(3))
    cells = [(x, y) for x in range(2, 17) for y in range(2, 11) if grid.free((x, y))]
    rng.shuffle(cells)
    chosen = [(15, 9)] + [c for c in cells if c != (15, 9)][:7]
    tasks = tuple(TaskSpec('T' + str(i), cell, 0 if i < 4 else 12 + (i - 4) * 5,
                           min(100, 67 + i * 4), 2 + i % 2, 1 + i % 3)
                  for i, cell in enumerate(chosen))
    mission = 'patrol-' + str(seed)
    observations, outages = [], []

    def add(task, state, observed, available, origin, record, **kwargs):
        observations.append(Observation(kwargs.get('mission', mission), task,
            kwargs.get('revision', 1), state, observed, available, origin, record, kwargs.get('relay', '')))

    if name in ('stale', 'mixed'):
        for i in (4, 5):
            add('T' + str(i), 'completed', -15, 0, 'R0', 'old-' + str(i), mission='previous-mission', revision=0)
    if name in ('replay', 'mixed'):
        add('T0', 'pending', 0, 0, 'R0', 'replayed')
        add('T0', 'pending', 0, 1, 'R0', 'replayed', relay='R1')
        add('T0', 'completed', 5, 5, 'R0', 'replayed', relay='R2')
    if name in ('conflict', 'mixed'):
        add('T1', 'pending', 0, 0, 'R0', 'pending-observation')
        add('T1', 'completed', 0, 0, 'R1', 'conflicting-observation')
    if name == 'fresh_error':
        add('T1', 'completed', 0, 0, 'R1', 'fresh-incorrect-observation')
    if name in ('outage', 'mixed'):
        outages.append(('R0', 8, 24))
    return Scenario(name, mission, grid, robots, tasks, tuple(observations), tuple(outages),
                    100, 12 if name == 'delayed' else 0)
