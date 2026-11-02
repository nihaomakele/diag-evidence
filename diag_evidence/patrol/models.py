"""Bounded immutable specifications shared by planner and simulator."""
from dataclasses import dataclass, asdict
import re


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value) is not None


def integer(value, low, high):
    return type(value) is int and low <= value <= high


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    blocked: frozenset = frozenset()

    def __post_init__(self):
        if not integer(self.width, 2, 64) or not integer(self.height, 2, 64):
            raise ValueError('Invalid patrol grid dimensions')
        object.__setattr__(self, 'blocked', frozenset(self.blocked))
        if any(not self.contains(cell) for cell in self.blocked):
            raise ValueError('Blocked cell outside grid')

    def contains(self, cell):
        return (isinstance(cell, tuple) and len(cell) == 2 and
                integer(cell[0], 0, self.width - 1) and integer(cell[1], 0, self.height - 1))

    def free(self, cell):
        return self.contains(cell) and cell not in self.blocked


@dataclass(frozen=True)
class RobotSpec:
    id: str
    home: tuple
    battery: int = 100

    def __post_init__(self):
        if not identifier(self.id) or not integer(self.battery, 1, 10000):
            raise ValueError('Invalid patrol robot')


@dataclass(frozen=True)
class TaskSpec:
    id: str
    cell: tuple
    release: int
    deadline: int
    service: int = 2
    priority: int = 1
    revision: int = 1

    def __post_init__(self):
        if (not identifier(self.id) or not integer(self.release, 0, 999)
                or not integer(self.deadline, self.release + 1, 1000)
                or not integer(self.service, 1, 100) or not integer(self.priority, 1, 10)
                or not integer(self.revision, 1, 100000)):
            raise ValueError('Invalid patrol task')


@dataclass(frozen=True)
class Observation:
    mission: str
    task: str
    revision: int
    state: str
    observed_at: int
    available_at: int
    origin: str
    record: str
    relay: str = ''

    def valid(self):
        return (all(identifier(x) for x in [self.mission, self.task, self.origin, self.record])
                and (self.relay == '' or identifier(self.relay))
                and integer(self.revision, 0, 100000)
                and self.state in ('pending', 'completed')
                and integer(self.observed_at, -1000, 1000)
                and integer(self.available_at, 0, 2000))


@dataclass(frozen=True)
class Scenario:
    name: str
    mission: str
    grid: Grid
    robots: tuple
    tasks: tuple
    observations: tuple = ()
    outages: tuple = ()
    horizon: int = 100
    report_delay: int = 0

    def __post_init__(self):
        if (not identifier(self.name) or not identifier(self.mission)
                or not integer(self.horizon, 1, 1000) or not integer(self.report_delay, 0, 100)
                or not 1 <= len(self.robots) <= 8 or not 1 <= len(self.tasks) <= 64
                or len(self.observations) > 10000 or len(self.outages) > 1000):
            raise ValueError('Invalid patrol scenario')
        robot_ids = [r.id for r in self.robots]
        task_ids = [t.id for t in self.tasks]
        if (len(set(robot_ids)) != len(robot_ids) or len(set(task_ids)) != len(task_ids)
                or len({r.home for r in self.robots}) != len(self.robots)
                or any(not self.grid.free(r.home) for r in self.robots)
                or any(not self.grid.free(t.cell) or t.deadline > self.horizon for t in self.tasks)
                or any(not isinstance(o, Observation) or not o.valid() for o in self.observations)):
            raise ValueError('Conflicting patrol identifiers or locations')
        for robot, start, end in self.outages:
            if robot not in robot_ids or not integer(start, 0, self.horizon - 1) or not integer(end, start + 1, self.horizon):
                raise ValueError('Invalid outage interval')


def scenario_from_dict(data):
    required = {'name', 'mission', 'grid', 'robots', 'tasks'}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - (required | {'observations', 'outages', 'horizon', 'report_delay'}):
        raise ValueError('Invalid scenario fields')
    grid = data['grid']
    if not isinstance(grid, dict) or set(grid) != {'width', 'height', 'blocked'}:
        raise ValueError('Invalid grid fields')
    if (not isinstance(data['robots'], list) or not isinstance(data['tasks'], list)
            or any(not isinstance(r, dict) or not {'id', 'home'} <= set(r) for r in data['robots'])
            or any(not isinstance(t, dict) or not {'id', 'cell', 'release', 'deadline'} <= set(t) for t in data['tasks'])):
        raise ValueError('Invalid robot or task fields')
    return Scenario(data['name'], data['mission'],
        Grid(grid['width'], grid['height'], frozenset(tuple(c) for c in grid['blocked'])),
        tuple(RobotSpec(**{**r, 'home': tuple(r['home'])}) for r in data['robots']),
        tuple(TaskSpec(**{**t, 'cell': tuple(t['cell'])}) for t in data['tasks']),
        tuple(Observation(**o) for o in data.get('observations', [])),
        tuple(tuple(o) for o in data.get('outages', [])), data.get('horizon', 100), data.get('report_delay', 0))


def scenario_to_dict(scenario):
    data = asdict(scenario)
    data['grid']['blocked'] = [list(c) for c in sorted(scenario.grid.blocked)]
    return data
