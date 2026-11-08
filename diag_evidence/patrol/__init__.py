"""Online fleet patrol: observations, allocation, execution and feedback."""

from .simulator import simulate
from .scenarios import make_scenario
from .execution import ExecutionConfig

__all__ = ['simulate', 'make_scenario', 'ExecutionConfig']
