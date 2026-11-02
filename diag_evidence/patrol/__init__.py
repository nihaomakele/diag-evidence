"""Online fleet patrol: observations, allocation, execution and feedback."""

from .simulator import simulate
from .scenarios import make_scenario

__all__ = ['simulate', 'make_scenario']
