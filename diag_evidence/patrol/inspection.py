"""Translate diagnostic output into scheduled inspections and track findings."""
from collections import Counter
from dataclasses import dataclass

from .models import TaskSpec, identifier, integer


@dataclass(frozen=True)
class Asset:
    id: str
    cell: tuple
    simulated_finding: str
    service: int = 2
    window: int = 30
    priority: int = 9

    def __post_init__(self):
        if (not identifier(self.id) or self.simulated_finding not in ('normal', 'fault', 'inconclusive')
                or not integer(self.service, 1, 100) or not integer(self.window, 1, 1000)
                or not integer(self.priority, 1, 10)):
            raise ValueError('Invalid inspection asset')


@dataclass(frozen=True)
class Alert:
    asset: str
    run_id: str
    observed_at: int
    available_at: int
    classification: str
    supported: bool

    def __post_init__(self):
        if (not identifier(self.asset) or not identifier(self.run_id)
                or not integer(self.observed_at, 0, 1000)
                or not integer(self.available_at, self.observed_at, 1000)
                or self.classification not in ('fault', 'normal', 'unavailable')
                or type(self.supported) is not bool):
            raise ValueError('Invalid diagnostic alert')


@dataclass(frozen=True)
class InspectionPlan:
    assets: tuple
    alerts: tuple

    def __post_init__(self):
        object.__setattr__(self, 'assets', tuple(self.assets))
        object.__setattr__(self, 'alerts', tuple(self.alerts))
        if (not 1 <= len(self.assets) <= 32 or len(self.alerts) > 256
                or any(not isinstance(a, Asset) for a in self.assets)
                or any(not isinstance(a, Alert) for a in self.alerts)):
            raise ValueError('Invalid inspection plan')
        ids = {a.id for a in self.assets}
        if len(ids) != len(self.assets) or any(a.asset not in ids for a in self.alerts):
            raise ValueError('Invalid inspection asset references')


def alert_from_result(*, asset, observed_at, available_at, result):
    """Read bounded fields from a DiagnosisAgent schema 1.0 result."""
    from ..haconfig.settings import class_labels

    def mapping(value):
        return value if isinstance(value, dict) else {}

    if not isinstance(result, dict) or result.get('schema_version') != '1.0':
        raise ValueError('Invalid diagnosis result schema')
    outcome = mapping(mapping(result.get('outcomes')).get('diagnosis'))
    prediction = mapping(mapping(outcome.get('data')).get('diagnosis'))
    schema = prediction.get('label_schema')
    label, confidence = prediction.get('predicted_class'), prediction.get('confidence')
    classification = 'unavailable'
    if (outcome.get('status') in ('ok', 'degraded')
            and prediction.get('model_available') is True and prediction.get('abstained', False) is False
            and schema in ('faults-v1', 'bearing-health-v1') and type(label) is int
            and type(confidence) in (int, float) and 0 < confidence <= 1
            and label in class_labels(schema)):
        classification = 'normal' if label == 3 else 'fault'
    strength, cap = result.get('effective_strength'), result.get('max_allowed_strength')
    supported = (classification != 'unavailable' and outcome.get('status') == 'ok'
                 and result.get('ok') is True
                 and result.get('terminal_state') == 'COMPLETED'
                 and result.get('support_level') == 'high'
                 and strength in ('moderate', 'strong')
                 and (cap == 'strong' or cap == strength))
    return Alert(asset, result.get('run_id'), observed_at, available_at, classification, supported)


def inspection_from_dict(data):
    if (not isinstance(data, dict) or set(data) != {'assets', 'alerts'}
            or not isinstance(data['assets'], list) or not isinstance(data['alerts'], list)
            or len(data['assets']) > 32 or len(data['alerts']) > 256):
        raise ValueError('Invalid inspection plan fields')
    assets = []
    for row in data['assets']:
        if not isinstance(row, dict) or not isinstance(row.get('cell'), list):
            raise ValueError('Invalid inspection asset fields')
        assets.append(Asset(**{**row, 'cell': tuple(row['cell'])}))
    alerts = []
    for row in data['alerts']:
        if not isinstance(row, dict):
            raise ValueError('Invalid inspection alert fields')
        alerts.append(alert_from_result(**row))
    return InspectionPlan(tuple(assets), tuple(alerts))


class InspectionRun:
    def __init__(self, plan, scenario, alarm_policy):
        if any(not scenario.grid.free(a.cell) for a in plan.assets):
            raise ValueError('Inspection asset is outside the navigable grid')
        self.assets = {a.id: a for a in plan.assets}
        self.alarm_policy = alarm_policy
        self.horizon = scenario.horizon
        self.alerts = iter(sorted(plan.alerts, key=lambda a: a.available_at))
        self.next_alert = next(self.alerts, None)
        self.seen = set()
        self.active = {}
        self.generated = set()
        self.findings = Counter()
        self.states = {a.id: {'state': 'uninspected', 'task': None, 'confirmed_at': None}
                       for a in plan.assets}
        self.serial = 0

    def advance(self, tick, confirmed, tasks, emit):
        for asset, task in list(self.active.items()):
            if task in confirmed:
                del self.active[asset]
                finding = self.assets[asset].simulated_finding
                self.states[asset] = {'state': finding, 'task': task, 'confirmed_at': tick}
                self.findings[finding] += 1
                emit(tick, 'inspection_result', asset=asset, task=task, finding=finding)
        while self.next_alert is not None and self.next_alert.available_at == tick:
            alert = self.next_alert
            self.next_alert = next(self.alerts, None)
            self.accept(alert, tick, tasks, emit)

    def accept(self, alert, tick, tasks, emit):
        key = alert.asset, alert.run_id
        if key in self.seen:
            reason = 'duplicate'
        elif tick - alert.observed_at > 12:
            reason = 'stale'
        elif alert.classification != 'fault':
            reason = alert.classification
        elif self.alarm_policy == 'verified' and not alert.supported:
            reason = 'insufficient_support'
        elif alert.asset in self.active:
            reason = 'already_pending'
        elif tick >= self.horizon or len(tasks) >= 64:
            reason = 'mission_capacity'
        else:
            reason = None
        self.seen.add(key)
        if reason:
            emit(tick, 'inspection_alert_ignored', asset=alert.asset, run_id=alert.run_id, reason=reason)
            return
        asset = self.assets[alert.asset]
        while True:
            self.serial += 1
            task = 'recheck-' + str(self.serial)
            if task not in tasks:
                break
        tasks[task] = TaskSpec(task, asset.cell, tick, min(tick + asset.window, self.horizon),
                               asset.service, asset.priority)
        self.generated.add(task)
        self.active[asset.id] = task
        self.states[asset.id] = {'state': 'awaiting_recheck', 'task': task, 'confirmed_at': None}
        emit(tick, 'inspection_requested', asset=asset.id, task=task,
             run_id=alert.run_id, priority=asset.priority, deadline=tasks[task].deadline)

    def snapshot(self):
        return [{'asset': asset, **state} for asset, state in self.states.items()]
