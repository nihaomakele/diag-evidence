"""Dispatch leases and executor replies for the patrol simulator."""

from dataclasses import asdict, dataclass
import heapq
import json

from .models import integer


@dataclass(frozen=True)
class ExecutionConfig:
    confirmation_timeout: int = 4
    probe_timeout: int = 16
    repeat_dispatch: bool = False

    def __post_init__(self):
        if (not integer(self.confirmation_timeout, 1, 100)
                or not integer(self.probe_timeout, 1, 200)
                or type(self.repeat_dispatch) is not bool):
            raise ValueError('Invalid patrol execution controls')


@dataclass(frozen=True)
class Dispatch:
    attempt: int
    robot: str
    task: str


@dataclass(frozen=True)
class Reply:
    dispatch: Dispatch
    state: str
    observed_at: int


@dataclass
class Lease:
    dispatch: Dispatch
    timer: int
    probing: bool = False


class ExecutionProtocol:
    """The scheduler receives worker status only through the modeled link.

    A dispatch ACK and a stop are synchronous in this simulator. Probes take
    one tick to reach the worker, and replies share the completion-link delay.
    A worker caches its outcome for the episode, keyed by dispatch attempt.
    """

    def __init__(self, config, report_delay, drop_push, event):
        self.config = config
        self.report_delay = report_delay
        self.drop_push = drop_push
        self.event = event
        self.leases = {}
        self.confirmed = {}
        self.timed_out = set()
        self.messages = 0
        self.payload_bytes = 0
        self.confirmation_lags = []
        self._worker = {}
        self._queue = []
        self._attempt = 0
        self._serial = 0

    def _emit(self, now, kind, dispatch, **details):
        self.event(now, kind, robot=dispatch.robot, task=dispatch.task,
                   attempt=dispatch.attempt, **details)

    def _packet(self, kind, dispatch, **fields):
        self.messages += 1
        payload = {'kind': kind, **asdict(dispatch), **fields}
        self.payload_bytes += len(json.dumps(payload, separators=(',', ':')).encode('utf-8'))

    def _schedule(self, at, kind, value):
        self._serial += 1
        heapq.heappush(self._queue, (at, self._serial, kind, value))

    def _deliver_dispatch(self, dispatch, now):
        self._packet('dispatch', dispatch)
        duplicate = dispatch in self._worker
        if not duplicate:
            self._worker[dispatch] = Reply(dispatch, 'running', now)
        self._packet('ack', dispatch, duplicate=duplicate)
        self._emit(now, 'dispatch_duplicate' if duplicate else 'dispatch_acknowledged', dispatch)
        return not duplicate

    def start(self, robot, task, now, expected_finish):
        if task in self.leases:
            raise RuntimeError('Task already has an active dispatch')
        self._attempt += 1
        dispatch = Dispatch(self._attempt, robot, task)
        self.leases[task] = Lease(dispatch, expected_finish + self.config.confirmation_timeout)
        if not self._deliver_dispatch(dispatch, now):
            raise RuntimeError('New dispatch already accepted')
        if self.config.repeat_dispatch:
            self._deliver_dispatch(dispatch, now)
        if task in self.timed_out:
            self.timed_out.remove(task)
            self._emit(now, 'execution_reassigned_after_timeout', dispatch)
        return dispatch

    def finish(self, task, robot, now):
        dispatch = self.leases[task].dispatch
        if dispatch.robot != robot or self._worker[dispatch].state != 'running':
            raise RuntimeError('Completion is not from the active executor')
        result = Reply(dispatch, 'succeeded', now)
        self._worker[dispatch] = result
        self._packet('result', dispatch, state=result.state, observed_at=now)
        if self.drop_push:
            self._emit(now, 'execution_result_dropped', dispatch)
        else:
            self._schedule(now + self.report_delay, 'reply', (result, 'push'))

    def stop(self, task, now, reason):
        dispatch = self.leases.pop(task).dispatch
        if self._worker[dispatch].state == 'running':
            outcome = 'failed' if reason == 'outage' else 'cancelled'
            self._worker[dispatch] = Reply(dispatch, outcome, now)
            if reason != 'outage':
                self._packet('cancel', dispatch)
                self._packet('cancel_ack', dispatch)
        self._emit(now, 'execution_closed', dispatch, reason=reason)
        return dispatch

    def _receive(self, reply, source, now):
        dispatch = reply.dispatch
        lease = self.leases.get(dispatch.task)
        if lease is None or lease.dispatch != dispatch:
            duplicate = reply.state == 'succeeded' and self.confirmed.get(dispatch.task) == dispatch
            self._emit(now, 'execution_duplicate_reply' if duplicate else 'execution_stale_reply',
                       dispatch, state=reply.state, source=source)
            return
        if reply.state == 'succeeded':
            del self.leases[dispatch.task]
            self.confirmed[dispatch.task] = dispatch
            self.confirmation_lags.append(now - reply.observed_at)
            self._emit(now, 'execution_confirmed', dispatch, source=source, finished_at=reply.observed_at)
        elif reply.state == 'running':
            lease.probing = False
            lease.timer = now + self.config.confirmation_timeout
            self._emit(now, 'execution_still_running', dispatch)
        else:
            raise RuntimeError('Terminal stop retained its scheduler lease')

    def advance(self, now, offline):
        while self._queue and self._queue[0][0] <= now:
            _, _, kind, value = heapq.heappop(self._queue)
            if kind == 'reply':
                reply, source = value
                self._receive(reply, source, now)
            else:
                dispatch = value
                if dispatch.robot not in offline:
                    reply = self._worker[dispatch]
                    self._packet('status_reply', dispatch, state=reply.state, observed_at=reply.observed_at)
                    self._schedule(now + self.report_delay, 'reply', (reply, 'probe'))

        expired = []
        for task, lease in list(self.leases.items()):
            if now < lease.timer:
                continue
            dispatch = lease.dispatch
            if lease.probing:
                expired.append(self.stop(task, now, 'probe_timeout'))
                self.timed_out.add(task)
                self._emit(now, 'execution_timed_out', dispatch)
            else:
                self._packet('status_query', dispatch)
                self._schedule(now + 1, 'probe', dispatch)
                self._emit(now, 'execution_queried', dispatch)
                lease.probing = True
                lease.timer = now + self.config.probe_timeout
        return expired
