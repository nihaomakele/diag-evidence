"""Deterministic closed-loop patrol with a separate physical task ledger."""
from collections import Counter
from dataclasses import asdict, dataclass, field
import heapq
import json

from .evidence import EvidenceInbox
from .models import Observation
from .planning import astar, allocate


@dataclass
class Robot:
    id: str
    home: tuple
    cell: tuple
    battery: int
    task: str | None = None
    path: list = field(default_factory=list)
    service_left: int = 0
    down: bool = False
    blocked_ticks: int = 0


def simulate(scenario, *, policy='verified', allocator='matching', record=True):
    if allocator not in ('matching', 'greedy') or type(record) is not bool:
        raise ValueError('Invalid patrol simulation controls')
    inbox = EvidenceInbox(scenario, policy)
    robots = {r.id: Robot(r.id, r.home, r.home, r.battery) for r in scenario.robots}
    tasks = {t.id: t for t in scenario.tasks}
    # Truth is private to environment/metrics; only oracle reads it for planning.
    completed, visits = {}, Counter()
    interrupted, counters = set(), Counter()
    frames, events, pending = [], [], []
    serial = 0
    routes = {}

    def route(start, goal):
        key = start, goal
        if key not in routes:
            path = astar(scenario.grid, start, goal)
            routes[key] = None if path is None else tuple(path)
            counters['static_path_searches'] += 1
        return None if routes[key] is None else list(routes[key])

    def event(tick, kind, **details):
        counters[kind] += 1
        if record:
            events.append({'tick': tick, 'kind': kind, **details})

    def enqueue(obs):
        nonlocal serial
        heapq.heappush(pending, (obs.available_at, serial, obs)); serial += 1
        counters['message_payload_bytes'] += len(json.dumps(asdict(obs), separators=(',', ':'), allow_nan=False).encode())

    for obs in scenario.observations:
        enqueue(obs)

    def belief(task, tick):
        if policy == 'oracle':
            return 'completed' if task in completed else 'pending'
        return inbox.status(task, tick)

    frame_start = 0
    for tick in range(scenario.horizon + 1):
        for robot in robots.values():
            down = any(r == robot.id and start <= tick < end for r, start, end in scenario.outages)
            if down and not robot.down:
                if robot.task is not None:
                    interrupted.add(robot.task)
                    event(tick, 'assignment_interrupted', robot=robot.id, task=robot.task)
                robot.task, robot.path, robot.service_left = None, [], 0
                event(tick, 'robot_down', robot=robot.id)
            elif robot.down and not down:
                event(tick, 'robot_recovered', robot=robot.id)
            robot.down = down
        while pending and pending[0][0] <= tick:
            _, _, obs = heapq.heappop(pending)
            reason = inbox.receive(obs, tick)
            event(tick, 'report_rejected' if reason else 'report_received',
                  task=obs.task, origin=obs.origin, record=obs.record, reason=reason,
                  state=obs.state, observed_at=obs.observed_at)
        for robot in robots.values():
            if robot.task is not None and belief(robot.task, tick) == 'completed':
                event(tick, 'assignment_cancelled', robot=robot.id, task=robot.task)
                robot.task, robot.path, robot.service_left = None, [], 0

        if tick < scenario.horizon:
            reserved = {r.task for r in robots.values() if r.task is not None}
            ready = [t for t in scenario.tasks if t.release <= tick < t.deadline and
                     t.id not in reserved and belief(t.id, tick) != 'completed']
            idle = [r for r in robots.values() if not r.down and r.task is None]
            paths, utilities = {}, {}
            for robot in idle:
                for task in ready:
                    path = route(robot.cell, task.cell)
                    back = route(task.cell, robot.home)
                    if path is None or back is None:
                        continue
                    cost = len(path) + task.service
                    if tick + cost > task.deadline or cost + len(back) > robot.battery:
                        continue
                    key = robot.id, task.id
                    paths[key] = path
                    utilities[key] = 1000 * task.priority - 10 * len(path) - 2 * task.service
            assignments = allocate([r.id for r in idle], [t.id for t in ready], utilities, allocator)
            if idle and ready:
                counters['assignment_rounds'] += 1
            for rid, tid in assignments:
                robot, task = robots[rid], tasks[tid]
                robot.task, robot.path, robot.service_left = tid, paths[(rid, tid)], task.service
                robot.blocked_ticks = 0
                event(tick, 'assigned', robot=rid, task=tid, utility=utilities[(rid, tid)],
                      belief=belief(tid, tick), route_steps=len(robot.path))
                if tid in interrupted:
                    event(tick, 'reassigned_after_outage', robot=rid, task=tid)
                    interrupted.remove(tid)
            # Idle robots return to distinct parking cells, keeping work sites clear.
            for robot in robots.values():
                if not robot.down and robot.task is None and robot.cell != robot.home:
                    robot.path = route(robot.cell, robot.home) or []
        if record:
            frames.append({'tick': tick,
                'robots': [{'id': r.id, 'cell': list(r.cell), 'battery': r.battery,
                            'task': r.task, 'status': 'down' if r.down else 'working' if r.task else 'idle',
                            'path': [list(c) for c in r.path]} for r in robots.values()],
                'tasks': [{'id': t.id, 'cell': list(t.cell), 'release': t.release, 'deadline': t.deadline,
                           'priority': t.priority, 'completed_at': completed.get(t.id),
                           'belief': belief(t.id, tick), 'assigned_to': next((r.id for r in robots.values() if r.task == t.id), None)}
                          for t in scenario.tasks], 'events': events[frame_start:]})
            frame_start = len(events)
        if tick == scenario.horizon:
            break

        old_positions = {rid: r.cell for rid, r in robots.items()}
        destinations, proposed = {}, {}
        for rid in sorted(robots):
            robot = robots[rid]
            proposed[rid] = robot.path[0] if not robot.down and robot.battery > 0 and robot.path else robot.cell
        # Conservatively reserve current cells; permit a following move only when
        # its occupant has already committed to vacate, and prohibit edge swaps.
        occupied = {r.cell: rid for rid, r in robots.items()}
        order = sorted(robots)
        order = order[tick % len(order):] + order[:tick % len(order)]
        for rid in order:
            robot, nxt = robots[rid], proposed[rid]
            occupant = occupied.get(nxt)
            allowed = (nxt not in destinations.values() and
                       (occupant is None or occupant == rid or
                        (occupant in destinations and destinations[occupant] != nxt
                         and destinations[occupant] != robot.cell)))
            destinations[rid] = nxt if allowed else robot.cell
        # A rejected following move can invalidate an earlier intention; current
        # cell ownership above makes all retained moves safe by construction.
        if len(set(destinations.values())) != len(robots):
            raise RuntimeError('Patrol vertex collision')
        for a in robots:
            for b in robots:
                if a < b and destinations[a] == old_positions[b] and destinations[b] == old_positions[a]:
                    raise RuntimeError('Patrol edge collision')
        for rid, robot in robots.items():
            nxt = destinations[rid]
            if nxt != robot.cell:
                if not scenario.grid.free(nxt) or abs(nxt[0] - robot.cell[0]) + abs(nxt[1] - robot.cell[1]) != 1:
                    raise RuntimeError('Invalid patrol movement')
                robot.cell = nxt; robot.path.pop(0); robot.battery -= 1
                robot.blocked_ticks = 0; counters['travel_steps'] += 1
            elif robot.path and not robot.down:
                robot.blocked_ticks += 1; counters['traffic_waits'] += 1
                if robot.blocked_ticks >= 2:
                    goal = tasks[robot.task].cell if robot.task else robot.home
                    others = frozenset(r.cell for r in robots.values() if r.id != rid)
                    replacement = astar(scenario.grid, robot.cell, goal, others)
                    counters['traffic_replans'] += 1
                    if replacement is not None:
                        back = route(goal, robot.home)
                        remaining = robot.service_left if robot.task else 0
                        if back is not None and len(replacement) + remaining + len(back) <= robot.battery:
                            robot.path = replacement
                    robot.blocked_ticks = 0
            elif (robot.task is not None and not robot.down and robot.cell == tasks[robot.task].cell
                    and robot.battery > 0):
                robot.service_left -= 1; robot.battery -= 1
                if robot.service_left == 0:
                    tid = robot.task
                    visits[tid] += 1
                    if tid not in completed:
                        completed[tid] = tick + 1
                        event(tick + 1, 'task_completed', robot=rid, task=tid)
                    else:
                        event(tick + 1, 'task_reinspected', robot=rid, task=tid)
                    obs = Observation(scenario.mission, tid, tasks[tid].revision, 'completed',
                        tick + 1, tick + 1 + scenario.report_delay, rid, 'visit-' + str(tick + 1) + '-' + tid)
                    enqueue(obs)
                    robot.task, robot.path, robot.service_left = None, [], 0
            if robot.battery < 0:
                raise RuntimeError('Negative robot battery')
    on_time = [t for t in scenario.tasks if t.id in completed and completed[t.id] <= t.deadline]
    metrics = {'tasks': len(tasks), 'on_time_completed': len(on_time),
        'on_time_completion_rate': len(on_time) / len(tasks),
        'weighted_on_time_rate': sum(t.priority for t in on_time) / sum(t.priority for t in scenario.tasks),
        'missed_tasks': len(tasks) - len(on_time), 'travel_steps': counters['travel_steps'],
        'reinspections': counters['task_reinspected'],
        'reassignments_after_outage': counters['reassigned_after_outage'],
        'assignment_cancellations': counters['assignment_cancelled'],
        'traffic_waits': counters['traffic_waits'], 'collision_count': 0,
        'reports_sent': serial, 'reports_pending': len(pending),
        'message_payload_bytes': counters['message_payload_bytes'],
        'energy_used': sum(r.battery for r in scenario.robots) - sum(r.battery for r in robots.values()),
        'assignment_rounds': counters['assignment_rounds'],
        'static_path_searches': counters['static_path_searches'], 'traffic_replans': counters['traffic_replans'],
        'completion_makespan': max(completed.values()) if len(completed) == len(tasks) else None,
        'reports_received': counters['report_received'] + counters['report_rejected'],
        'message_rejections': dict(inbox.rejections),
        'completion_times': dict(sorted(completed.items()))}
    return {'scenario': scenario.name, 'mission': scenario.mission, 'policy': policy, 'allocator': allocator,
            'grid': {'width': scenario.grid.width, 'height': scenario.grid.height,
                     'blocked': [list(c) for c in sorted(scenario.grid.blocked)]},
            'metrics': metrics, 'frames': frames, 'events': events}
