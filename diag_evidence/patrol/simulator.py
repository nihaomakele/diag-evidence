"""Deterministic closed-loop patrol with a separate physical task ledger."""
from collections import Counter
from dataclasses import asdict, dataclass, field
import heapq
import json

from .evidence import EvidenceInbox
from .execution import ExecutionConfig, ExecutionProtocol
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


def simulate(scenario, *, policy='verified', allocator='matching', record=True,
             execution=None, drop_completion_reports=False, traffic='legacy'):
    if (allocator not in ('matching', 'greedy') or type(record) is not bool
            or (execution is not None and not isinstance(execution, ExecutionConfig))
            or type(drop_completion_reports) is not bool
            or traffic not in ('legacy', 'yield')):
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

    protocol = (ExecutionProtocol(execution, scenario.report_delay, drop_completion_reports, event)
                if execution is not None else None)

    def enqueue(obs):
        nonlocal serial
        heapq.heappush(pending, (obs.available_at, serial, obs)); serial += 1
        counters['message_payload_bytes'] += len(json.dumps(asdict(obs), separators=(',', ':'), allow_nan=False).encode())

    for obs in scenario.observations:
        enqueue(obs)

    def belief(task, tick):
        if policy == 'oracle':
            return 'completed' if task in completed else 'pending'
        if protocol is not None and task in protocol.confirmed:
            return 'completed'
        return inbox.status(task, tick)

    def clear_assignment(robot):
        robot.task, robot.path, robot.service_left = None, [], 0

    def detour(robot, occupied):
        goal = tasks[robot.task].cell if robot.task else robot.home
        replacement = astar(scenario.grid, robot.cell, goal, occupied)
        counters['traffic_replans'] += 1
        if replacement is None:
            return None
        back = route(goal, robot.home)
        remaining = robot.service_left if robot.task else 0
        if back is not None and len(replacement) + remaining + len(back) <= robot.battery:
            return replacement
        return None

    def traffic_priority(rid):
        task = tasks.get(robots[rid].task)
        if task is None:
            return True, 0, 0, rid
        return False, task.deadline, -task.priority, rid

    frame_start = 0
    for tick in range(scenario.horizon + 1):
        for robot in robots.values():
            down = any(r == robot.id and start <= tick < end for r, start, end in scenario.outages)
            if down and not robot.down:
                if robot.task is not None:
                    interrupted.add(robot.task)
                    event(tick, 'assignment_interrupted', robot=robot.id, task=robot.task)
                    if protocol is not None:
                        protocol.stop(robot.task, tick, 'outage')
                clear_assignment(robot)
                event(tick, 'robot_down', robot=robot.id)
            elif robot.down and not down:
                event(tick, 'robot_recovered', robot=robot.id)
            robot.down = down
        if protocol is not None:
            offline = {r.id for r in robots.values() if r.down}
            for dispatch in protocol.advance(tick, offline):
                robot = robots[dispatch.robot]
                if robot.task == dispatch.task:
                    clear_assignment(robot)
        while pending and pending[0][0] <= tick:
            _, _, obs = heapq.heappop(pending)
            reason = inbox.receive(obs, tick)
            event(tick, 'report_rejected' if reason else 'report_received',
                  task=obs.task, origin=obs.origin, record=obs.record, reason=reason,
                  state=obs.state, observed_at=obs.observed_at)
        for robot in robots.values():
            if robot.task is not None and belief(robot.task, tick) == 'completed':
                event(tick, 'assignment_cancelled', robot=robot.id, task=robot.task)
                if protocol is not None:
                    protocol.stop(robot.task, tick, 'observation')
                clear_assignment(robot)

        if tick < scenario.horizon:
            reserved = {r.task for r in robots.values() if r.task is not None}
            if protocol is not None:
                reserved.update(protocol.leases)
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
                path = paths[(rid, tid)]
                execution_details = {}
                if protocol is not None:
                    expected_finish = tick + len(path) + task.service
                    dispatch = protocol.start(rid, tid, tick, expected_finish)
                    execution_details['attempt'] = dispatch.attempt
                robot.task, robot.path, robot.service_left = tid, path, task.service
                robot.blocked_ticks = 0
                event(tick, 'assigned', robot=rid, task=tid, utility=utilities[(rid, tid)],
                      belief=belief(tid, tick), route_steps=len(robot.path), **execution_details)
                if tid in interrupted:
                    event(tick, 'reassigned_after_outage', robot=rid, task=tid)
                    interrupted.remove(tid)
            # Idle robots return to distinct parking cells, keeping work sites clear.
            for robot in robots.values():
                if (not robot.down and robot.task is None and robot.cell != robot.home
                        and (traffic == 'legacy' or not robot.path)):
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
            if protocol is not None:
                frames[-1]['execution_leases'] = {
                    task: {'attempt': lease.dispatch.attempt, 'robot': lease.dispatch.robot,
                           'query_pending': lease.probing}
                    for task, lease in protocol.leases.items()
                }
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
        yielding, holding = {}, set()
        if traffic == 'yield':
            for rid in sorted(robots):
                other = occupied.get(proposed[rid])
                if other is None or other <= rid or proposed[other] != old_positions[rid]:
                    continue
                if max(robots[rid].blocked_ticks, robots[other].blocked_ticks) == 0:
                    continue
                # Keep the original reservation rules and replace at most one
                # route for a blocked pair. Retry after another wait if no
                # energy-feasible detour exists from these positions.
                holding.update((rid, other))
                first, second = sorted((rid, other), key=traffic_priority)
                for yielder, keeper in ((second, first), (first, second)):
                    alternative = detour(robots[yielder], occupied.keys() - {old_positions[yielder]})
                    if alternative:
                        yielding[yielder] = (keeper, alternative)
                        break
        for rid, robot in robots.items():
            nxt = destinations[rid]
            if nxt != robot.cell:
                if not scenario.grid.free(nxt) or abs(nxt[0] - robot.cell[0]) + abs(nxt[1] - robot.cell[1]) != 1:
                    raise RuntimeError('Invalid patrol movement')
                robot.cell = nxt; robot.path.pop(0); robot.battery -= 1
                robot.blocked_ticks = 0; counters['travel_steps'] += 1
            elif robot.path and not robot.down:
                robot.blocked_ticks += 1; counters['traffic_waits'] += 1
                if rid in yielding:
                    keeper, robot.path = yielding[rid]
                    robot.blocked_ticks = 0
                    event(tick + 1, 'traffic_yield', robot=rid, keeper=keeper,
                          task=robot.task, route_steps=len(robot.path))
                elif rid in holding:
                    robot.blocked_ticks = 0
                elif robot.blocked_ticks >= 2:
                    others = frozenset(r.cell for r in robots.values() if r.id != rid)
                    replacement = detour(robot, others)
                    if replacement is not None:
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
                    if protocol is not None:
                        protocol.finish(tid, rid, tick + 1)
                    elif drop_completion_reports:
                        event(tick + 1, 'completion_report_dropped', robot=rid, task=tid)
                    else:
                        obs = Observation(scenario.mission, tid, tasks[tid].revision, 'completed',
                            tick + 1, tick + 1 + scenario.report_delay, rid, 'visit-' + str(tick + 1) + '-' + tid)
                        enqueue(obs)
                    clear_assignment(robot)
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
    if drop_completion_reports:
        metrics['completion_reports_dropped'] = (counters['completion_report_dropped']
                                                 + counters['execution_result_dropped'])
    if traffic == 'yield':
        metrics['traffic_yields'] = counters['traffic_yield']
    if protocol is not None:
        lags = protocol.confirmation_lags
        metrics.update(
            execution_confirmed=len(lags),
            execution_unresolved=len(protocol.leases),
            execution_unconfirmed_completed=len(completed.keys() - protocol.confirmed.keys()),
            execution_queries=counters['execution_queried'],
            execution_timeouts=counters['execution_timed_out'],
            reassignments_after_timeout=counters['execution_reassigned_after_timeout'],
            duplicate_dispatches_ignored=counters['dispatch_duplicate'],
            stale_replies_ignored=counters['execution_stale_reply'],
            duplicate_replies_ignored=counters['execution_duplicate_reply'],
            execution_messages=protocol.messages,
            execution_payload_bytes=protocol.payload_bytes,
            mean_confirmation_lag=sum(lags) / len(lags) if lags else None,
        )
    return {'scenario': scenario.name, 'mission': scenario.mission, 'policy': policy, 'allocator': allocator,
            'grid': {'width': scenario.grid.width, 'height': scenario.grid.height,
                     'blocked': [list(c) for c in sorted(scenario.grid.blocked)]},
            'metrics': metrics, 'frames': frames, 'events': events}
