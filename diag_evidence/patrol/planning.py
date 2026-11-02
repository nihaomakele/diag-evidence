"""Single-robot A* and exact bounded one-step assignment with idle actions."""
import heapq
import math


def astar(grid, start, goal, occupied=frozenset()):
    if not grid.free(start) or not grid.free(goal):
        return None
    if start == goal:
        return []
    blocked = grid.blocked | frozenset(occupied) - {start}
    if goal in blocked:
        return None
    heap = [(abs(goal[0] - start[0]) + abs(goal[1] - start[1]), 0, start)]
    cost, previous = {start: 0}, {}
    while heap:
        _, g, cell = heapq.heappop(heap)
        if g != cost[cell]:
            continue
        if cell == goal:
            path = []
            while cell != start:
                path.append(cell); cell = previous[cell]
            return list(reversed(path))
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nxt = cell[0] + dx, cell[1] + dy
            if grid.contains(nxt) and nxt not in blocked and g + 1 < cost.get(nxt, math.inf):
                cost[nxt], previous[nxt] = g + 1, cell
                h = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(heap, (g + 1 + h, g + 1, nxt))
    return None


def allocate(robot_ids, task_ids, utilities, method='matching'):
    """Maximum additive one-step utility, not a full-horizon route optimum."""
    robots, tasks = sorted(robot_ids), sorted(task_ids)
    if (not 0 <= len(robots) <= 8 or len(tasks) > 64 or len(set(robots)) != len(robots)
            or len(set(tasks)) != len(tasks) or method not in ('matching', 'greedy')):
        raise ValueError('Invalid bounded allocation')
    if any(r not in robots or t not in tasks or isinstance(v, bool) or
           not isinstance(v, (int, float)) or not math.isfinite(v) for (r, t), v in utilities.items()):
        raise ValueError('Invalid allocation utility')
    if method == 'greedy':
        result, used_r, used_t = [], set(), set()
        for (r, t), value in sorted(utilities.items(), key=lambda x: (-x[1], x[0])):
            if value > 0 and r not in used_r and t not in used_t:
                result.append((r, t)); used_r.add(r); used_t.add(t)
        return sorted(result)
    states = {0: (0, ())}
    for task in tasks:
        updated = dict(states)
        for mask, (total, pairs) in states.items():
            for i, robot in enumerate(robots):
                value = utilities.get((robot, task), 0)
                if mask & (1 << i) or value <= 0:
                    continue
                nxt = mask | (1 << i)
                candidate = total + value, tuple(sorted(pairs + ((robot, task),)))
                old = updated.get(nxt)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    updated[nxt] = candidate
        states = updated
    return list(min(states.values(), key=lambda item: (-item[0], -len(item[1]), item[1]))[1])
