"""Observation acceptance before each scheduling decision."""
from collections import Counter


class EvidenceInbox:
    def __init__(self, scenario, policy='verified', max_age=8):
        if policy not in ('direct', 'verified', 'oracle') or type(max_age) is not int or not 0 <= max_age <= 1000:
            raise ValueError('Invalid patrol evidence policy')
        self.scenario, self.policy, self.max_age = scenario, policy, max_age
        self.tasks = {t.id: t for t in scenario.tasks}
        self.origins = {r.id for r in scenario.robots}
        self.seen, self.by_task, self.direct, self.tainted = {}, {}, {}, set()
        self.rejections = Counter()

    def receive(self, obs, now):
        reason = None
        if not obs.valid() or obs.task not in self.tasks or obs.available_at > now:
            reason = 'invalid_message'
        elif self.policy == 'verified':
            if obs.mission != self.scenario.mission or obs.revision != self.tasks[obs.task].revision:
                reason = 'wrong_mission_or_revision'
            elif obs.origin not in self.origins:
                reason = 'unknown_origin'
            elif (obs.observed_at > now or obs.available_at < obs.observed_at
                  or obs.observed_at < self.tasks[obs.task].release
                  or (obs.state == 'pending' and now - obs.observed_at > self.max_age)):
                reason = 'stale_or_future'
            else:
                identity = obs.mission, obs.origin, obs.record
                payload = obs.task, obs.revision, obs.state, obs.observed_at
                if identity in self.tainted:
                    reason = 'identity_conflict'
                elif identity in self.seen:
                    reason = 'duplicate_record' if self.seen[identity] == payload else 'identity_conflict'
                    if reason == 'identity_conflict':
                        self.tainted.add(identity)
                        for evidence in self.by_task.values():
                            evidence.pop((obs.origin, obs.record), None)
                else:
                    self.seen[identity] = payload
        if reason:
            self.rejections[reason] += 1
            return reason
        if self.policy != 'verified':
            self.direct[obs.task] = obs.state
        else:
            origins = self.by_task.setdefault(obs.task, {})
            prior = [o for o in origins.values() if o.origin == obs.origin]
            if prior and obs.observed_at < max(o.observed_at for o in prior):
                self.rejections['older_observation'] += 1
                return 'older_observation'
            origins[(obs.origin, obs.record)] = obs
        return None

    def status(self, task, now):
        if self.policy != 'verified':
            return self.direct.get(task, 'unknown')
        observations = list(self.by_task.get(task, {}).values())
        if not observations:
            return 'unknown'
        # Completion certificates remain valid for the current immutable task
        # revision after acceptance. Freshness is checked when received.
        latest = max(o.observed_at for o in observations)
        states = {o.state for o in observations if o.observed_at == latest}
        return states.pop() if len(states) == 1 else 'conflict'
