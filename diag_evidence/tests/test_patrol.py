from collections import deque
from dataclasses import replace
import itertools
import json
import math
import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from diag_evidence.patrol.models import Grid, RobotSpec, TaskSpec, Observation, Scenario, scenario_from_dict, scenario_to_dict
from diag_evidence.patrol.planning import astar, allocate
from diag_evidence.patrol.evidence import EvidenceInbox
from diag_evidence.patrol.scenarios import make_scenario, SCENARIOS
from diag_evidence.patrol.simulator import simulate
from diag_evidence.patrol.demo import run_demo
from diag_evidence.patrol.evaluation import evaluate_suite, freeze_experiment, source_hashes


def tiny(**changes):
    return replace(Scenario('tiny', 'mission', Grid(8, 4), (RobotSpec('R0', (0, 0), 30),),
        (TaskSpec('T0', (2, 0), 0, 15, 2),), horizon=15), **changes)


def observation(**changes):
    return replace(Observation('mission', 'T0', 1, 'completed', 2, 2, 'R0', 'source-record'), **changes)


class PlanningTests(unittest.TestCase):
    def test_astar_matches_independent_breadth_first_search(self):
        grid = Grid(7, 6, frozenset({(3, y) for y in range(5)}))
        start, goal = (0, 0), (6, 0)
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            cell, distance = queue.popleft()
            if cell == goal:
                break
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = cell[0] + dx, cell[1] + dy
                if grid.free(nxt) and nxt not in seen:
                    queue.append((nxt, distance + 1)); seen.add(nxt)
        path = astar(grid, start, goal)
        self.assertEqual(len(path), distance)
        self.assertEqual(path[-1], goal)
        self.assertTrue(all(grid.free(c) for c in path))

    def test_unreachable_and_stationary_paths(self):
        grid = Grid(3, 3, frozenset({(1, y) for y in range(3)}))
        self.assertIsNone(astar(grid, (0, 0), (2, 0)))
        self.assertEqual(astar(grid, (0, 0), (0, 0)), [])

    def test_dynamic_occupied_cells(self):
        self.assertIsNone(astar(Grid(3, 2), (0, 0), (2, 0), {(2, 0)}))
        self.assertNotIn((1, 0), astar(Grid(3, 2), (0, 0), (2, 0), {(1, 0)}))

    def test_matching_improves_known_greedy_counterexample(self):
        edges = {('A', 'X'): 9, ('A', 'Y'): 8, ('B', 'X'): 8}
        self.assertEqual(allocate(['A', 'B'], ['X', 'Y'], edges), [('A', 'Y'), ('B', 'X')])
        self.assertEqual(allocate(['A', 'B'], ['X', 'Y'], edges, 'greedy'), [('A', 'X')])

    def test_matching_matches_bruteforce(self):
        rng = random.Random(3)
        for _ in range(20):
            robots, tasks = ['A', 'B', 'C'], ['X', 'Y', 'Z', 'W']
            values = {(r, t): rng.randrange(-2, 10) for r in robots for t in tasks}
            optimum = 0
            for choices in itertools.product([None] + tasks, repeat=3):
                chosen = [t for t in choices if t is not None]
                if len(chosen) == len(set(chosen)):
                    optimum = max(optimum, sum(values[(r, t)] for r, t in zip(robots, choices) if t is not None))
            pairs = allocate(robots, tasks, values)
            self.assertEqual(sum(values[p] for p in pairs), optimum)

    def test_matching_ties_and_idle_are_deterministic(self):
        self.assertEqual(allocate(['B', 'A'], ['X'], {('A', 'X'): 1, ('B', 'X'): 1}), [('A', 'X')])
        self.assertEqual(allocate(['A'], ['X'], {('A', 'X'): -1}), [])
        self.assertEqual(allocate([], [], {}), [])

    def test_invalid_allocation_inputs(self):
        for edges in [{('A', 'X'): math.nan}, {('B', 'X'): 1}, {('A', 'X'): True}]:
            with self.assertRaises(ValueError):
                allocate(['A'], ['X'], edges)
        with self.assertRaises(ValueError):
            allocate([str(i) for i in range(9)], [], {})


class ObservationTests(unittest.TestCase):
    def test_old_mission_changes_direct_but_verified_rejects(self):
        old = observation(mission='previous', revision=0)
        direct, checked = EvidenceInbox(tiny(), 'direct'), EvidenceInbox(tiny())
        self.assertIsNone(direct.receive(old, 2))
        self.assertEqual(direct.status('T0', 2), 'completed')
        self.assertEqual(checked.receive(old, 2), 'wrong_mission_or_revision')
        self.assertEqual(checked.status('T0', 2), 'unknown')

    def test_completion_observation_survives_delivery_delay(self):
        inbox = EvidenceInbox(tiny())
        self.assertIsNone(inbox.receive(observation(available_at=14), 14))
        self.assertEqual(inbox.status('T0', 15), 'completed')

    def test_pending_freshness_and_future_observations(self):
        inbox = EvidenceInbox(tiny())
        self.assertEqual(inbox.receive(observation(state='pending', available_at=14), 14), 'stale_or_future')
        self.assertEqual(inbox.receive(observation(observed_at=5, available_at=2), 2), 'stale_or_future')

    def test_unknown_origin_and_unavailable_message(self):
        inbox = EvidenceInbox(tiny())
        self.assertEqual(inbox.receive(observation(origin='other'), 2), 'unknown_origin')
        self.assertEqual(inbox.receive(observation(available_at=3), 2), 'invalid_message')

    def test_forwarding_preserves_record_identity(self):
        inbox = EvidenceInbox(tiny())
        inbox.receive(observation(), 2)
        self.assertEqual(inbox.receive(observation(relay='R1', available_at=3), 3), 'duplicate_record')

    def test_conflicting_identity_revokes_prior_record(self):
        inbox = EvidenceInbox(tiny())
        inbox.receive(observation(), 2)
        self.assertEqual(inbox.receive(observation(state='pending'), 2), 'identity_conflict')
        self.assertEqual(inbox.status('T0', 2), 'unknown')
        self.assertEqual(inbox.receive(observation(), 3), 'identity_conflict')

    def test_same_time_conflict_and_later_resolution(self):
        scenario = replace(tiny(), robots=(RobotSpec('R0', (0, 0)), RobotSpec('R1', (0, 1))))
        inbox = EvidenceInbox(scenario)
        inbox.receive(observation(), 2)
        inbox.receive(observation(origin='R1', record='other', state='pending'), 2)
        self.assertEqual(inbox.status('T0', 2), 'conflict')
        inbox.receive(observation(observed_at=3, available_at=3, record='new'), 3)
        self.assertEqual(inbox.status('T0', 3), 'completed')

    def test_old_arriving_observation_does_not_reverse_newer(self):
        inbox = EvidenceInbox(tiny())
        inbox.receive(observation(observed_at=4, available_at=4), 4)
        self.assertEqual(inbox.receive(observation(observed_at=3, available_at=5, record='old', state='pending'), 5), 'older_observation')
        self.assertEqual(inbox.status('T0', 5), 'completed')

    def test_valid_fresh_error_is_not_given_hidden_truth(self):
        inbox = EvidenceInbox(tiny())
        self.assertIsNone(inbox.receive(observation(), 2))
        self.assertEqual(inbox.status('T0', 2), 'completed')


class SimulationTests(unittest.TestCase):
    def test_configuration_roundtrip_and_validation(self):
        scenario = make_scenario('mixed', 7)
        self.assertEqual(scenario_from_dict(json.loads(json.dumps(scenario_to_dict(scenario)))), scenario)
        with self.assertRaises(ValueError):
            Grid(True, 3)
        with self.assertRaises(ValueError):
            replace(tiny(), robots=(RobotSpec('R0', (0, 0)), RobotSpec('R0', (1, 0))))
        with self.assertRaises(ValueError):
            scenario_from_dict({'extra': 1})

    def test_hand_calculated_completion_and_energy(self):
        run = simulate(tiny())
        self.assertEqual(run['metrics']['completion_times'], {'T0': 4})
        self.assertEqual(run['metrics']['completion_makespan'], 4)
        self.assertEqual(run['metrics']['energy_used'], 6)
        self.assertEqual(run['metrics']['travel_steps'], 4)

    def test_release_and_battery_constraints(self):
        scenario = replace(tiny(), tasks=(TaskSpec('T0', (2, 0), 3, 15, 2),))
        run = simulate(scenario)
        self.assertEqual(run['metrics']['completion_times'], {'T0': 7})
        low = replace(tiny(), robots=(RobotSpec('R0', (0, 0), 5),))
        self.assertEqual(simulate(low)['metrics']['on_time_completed'], 0)

    def test_impossible_deadline_excludes_assignment(self):
        scenario = replace(tiny(), tasks=(TaskSpec('T0', (6, 0), 0, 3, 2),))
        run = simulate(scenario)
        self.assertFalse(any(e['kind'] == 'assigned' for e in run['events']))

    def test_outage_releases_and_reassigns_task(self):
        scenario = replace(tiny(), tasks=(TaskSpec('T0', (5, 0), 0, 15, 2),), outages=(('R0', 2, 5),))
        run = simulate(scenario)
        self.assertEqual(run['metrics']['on_time_completed'], 1)
        self.assertEqual(run['metrics']['reassignments_after_outage'], 1)

    def test_validation_changes_dispatch_before_mission_ends(self):
        scenario = make_scenario('mixed', 7)
        direct, checked = simulate(scenario, policy='direct'), simulate(scenario)
        self.assertEqual(direct['frames'][0]['tasks'][1]['belief'], 'completed')
        self.assertEqual(checked['frames'][0]['tasks'][1]['belief'], 'conflict')
        self.assertGreater(checked['metrics']['on_time_completed'], direct['metrics']['on_time_completed'])
        self.assertTrue(any(e['kind'] == 'assigned' and e['tick'] < 10 for e in checked['events']))

    def test_clean_policies_have_identical_physical_metrics(self):
        scenario = make_scenario('clean', 3)
        self.assertEqual(simulate(scenario, policy='direct')['metrics'], simulate(scenario)['metrics'])

    def test_fresh_error_negative_control(self):
        scenario = make_scenario('fresh_error', 7)
        direct = simulate(scenario, policy='direct')['metrics']['on_time_completed']
        checked = simulate(scenario)['metrics']['on_time_completed']
        oracle = simulate(scenario, policy='oracle')['metrics']['on_time_completed']
        self.assertEqual(checked, direct)
        self.assertGreater(oracle, checked)

    def test_collision_free_budgeted_execution_across_development_seeds(self):
        for seed in range(4):
            for name in SCENARIOS:
                run = simulate(make_scenario(name, seed))
                self.assertEqual(run['metrics']['collision_count'], 0)
                for frame in run['frames']:
                    self.assertEqual(len({tuple(r['cell']) for r in frame['robots']}), 3)
                    self.assertTrue(all(r['battery'] >= 0 for r in frame['robots']))
                    active = [r['task'] for r in frame['robots'] if r['task'] is not None]
                    self.assertEqual(len(active), len(set(active)))

    def test_trace_events_match_frame_time(self):
        run = simulate(tiny())
        flattened = []
        for frame in run['frames']:
            self.assertTrue(all(e['tick'] == frame['tick'] for e in frame['events']))
            flattened.extend(frame['events'])
        self.assertEqual(flattened, run['events'])

    def test_seeded_replay_is_deterministic(self):
        scenario = make_scenario('mixed', 7)
        self.assertEqual(simulate(scenario), simulate(scenario))
        self.assertEqual(simulate(scenario, record=False)['metrics'], simulate(scenario)['metrics'])


class DeliveryTests(unittest.TestCase):
    def test_demo_no_write_has_no_filesystem_side_effects(self):
        with mock.patch.object(Path, 'mkdir', side_effect=AssertionError('unexpected output')):
            result = run_demo(tiny(), write=False)
        self.assertEqual(result['metrics']['verified']['on_time_completed'], 1)
        self.assertNotIn('output_dir', result)

    def test_replay_embeds_reproducible_data_and_offline_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_demo(tiny(), directory)
            path = Path(result['replay_path'])
            text = path.read_text()
            payload = text.split('<script id="patrol-data" type="application/json">', 1)[1].split('</script>', 1)[0]
            embedded = json.loads(payload)
            self.assertEqual(embedded['runs']['verified'], simulate(tiny()))
            self.assertNotIn('__PATROL_DATA__', text)
            self.assertNotIn('<script src=', text)
            self.assertIn('id="scrub"', text)
            self.assertIn('id="play"', text)
            self.assertEqual(scenario_from_dict(json.loads((path.parent / 'scenario.json').read_text())), tiny())

    def test_paired_suite_reuses_same_exogenous_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = evaluate_suite(root / 'dev', seeds=(0,))
            rows = json.loads((root / 'dev/episodes.json').read_text())
            self.assertEqual(summary['episodes'], 48)
            self.assertEqual(summary['collision_count'], 0)
            for name in SCENARIOS:
                self.assertEqual(len({r['scenario_sha256'] for r in rows if r['scenario'] == name}), 1)

    def test_held_out_requires_matching_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = freeze_experiment(root / 'freeze.json')
            self.assertEqual(frozen['code_sha256'], source_hashes())
            frozen['code_sha256']['models.py'] = '0' * 64
            (root / 'freeze.json').write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):
                evaluate_suite(root / 'held', split='held_out', freeze_path=root / 'freeze.json')
            self.assertFalse((root / 'held').exists())
            with self.assertRaises(ValueError):
                evaluate_suite(root / 'other', split='held_out')

    def test_freeze_checks_development_source_and_seed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_experiment(Path(directory) / 'freeze.json', {'split': 'development', 'source_sha256': {}})
