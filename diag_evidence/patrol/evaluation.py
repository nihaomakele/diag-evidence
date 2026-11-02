"""Paired seeded suites with explicit source freeze before held-out seeds."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

from .models import scenario_to_dict
from .scenarios import SCENARIOS, make_scenario
from .simulator import simulate

POLICIES = ('direct', 'verified', 'oracle')
ALLOCATORS = ('greedy', 'matching')
CORE_FILES = ('__init__.py', 'models.py', 'planning.py', 'evidence.py', 'scenarios.py', 'simulator.py', 'evaluation.py')
SPEC = {'format': 'patrol-experiment/v1', 'scenarios': list(SCENARIOS),
        'policies': list(POLICIES), 'allocators': list(ALLOCATORS),
        'development_seeds': list(range(20)), 'held_out_seeds': list(range(20, 40)),
        'primary': 'verified minus direct with matching, paired by scenario and seed'}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in CORE_FILES}


def freeze_experiment(path, development_summary=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Patrol freeze already exists')
    if development_summary is not None and (development_summary.get('split') != 'development'
            or development_summary.get('source_sha256') != source_hashes()
            or development_summary.get('seeds') != SPEC['development_seeds']):
        raise ValueError('Development results differ from current source or full seed protocol')
    receipt = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'spec': SPEC, 'code_sha256': source_hashes(),
               'development_summary_sha256': digest(development_summary) if development_summary is not None else None}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, receipt)
    return receipt


def evaluate_suite(output_dir, *, split='development', freeze_path=None, seeds=None):
    if split not in ('development', 'held_out'):
        raise ValueError('Invalid patrol evaluation split')
    if split == 'held_out':
        if freeze_path is None or seeds is not None:
            raise ValueError('Held-out evaluation requires the fixed freeze and seed set')
        frozen = json.loads(Path(freeze_path).read_text())
        if frozen['spec'] != SPEC or frozen['code_sha256'] != source_hashes():
            raise ValueError('Patrol source or specification changed after freeze')
        selected_seeds = SPEC['held_out_seeds']
    else:
        selected_seeds = list(SPEC['development_seeds'] if seeds is None else seeds)
        if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds) or any(type(s) is not int or s not in SPEC['development_seeds'] for s in selected_seeds):
            raise ValueError('Invalid development seed selection')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {'started_at': datetime.now(timezone.utc).isoformat(),
        'split': split, 'spec': SPEC, 'seeds': selected_seeds, 'code_sha256': source_hashes(),
        'freeze_sha256': hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest() if freeze_path else None})
    episodes = []
    grouped = defaultdict(list)
    for name in SCENARIOS:
        for seed in selected_seeds:
            scenario = make_scenario(name, seed)
            input_hash = digest(scenario_to_dict(scenario))
            for allocator in ALLOCATORS:
                for policy in POLICIES:
                    metrics = simulate(scenario, policy=policy, allocator=allocator, record=False)['metrics']
                    row = {'scenario': name, 'seed': seed, 'allocator': allocator, 'policy': policy,
                           'scenario_sha256': input_hash, 'metrics': metrics}
                    episodes.append(row); grouped[(name, allocator, policy)].append(metrics)
    means = []
    fields = ('on_time_completion_rate', 'weighted_on_time_rate', 'travel_steps', 'energy_used',
              'reinspections', 'missed_tasks', 'reassignments_after_outage', 'traffic_waits',
              'reports_sent', 'message_payload_bytes', 'collision_count')
    for (name, allocator, policy), values in sorted(grouped.items()):
        means.append({'scenario': name, 'allocator': allocator, 'policy': policy, 'missions': len(values),
                      **{field: mean(v[field] for v in values) for field in fields}})
    lookup = {(r['scenario'], r['seed'], r['allocator'], r['policy']): r for r in episodes}
    paired = []
    for name in SCENARIOS:
        for allocator in ALLOCATORS:
            differences = [lookup[(name, seed, allocator, 'verified')]['metrics']['on_time_completion_rate'] -
                           lookup[(name, seed, allocator, 'direct')]['metrics']['on_time_completion_rate'] for seed in selected_seeds]
            paired.append({'scenario': name, 'allocator': allocator, 'mean_completion_rate_difference': mean(differences),
                           'improved': sum(d > 0 for d in differences), 'equal': sum(d == 0 for d in differences),
                           'worse': sum(d < 0 for d in differences), 'missions_per_policy': len(differences)})
    summary = {'format': 'patrol-evaluation/v1', 'split': split, 'episodes': len(episodes),
               'seeds': selected_seeds, 'means': means, 'paired': paired,
               'source_sha256': source_hashes(), 'collision_count': sum(r['metrics']['collision_count'] for r in episodes)}
    write_json(output / 'episodes.json', episodes)
    write_json(output / 'summary.json', summary)
    write_json(output / 'checksums.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()})
    return summary
