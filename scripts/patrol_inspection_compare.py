"""Compare alert policies on paired patrol layouts using the same fixture."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from diag_evidence.patrol import ExecutionConfig, make_scenario, simulate
from diag_evidence.patrol.evaluation import digest, write_json
from diag_evidence.patrol.inspection import inspection_from_dict
from diag_evidence.patrol.models import scenario_to_dict

from patrol_execution_compare import source_hashes


SEEDS = list(range(80, 100))
MODES = ('all', 'verified')
METRICS = ('inspection_tasks', 'inspection_on_time', 'inspection_confirmed',
           'inspection_fault_findings', 'inspection_normal_findings',
           'routine_on_time', 'energy_used', 'traffic_waits', 'collision_count')


def compare(inspection_file, output):
    payload = inspection_file.read_bytes()
    plan = inspection_from_dict(json.loads(payload))
    execution = ExecutionConfig()
    protocol = {'seeds': SEEDS, 'scenario': 'clean', 'alarm_policies': MODES,
                'fixture_sha256': sha256(payload).hexdigest(), 'observation_policy': 'verified',
                'allocator': 'matching', 'traffic': 'yield', 'execution': asdict(execution),
                'source_sha256': source_hashes()}
    import diag_evidence.patrol.inspection as inspection
    protocol['source_sha256']['inspection.py'] = sha256(Path(inspection.__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    episodes = []
    for seed in SEEDS:
        scenario = make_scenario('clean', seed)
        inputs = digest({'scenario': scenario_to_dict(scenario),
                         'fixture_sha256': protocol['fixture_sha256']})
        for mode in MODES:
            run = simulate(scenario, execution=execution, traffic='yield',
                           inspection=plan, alarm_policy=mode, record=False)
            episodes.append({'seed': seed, 'alarm_policy': mode, 'input_sha256': inputs,
                             'metrics': run['metrics'], 'assets': run['inspection_assets']})
    averages = {mode: {
        key: mean(row['metrics'][key] for row in episodes if row['alarm_policy'] == mode)
        for key in METRICS
    } for mode in MODES}
    write_json(output / 'episodes.json', episodes)
    write_json(output / 'summary.json', {'runs': len(episodes), 'pairs': len(episodes) // 2, 'means': averages})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compare(args.inspection, args.output_dir)
