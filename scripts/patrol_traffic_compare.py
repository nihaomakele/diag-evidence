"""Compare the two traffic policies while keeping execution confirmation fixed."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from diag_evidence.patrol import make_scenario, simulate
from diag_evidence.patrol.evaluation import digest, write_json
from diag_evidence.patrol.models import scenario_to_dict

from patrol_execution_compare import CONFIG, PROFILES, source_hashes


SPLITS = {'seen': list(range(40, 60)), 'new': list(range(60, 80))}
FIELDS = ('on_time_completion_rate', 'reinspections', 'energy_used',
          'traffic_waits', 'traffic_replans', 'static_path_searches', 'traffic_yields')


def compare(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_hashes(),
        'profiles': PROFILES,
        'splits': SPLITS,
        'execution': asdict(CONFIG),
        'observation_policy': 'verified',
        'allocator': 'matching',
    }
    write_json(output / 'protocol.json', protocol)
    episodes, summaries = [], {}
    for split, seeds in SPLITS.items():
        summaries[split] = {}
        for profile, (name, delay, drop) in PROFILES.items():
            pairs = []
            for seed in seeds:
                scenario = make_scenario(name, seed)
                if delay is not None:
                    scenario = replace(scenario, report_delay=delay)
                input_hash = digest({'scenario': scenario_to_dict(scenario),
                                     'drop_completion_reports': drop})
                pair = {}
                for traffic in ('legacy', 'yield'):
                    metrics = simulate(scenario, record=False, execution=CONFIG,
                                       drop_completion_reports=drop, traffic=traffic)['metrics']
                    pair[traffic] = metrics
                    episodes.append({'split': split, 'profile': profile, 'seed': seed,
                                     'input_sha256': input_hash, 'traffic': traffic, 'metrics': metrics})
                pairs.append(pair)
            deltas = [p['yield']['on_time_completed'] - p['legacy']['on_time_completed'] for p in pairs]
            summaries[split][profile] = {
                'means': {
                    mode: {field: mean(p[mode].get(field, 0) for p in pairs) for field in FIELDS}
                    for mode in ('legacy', 'yield')
                },
                'improved_pairs': sum(d > 0 for d in deltas),
                'unchanged_pairs': sum(d == 0 for d in deltas),
                'worse_pairs': sum(d < 0 for d in deltas),
            }
    if protocol['source_sha256'] != source_hashes():
        raise RuntimeError('Patrol source changed during the comparison')
    write_json(output / 'episodes.json', episodes)
    write_json(output / 'summary.json', {'runs': len(episodes), 'pairs': len(episodes) // 2,
                                         'results': summaries})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    compare(parser.parse_args().output_dir)
