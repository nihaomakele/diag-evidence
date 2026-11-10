"""Paired validation of tracked patrol execution against the existing mode."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from diag_evidence.patrol import ExecutionConfig, make_scenario, simulate
from diag_evidence.patrol.evaluation import CORE_FILES, digest, write_json
from diag_evidence.patrol.models import scenario_to_dict
from diag_evidence.patrol import execution


SEEDS = list(range(40, 60))
PROFILES = {
    'clean': ('clean', None, False),
    'delayed': ('delayed', None, False),
    'delayed_loss': ('delayed', None, True),
    'outage': ('outage', None, False),
    'long_delay': ('delayed', 32, False),
}
CONFIG = ExecutionConfig()
FIELDS = ('on_time_completion_rate', 'reinspections', 'energy_used',
          'reassignments_after_outage', 'reassignments_after_timeout',
          'execution_queries', 'execution_unconfirmed_completed', 'modeled_payload_bytes')
SOURCES = (*CORE_FILES, 'execution.py')


def source_hashes():
    folder = Path(execution.__file__).parent
    return {name: sha256((folder / name).read_bytes()).hexdigest() for name in SOURCES}


def compare(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = output / 'comparison_protocol.json'
    frozen = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'baseline_commit': 'e0f6f4a67e29bf36e8e004564fe061677dcc2cdf',
        'source_sha256': source_hashes(), 'profiles': PROFILES,
        'seeds': SEEDS, 'policy': 'verified', 'allocator': 'matching',
        'confirmed_settings': asdict(CONFIG),
    }
    write_json(protocol_path, frozen)
    episodes = []
    for profile, (name, delay, drop) in PROFILES.items():
        for seed in SEEDS:
            scenario = make_scenario(name, seed)
            if delay is not None:
                scenario = replace(scenario, report_delay=delay)
            input_hash = digest({'scenario': scenario_to_dict(scenario), 'drop_completion_reports': drop})
            for mode, config in (('legacy', None), ('confirmed', CONFIG)):
                metrics = simulate(scenario, record=False, execution=config,
                                   drop_completion_reports=drop)['metrics']
                metrics['modeled_payload_bytes'] = metrics['message_payload_bytes'] + metrics.get('execution_payload_bytes', 0)
                assert metrics['collision_count'] == 0
                episodes.append({'profile': profile, 'seed': seed, 'mode': mode,
                                 'input_sha256': input_hash, 'metrics': metrics})
    assert frozen['source_sha256'] == source_hashes()
    for i in range(0, len(episodes), 2):
        assert episodes[i]['input_sha256'] == episodes[i + 1]['input_sha256']
    means = {}
    for profile in PROFILES:
        means[profile] = {}
        for mode in ('legacy', 'confirmed'):
            metrics = [r['metrics'] for r in episodes if r['profile'] == profile and r['mode'] == mode]
            means[profile][mode] = {
                field: mean(m[field] for m in metrics) if all(field in m for m in metrics) else None
                for field in FIELDS
            }
    write_json(output / 'comparison_episodes.json', episodes)
    write_json(output / 'comparison_summary.json',
               {'runs': len(episodes), 'matched_inputs': len(episodes) // 2, 'means': means,
                'protocol_sha256': sha256(protocol_path.read_bytes()).hexdigest()})
    print(json.dumps(means, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for paired results')
    compare(parser.parse_args().output_dir)
