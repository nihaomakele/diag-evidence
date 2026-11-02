"""Portable offline replay artifacts for fleet patrol demonstrations."""
from importlib.resources import files
import json
from pathlib import Path
import uuid

from .evaluation import POLICIES, write_json
from .models import scenario_to_dict
from .simulator import simulate


def run_demo(scenario, output_dir='artifacts/patrol', *, allocator='matching', write=True):
    if type(write) is not bool:
        raise ValueError('Invalid patrol output control')
    runs = {policy: simulate(scenario, policy=policy, allocator=allocator, record=write) for policy in POLICIES}
    summary = {'scenario': scenario.name, 'mission': scenario.mission, 'allocator': allocator,
               'metrics': {name: run['metrics'] for name, run in runs.items()}}
    if write:
        output = Path(output_dir) / ('patrol-' + uuid.uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=False)
        data = {'format': 'patrol-replay/v1', 'scenario': scenario_to_dict(scenario), 'runs': runs}
        write_json(output / 'scenario.json', data['scenario'])
        write_json(output / 'runs.json', data)
        write_json(output / 'summary.json', summary)
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        template = files(__package__).joinpath('replay_template.html').read_text(encoding='utf-8')
        (output / 'replay.html').write_text(template.replace('__PATROL_DATA__', payload), encoding='utf-8')
        summary = {**summary, 'output_dir': str(output), 'replay_path': str(output / 'replay.html')}
    return summary
