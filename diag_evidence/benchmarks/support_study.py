"""Frozen source-only support screening evaluated on two new HIT-SM rigs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .baseline import extract_features
from .grouped_validation import digest, write_json
from .health_models import StudyModel
from .hust_data import load_windows
from .hust_health import classification_metrics
from .ottawa_external import frozen_probabilities
from .support import fit_support, support_scores, calibrate_record_threshold, acceptance_masks
from . import hit_data


def summarize(rows, probabilities, distances, threshold):
    y = np.array([r['label'] for r in rows])
    predicted = probabilities.argmax(axis=1)
    masks = acceptance_masks(probabilities, distances, threshold)
    records = []
    for name in sorted({r['record_id'] for r in rows}):
        indices = np.array([i for i, r in enumerate(rows) if r['record_id'] == name])
        labels = set(y[indices].tolist())
        if len(labels) != 1:
            raise ValueError('Conflicting record labels')
        guess = int(np.bincount(predicted[indices], minlength=4).argmax())
        records.append({'record_id': name, 'label': int(y[indices[0]]), 'prediction': guess,
            'windows': len(indices), 'maximum_distance': float(distances[indices].max()),
            'record_accepted': {key: bool(mask[indices].all()) for key, mask in masks.items()},
            'accepted_windows': {key: int(mask[indices].sum()) for key, mask in masks.items()}})
    ry, rp = np.array([r['label'] for r in records]), np.array([r['prediction'] for r in records])
    arms = {key: {'windows': classification_metrics(y, predicted, mask),
                  'records': classification_metrics(ry, rp, np.array([r['record_accepted'][key] for r in records]))}
            for key, mask in masks.items()}
    return {'raw_windows': classification_metrics(y, predicted),
            'raw_records': classification_metrics(ry, rp), 'arms': arms, 'per_record': records}


def _hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'checksums.json'}


def run_support_study(cache_dir, model_dir, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
        'learner': 'fixed existing shape/logistic_C1', 'calibration_alpha': .05,
        'calibration_unit': 'maximum window distance per original source record',
        'cross_validation': 'four leave-type-out folds, fold-specific scaling and record medians',
        'HIT_used_to_fit_or_tune': False, 'source_results_role': 'development calibration'})
    prior = Path(model_dir)
    model = json.loads((prior / 'external_model.json').read_text())
    prior_receipt = json.loads((prior / 'external_selection.json').read_text())
    if model['candidate'] != 'shape/logistic_C1' or digest(model) != prior_receipt['model_parameter_sha256']:
        raise ValueError('Expected unchanged frozen source classifier')
    signals, rows = load_windows(cache_dir, normalize=False)
    x = np.stack([extract_features(signal) for signal in signals])
    del signals
    y, groups = np.array([r['label'] for r in rows]), np.array([r['bearing_type'] for r in rows])
    oof_probabilities, oof_distances = np.empty((len(rows), 4)), np.empty(len(rows))
    folds = []
    for group in sorted(set(groups.tolist())):
        train, val = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
        training = [{**rows[i], 'split': 'train'} for i in train]
        learner = StudyModel('shape', 'logistic_C1').fit(x[train], y[train], training)
        support = fit_support(x[train], training)
        oof_probabilities[val] = learner.probabilities(x[val])
        oof_distances[val], _ = support_scores(x[val], support)
        train_ids = sorted({r['record_id'] for r in training})
        val_ids = sorted({rows[i]['record_id'] for i in val})
        if set(train_ids) & set(val_ids):
            raise ValueError('Source record leakage')
        folds.append({'validation_type': group, 'training_records': train_ids,
                      'validation_records': val_ids, 'support_sha256': digest(support)})
    calibration = calibrate_record_threshold(oof_distances, [{**r, 'split': 'validation'} for r in rows])
    training = [{**r, 'split': 'train'} for r in rows]
    refit = StudyModel('shape', 'logistic_C1').fit(x, y, training)
    if refit.export() != model:
        raise ValueError('Source refit differs from the existing frozen classifier')
    checkpoint = {'format': 'support-study/v1', 'classifier': model, 'support': fit_support(x, training)}
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['support.py', 'support_study.py', 'hit_data.py', 'hit_manifest.json']}
    selection = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'checkpoint_sha256': digest(checkpoint),
        'calibration': calibration, 'source_records': refit.training_records,
        'source_model_parameter_sha256': digest(model), 'code_sha256': code,
        'data_manifest_sha256': digest(hit_data.manifest()), 'source_classifier_preserved': True,
        'classifier_quality_screen_passed': prior_receipt['quality_screen_passed'],
        'target_fitting': False, 'default_promoted': False}
    write_json(output / 'source_folds.json', folds)
    write_json(output / 'source_calibration.json', summarize(rows, oof_probabilities, oof_distances, calibration['threshold']))
    write_json(output / 'checkpoint.json', checkpoint)
    write_json(output / 'frozen_selection.json', selection)
    write_json(output / 'checksums.json', _hashes(output))
    return selection


def evaluate_hit_support(selection_dir, cache_dir, output_dir, *, download=False):
    selection_dir, output = Path(selection_dir), Path(output_dir)
    frozen_bytes = (selection_dir / 'frozen_selection.json').read_bytes()
    selection = json.loads(frozen_bytes)
    checkpoint = json.loads((selection_dir / 'checkpoint.json').read_text())
    if checkpoint['format'] != 'support-study/v1' or digest(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Support checkpoint differs from frozen receipt')
    for name, expected in selection['code_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Support experiment code differs from frozen version')
    if digest(hit_data.manifest()) != selection['data_manifest_sha256']:
        raise ValueError('HIT manifest differs from frozen version')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
        'selection_sha256_before_data': hashlib.sha256(frozen_bytes).hexdigest(),
        'checkpoint_sha256': selection['checkpoint_sha256'], 'target_role': 'two new rigs; known true classes 0,1,3'})
    signals, rows, receipts, layouts = hit_data.load_windows(cache_dir, download=download)
    features = np.stack([extract_features(signal) for signal in signals])
    probabilities = frozen_probabilities(checkpoint['classifier'], features)
    distances, nearest = support_scores(features, checkpoint['support'])
    threshold = selection['calibration']['threshold']
    metrics = {}
    for rig in ['selfbuilt', 'spectraquest']:
        indices = np.array([i for i, r in enumerate(rows) if r['rig'] == rig])
        metrics[rig] = summarize([rows[i] for i in indices], probabilities[indices], distances[indices], threshold)
    masks = acceptance_masks(probabilities, distances, threshold)
    predictions = [{**row, 'prediction': int(probabilities[i].argmax()),
        'probabilities': probabilities[i].tolist(), 'support_distance': float(distances[i]),
        'nearest_source_record': nearest[i], 'accepted': {name: bool(mask[i]) for name, mask in masks.items()}}
        for i, row in enumerate(rows)]
    if (selection_dir / 'frozen_selection.json').read_bytes() != frozen_bytes:
        raise RuntimeError('Frozen source selection changed during evaluation')
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'data_receipts.json', receipts)
    write_json(output / 'signal_layouts.json', layouts)
    write_json(output / 'checksums.json', _hashes(output))
    return metrics
