"""V2WScore: build-gated, family-balanced aggregate shared by agents and the human-assisted reference.

S = build * (F + G + D) / 3 per configuration, F = alpha * success + (1 - alpha) * progress, and every error metric
becomes quality = max(0, 1 - error / tolerance). Configurations are averaged within a family and families get equal
weight. The common scope is fixed from the GT metric contracts in the manifest before any result is read; a missing or
invalid applicable measurement blocks the score instead of shrinking the denominator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import numbers
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean

from . import __version__

OUTCOMES = ['build', 'task_success', 'progress.progress']
DIMENSIONS = ('geometry', 'dynamics')
NUMERIC_OUTCOMES = {'progress.progress'}
RATES = {'build', 'task_success'}
SCENE_REVISION = 'scene_cd_v5/rev8-assembly-object-alignment'
TERMINAL_POSITION_VERSION = 'terminal-position/1.1'
TERMINAL_POSITION_COMPATIBLE = {'terminal-position/1.0', TERMINAL_POSITION_VERSION}
UNIT_DOMAIN = {'scene_missing_fraction', 'scene_extra_fraction', 'progress.progress', 'mask_iou_topview'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str, allow_nan=False).encode()).hexdigest()


def value_at(record, path):
    for part in path.split('.'):
        if not isinstance(record, dict):
            return None
        record = record.get(part)
    return record


def put_at(record, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        if not isinstance(record.get(part), dict):
            record[part] = {}
        record = record[part]
    record[parts[-1]] = value


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


# ---------------------------------------------------------------- metric contracts
def metric_contract(family, columns):
    """(required, applicable, not-applicable reasons) for a configuration; applicability is GT-owned."""
    kind = family.get('eval2')
    required = {'build', 'task_success'}
    applicable = set(required)
    if kind == 'rigid':
        applicable |= {'scene_chamfer_cm', 'object.shape_cd_cm', 'object.center_err_cm', 'object.size_err_cm',
                       'ape.trans_cm', 'rpe.trans_rpe_cm', 'rpe.rot_rpe_deg', 'hidden_replay_success', 'progress.progress',
                       'own_trajectory.ape.trans_cm', 'own_trajectory.ape.rot_deg', 'own_trajectory.rpe.trans_rpe_cm',
                       'own_trajectory.duration_normalized.ape.trans_cm', 'own_trajectory.duration_normalized.time_scale'}
        required |= applicable
        applicable |= {'scene_missing_fraction', 'scene_extra_fraction', 'table_depth_err_cm'}
        required |= {'scene_missing_fraction', 'scene_extra_fraction', 'table_depth_err_cm'}
        if family.get('eval2_kind') not in {'rigid_lift', 'rigid_push'}:   # relative target pose is undefined for lifting/pushing
            applicable.add('relative.rel_trans_err_cm')
            required.add('relative.rel_trans_err_cm')
    elif kind == 'artic':
        applicable |= {'scene_chamfer_cm', 'hidden_replay_success', 'progress.progress'}
        required |= {'hidden_replay_success', 'progress.progress'}
    elif kind == 'twin':
        applicable |= {'scene_chamfer_cm', 'object.shape_cd_cm', 'object.center_err_cm', 'object.size_err_cm', 'progress.progress'}
        required |= applicable
    configured = family.get('metric_contract', {})
    applicable |= set(configured.get('applicable', []))
    required |= set(configured.get('required', []))
    declared_na = configured.get('not_applicable', {})
    applicable -= set(declared_na)
    required -= set(declared_na)
    required |= applicable - set(configured.get('optional', []))
    return required, applicable, {key: declared_na.get(key, 'not defined by this evaluator contract') for key in columns if key not in applicable}


def validate_record(raw, family, columns):
    """Normalized copy of a metric record and the issues that block it; malformed rates never become successes."""
    if not isinstance(raw, dict):
        return None, ['missing metric record'], {}
    record = deepcopy(raw)
    required, applicable, na = metric_contract(family, columns)
    review = family.get('case_metric_review')
    if review:
        for key in review['metrics']:
            put_at(record, key, None)
    issues = []
    if raw.get('error') and raw.get('error_kind') != 'agent':
        issues.append('evaluator metric error: ' + str(raw['error']))
    if 'build' in applicable and type(raw.get('build')) is not bool:
        issues.append('build: expected boolean')
        record['build'] = None
    if record.get('build') is False:   # a method build failure is a defined failed task
        record['task_success'] = False if 'task_success' in applicable else None
        for key in NUMERIC_OUTCOMES & applicable:
            put_at(record, key, None if issues else 0.)
    for key in columns:
        if key in NUMERIC_OUTCOMES and key in na:
            put_at(record, key, None)
            continue
        value = value_at(record, key)
        if value is None:
            if key in required and (key in RATES or key in NUMERIC_OUTCOMES or record.get('build') is True):
                issues.append(key + ': unexpectedly missing')
            continue
        rate = key in RATES
        valid = type(value) is bool if rate else (isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value)))
        if not valid:
            issues.append(key + ': expected ' + ('boolean' if rate else 'finite number'))
            put_at(record, key, None)
        elif not rate and (float(value) < 0 or (key in UNIT_DOMAIN and float(value) > 1)):
            issues.append(key + ': outside metric domain')
            put_at(record, key, None)
    if record.get('build') is True:
        for metric, version_key, expected in (('object.size_err_cm', 'object.size_protocol', 'intrinsic-obb-full-extents/2'),
                                              ('own_trajectory.rpe.trans_rpe_cm', 'own_trajectory.rpe.protocol', 'base-displacement-rpe/1')):
            if metric in applicable and metric in columns and value_at(record, metric) is not None and value_at(record, version_key) != expected:
                issues.append(metric + ': stale or missing formula version ' + expected)
        if family.get('eval2') == 'rigid' and 'progress.progress' in applicable and 'progress.progress' in columns \
                and value_at(record, 'progress.protocol') != 'task-progress/2':
            issues.append('progress: stale or missing task-progress/2 formula')
        if family.get('eval2') in {'rigid', 'artic'} and 'scene_chamfer_cm' in columns and 'scene_chamfer_cm' in applicable:
            if record.get('scene_metric_revision') != SCENE_REVISION:
                issues.append('scene: stale or missing ' + SCENE_REVISION + ' revision')
            else:
                for key, cap in (('scene_chamfer_cm', 20.), ('table_depth_err_cm', 10.)):
                    value = value_at(record, key)
                    if key in columns and isinstance(value, numbers.Real) and not isinstance(value, bool) and value > cap:
                        issues.append(key + ': exceeds the scene cap')
                        put_at(record, key, None)
    return record, sorted(set(issues)), na


# ---------------------------------------------------------------- configuration
def subdimension_parents(config):
    groups = config.get('subdimensions') or {}
    parents = {}
    for name, keys in groups.items():
        owners = {d for d in DIMENSIONS for key in keys if key in config[d]}
        parents[name] = owners.pop() if len(owners) == 1 else None
    return parents


def validate_config(config):
    if not config.get('version') or not config.get('status'):
        raise ValueError('score config must identify its version and status')
    if not number(config.get('alpha')) or not 0 <= config['alpha'] <= 1:
        raise ValueError('alpha must be in [0, 1]')
    for dimension in DIMENSIONS:
        if not isinstance(config.get(dimension), dict) or not config[dimension]:
            raise ValueError('empty metric configuration: ' + dimension)
        for key, tau in config[dimension].items():
            if not number(tau) or tau <= 0:
                raise ValueError('tolerance must be positive: ' + key)
    if set(config['geometry']) & set(config['dynamics']):
        raise ValueError('a metric cannot appear in both dimensions')
    for old, new in config.get('geometry_replacements', {}).items():
        if old not in config['geometry'] or new not in config['geometry'] or old == new:
            raise ValueError('invalid geometry replacement')
    groups = config.get('subdimensions')
    if groups is None:
        return
    parents, assigned = subdimension_parents(config), defaultdict(list)
    for name, keys in groups.items():
        if name in ('F', 'G', 'D', 'S') or not keys or len(set(keys)) != len(keys) or parents[name] is None:
            raise ValueError('invalid subdimension: ' + name)
        assigned[parents[name]].extend(keys)
    for dimension, keys in assigned.items():
        if len(set(keys)) != len(keys) or set(keys) != set(config[dimension]):
            raise ValueError('subdimensions must partition every metric of ' + dimension)
    owner = {key: name for name, keys in groups.items() for key in keys}
    if any(owner.get(old) != owner.get(new) for old, new in config.get('geometry_replacements', {}).items()):
        raise ValueError('a geometry replacement cannot cross subdimensions')


# ---------------------------------------------------------------- scope and score
def make_scope(rows, config):
    """Common scope from the manifest's GT metric contracts, fixed before any result is read."""
    validate_config(config)
    columns = OUTCOMES + list(config['geometry']) + list(config['dynamics'])
    out, seen = [], set()
    for row in rows:
        identity = (row['family'], row['sample'])
        if identity in seen:
            raise ValueError('duplicate configuration: ' + str(identity))
        seen.add(identity)
        spec = row['metric_family']
        _, applicable, na = metric_contract(spec, columns)
        geometry = [k for k in config['geometry'] if k in applicable]
        for old, replacement in config.get('geometry_replacements', {}).items():
            if old in geometry and replacement in geometry:
                geometry.remove(old)
        dynamics = [k for k in config['dynamics'] if k in applicable]
        reasons = [key + ': ' + na[key] for key in OUTCOMES if key not in applicable]
        if not geometry:
            reasons.append('no applicable geometric measurement')
        if not dynamics:
            reasons.append('no applicable dynamic measurement')
        out.append(dict(family=row['family'], sample=row['sample'], metric_family=spec, geometry=geometry, dynamics=dynamics,
                        eligible=not reasons, exclusion_reasons=reasons))
    if not out:
        raise ValueError('empty manifest')
    return dict(config_sha256=digest(config), rows=sorted(out, key=lambda r: (r['family'], r['sample'])))


def score(records, scope, config):
    """records: {(family, sample): metric record}. Returns the full report."""
    subdimensions = config.get('subdimensions') or {}
    parents = subdimension_parents(config)
    issues, scored = [], []
    diagnostic_keys = OUTCOMES + list(config['geometry']) + list(config['dynamics'])
    for gt in scope['rows']:
        identity = (gt['family'], gt['sample'])
        raw = records.get(identity)
        entry = {k: deepcopy(gt[k]) for k in ('family', 'sample', 'eligible', 'exclusion_reasons')}
        entry.update(status='not_applicable', build=None, F=None, G=None, D=None, S=None, geometry={}, dynamics={},
                     diagnostics={k: value_at(raw, k) for k in diagnostic_keys} if isinstance(raw, dict) else {})
        if subdimensions:
            entry['subdimensions'] = {name: None for name in subdimensions}
        local = []
        if raw is None:
            local.append('missing record')
        elif raw.get('error_kind') == 'evaluator':
            local.append('evaluator failure: ' + str(raw.get('error')))
        elif type(raw.get('build')) is not bool:
            local.append('build must be boolean')
        else:
            entry['build'] = raw['build']
        if not local and gt['eligible']:
            columns = OUTCOMES + gt['geometry'] + gt['dynamics']
            rec, errors, _ = validate_record(raw, gt['metric_family'], columns)
            local.extend(errors)
            for key, verdict in (rec.get('metric_status') or {}).items():
                if key in columns and isinstance(verdict, dict) and verdict.get('status') == 'error':
                    local.append(key + ': ' + str(verdict.get('reason', 'metric error')))
            if not local and rec['build'] is False:
                entry.update(status='build_failed', F=0., S=0., gated_F=0., gated_G=0., gated_D=0.)
                if subdimensions:
                    entry['gated_sub'] = {name: 0. for name in subdimensions if any(k in gt[parents[name]] for k in subdimensions[name])}
            elif not local:
                s, q = rec['task_success'], value_at(rec, 'progress.progress')
                if type(s) is not bool or not number(q) or not 0 <= q <= 1:
                    local.append('invalid functional outcome')
                for dimension in DIMENSIONS:
                    for key in gt[dimension]:
                        error = value_at(rec, key)
                        if not number(error) or error < 0:
                            local.append('invalid applicable error: ' + key)
                            continue
                        tau = config[dimension][key]
                        entry[dimension][key] = dict(error=error, tau=tau, quality=max(0., 1. - error / tau))
                if not local:
                    F = config['alpha'] * float(s) + (1. - config['alpha']) * q
                    G = mean(v['quality'] for v in entry['geometry'].values())
                    D = mean(v['quality'] for v in entry['dynamics'].values())
                    entry.update(status='ok', F=F, G=G, D=D, S=(F + G + D) / 3, gated_F=F, gated_G=G, gated_D=D)
                    for name, keys in subdimensions.items():
                        present = [k for k in keys if k in entry[parents[name]]]
                        if present:
                            value = mean(entry[parents[name]][k]['quality'] for k in present)
                            entry['subdimensions'][name] = value
                            entry.setdefault('gated_sub', {})[name] = value
        if local:
            entry['status'] = 'blocked'
            issues.extend(dict(family=identity[0], sample=identity[1], reason=x) for x in sorted(set(local)))
        scored.append(entry)
    eligible = [r for r in scored if r['eligible']]
    groups = defaultdict(list)
    for row in eligible:
        groups[row['family']].append(row)
    ready = not issues and bool(groups)
    by_family = {}
    for family, rows in sorted(groups.items()):
        ok = all(r['status'] in {'ok', 'build_failed'} for r in rows)
        by_family[family] = dict(n=len(rows), build_failed=sum(r['status'] == 'build_failed' for r in rows),
                                 score=100 * mean(r['S'] for r in rows) if ok else None,
                                 **{k: mean(r['gated_' + k] for r in rows) if ok else None for k in ('F', 'G', 'D')})
        if subdimensions:
            by_family[family]['subdimensions'] = {}
            for name in subdimensions:
                present = [r['gated_sub'][name] for r in rows if ok and name in (r.get('gated_sub') or {})]
                by_family[family]['subdimensions'][name] = dict(n=len(present), value=mean(present) if present else None)
    groups_report = {}
    for name in subdimensions:
        families = [f for f, v in by_family.items() if v['subdimensions'][name]['value'] is not None]
        groups_report[name] = dict(dimension=parents[name], metrics=list(subdimensions[name]),
                                   value=mean(by_family[f]['subdimensions'][name]['value'] for f in families) if ready and families else None,
                                   families=len(families))
    return dict(v2w_version=__version__, score_config=config['version'], config_sha256=scope['config_sha256'],
                readiness='ready' if ready else 'blocked',
                v2w_score=mean(r['score'] for r in by_family.values()) if ready else None,
                dimensions={k: mean(r[k] for r in by_family.values()) if ready else None for k in ('F', 'G', 'D')},
                subdimensions=groups_report, configurations=len(scored), eligible=len(eligible), families=len(groups),
                exclusion_counts=dict(Counter(reason for r in scored for reason in r['exclusion_reasons'])),
                by_family=by_family, issues=issues, samples=scored)


def load_config(path=None):
    config = json.loads(Path(path or Path(__file__).parent / 'config/score.json').read_text())
    validate_config(config)
    return config


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = report['samples']
    (out / 'summary.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=1) + '\n')
    (out / 'samples.json').write_text(json.dumps(samples, indent=1) + '\n')
    with (out / 'samples.csv').open('w', newline='') as f:
        keys = ['family', 'sample', 'eligible', 'status', 'build', 'F', 'G', 'D', 'S']
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: row[k] for k in keys})
