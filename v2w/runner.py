"""Experiment runner: coding agent -> simulation package -> evaluator -> metric record, resumable per configuration.

runs/<experiment>/<family>/<sample>/
    pkg/            the submitted protocol package (agent output, or copied in for `v2w eval`)
    agent.log       agent transcript
    eval/           evaluator artifacts (eval.json, scene/replay/rollout json, videos)
    metrics.json    the metric record the score reads
"""
from __future__ import annotations

import importlib
import json
import os
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import benchmark, paths

INFRA_ERRORS = ('vk::', 'Vulkan', 'DeviceLost', 'CUDA error', 'CUDA out of memory', 'cudaError', 'Segmentation fault', 'Bus error')
AGENT_TIMEOUT = 3 * 3600
EVAL_TIMEOUT = 5400


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str) + '\n')
    tmp.replace(path)


def log(directory, message):
    with open(Path(directory) / 'run.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')


class Experiment:
    def __init__(self, root):
        self.root = Path(root)

    def dir(self, config):
        return self.root / config.family / config.sample

    # ------------------------------------------------------------ agent
    def agent(self, config, model, agent_cmd=None, agent_args=(), timeout=AGENT_TIMEOUT):
        d = self.dir(config)
        pkg = d / 'pkg'
        if (pkg / 'protocol.json').exists():
            return 'done'
        d.mkdir(parents=True, exist_ok=True)
        if agent_cmd:   # a custom agent gets a task workspace, never the sample directory
            task = d / 'task'
            subprocess.run([paths.tool('python'), '-m', 'v2w.agents.run', '--profile', config.profile, '--sample', str(config.sample_dir),
                            '--out', str(pkg), '--prepare', str(task)], cwd=paths.REPO, env=paths.env(), check=True, capture_output=True)
            fields = dict(task=task, video=task / 'video.mp4', brief=task / 'brief.md', out=pkg, model=model or '',
                          profile=config.profile or '', family=config.family)
            cmd, shell = agent_cmd.format(**fields), True
        else:
            cmd = [paths.tool('python'), '-m', 'v2w.agents.run', '--profile', config.profile, '--sample', str(config.sample_dir),
                   '--out', str(pkg), *(['--model', model] if model else []), *agent_args]
            shell = False
        t0 = time.time()
        with open(d / 'agent.log', 'a') as f:
            f.write(f'===== {time.strftime("%Y-%m-%d %H:%M:%S")} {cmd if shell else shlex.join(cmd)}\n')
            f.flush()
            proc = subprocess.Popen(cmd, shell=shell, cwd=paths.REPO, env=paths.env(), stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:   # stop the agent and everything it started
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                rc = -9
        ok = rc == 0 and (pkg / 'protocol.json').exists()
        record = dict(rc=rc, seconds=round(time.time() - t0), package=ok)
        if agent_cmd and ok:
            from v2w.agents.run import check
            record['check_errors'] = check(d / 'task')
        dump(d / 'agent.json', record)
        return 'done' if ok else 'failed'

    # ------------------------------------------------------------ evaluator
    def evaluate(self, config, gpu=0, video=False, force=False, retry=True):
        d = self.dir(config)
        pkg, ev = d / 'pkg', d / 'eval'
        out = ev / 'eval.json'
        if out.exists() and not force:
            return 'done'
        ev.mkdir(parents=True, exist_ok=True)
        if not (pkg / 'protocol.json').exists():
            if not (d / 'agent.json').exists():
                return 'pending'
            # The agent ran and delivered no package: a method build failure.
            dump(out, dict(sample=config.sample, build=False, error='agent produced no package', error_kind='agent'))
            return 'done'
        tool, module, extra = benchmark.EVALUATORS[config.evaluator]
        cmd = [paths.tool(tool), '-m', module, '--run', str(pkg), '--sample', str(config.sample_dir), '--out', str(out), *extra]
        if video:
            cmd += ['--video-dir', str(ev / 'video')]
        with open(ev / 'eval.log', 'a') as f:
            f.write(f'===== {time.strftime("%Y-%m-%d %H:%M:%S")} {shlex.join(cmd)}\n')
            f.flush()
            try:
                rc = subprocess.run(cmd, cwd=paths.REPO, env=paths.env(CUDA_VISIBLE_DEVICES=gpu), stdout=f, stderr=subprocess.STDOUT,
                                    timeout=EVAL_TIMEOUT).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        if not out.exists():   # an evaluator crash is never attributed to the method
            dump(out, dict(sample=config.sample, build=False, error=f'evaluator exited with {rc}, see eval/eval.log', error_kind='evaluator'))
        record = load(out)
        infra = infra_error(ev)
        if infra and not record.get('build'):
            if retry:   # the renderer or GPU died under the evaluator: retry once on another GPU
                log(d, f'infrastructure error ({infra}); retrying')
                for p in ev.iterdir():
                    if p.is_file():
                        p.unlink()
                return self.evaluate(config, pick_gpu(exclude=(int(gpu),)), video, force=True, retry=False)
            record.update(error_kind='evaluator', error=f'infrastructure error in the evaluator: {infra}')
            dump(out, record)
        return 'done'

    # ------------------------------------------------------------ metrics
    def metrics(self, config, force=False):
        d = self.dir(config)
        out, ev_path = d / 'metrics.json', d / 'eval/eval.json'
        if not ev_path.exists():
            return 'pending'
        if out.exists() and not force and out.stat().st_mtime >= ev_path.stat().st_mtime:
            return 'done'
        ev = load(ev_path) or {}
        if ev.get('error_kind') in ('agent', 'evaluator') and not ev.get('build'):
            record = dict(sample=config.sample, build=False if ev['error_kind'] == 'agent' else None,
                          task_success=False if ev['error_kind'] == 'agent' else None, error_kind=ev['error_kind'], error=ev.get('error'))
        else:
            track = importlib.import_module(f'v2w.tracks.{config.track}')
            try:
                record = track.metrics(config.spec, config.sample_dir, d / 'eval')
                record.setdefault('sample', config.sample)
            except Exception as exc:   # a metric failure blocks the score; it is not a method failure
                record = dict(sample=config.sample, error=f'{type(exc).__name__}: {exc}', error_kind='evaluator')
        dump(out, record)
        return 'done'

    def records(self, configs):
        return {(c.family, c.sample): load(self.dir(c) / 'metrics.json') for c in configs}


def infra_error(ev_dir):
    for name in ('scene_pkg.json', 'replay_pkg.json', 'rollout_pkg.json', 'eval.json'):
        record = load(Path(ev_dir) / name) or {}
        for key in ('error', 'traceback'):
            text = str(record.get(key) or '')
            hit = next((t for t in INFRA_ERRORS if t in text), None)
            if hit:
                return text.strip().splitlines()[-1][:160] if text.strip() else hit
    return None


def pick_gpu(exclude=()):
    """Least-loaded visible GPU."""
    try:
        res = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return 0
    rows = []
    for line in res.stdout.strip().splitlines():
        try:
            i, u, m = [int(float(x)) for x in line.split(',')]
            rows.append((u, m, i))
        except ValueError:
            pass
    rows = [r for r in rows if r[2] not in set(exclude)] or rows
    return min(rows)[2] if rows else 0


def run(experiment, configs, steps, model=None, agent_cmd=None, agent_args=(), gpus=(0,), workers=1, video=False, force=False):
    """Run the requested steps for every configuration; GPUs are assigned round-robin per worker."""
    exp = Experiment(experiment)
    exp.root.mkdir(parents=True, exist_ok=True)

    def one(item):
        i, config = item
        gpu = gpus[i % len(gpus)]
        status = {}
        try:
            if 'agent' in steps:
                status['agent'] = exp.agent(config, model, agent_cmd, agent_args)
            if 'eval' in steps:
                status['eval'] = exp.evaluate(config, gpu, video=video, force=force)
            if 'metrics' in steps:
                status['metrics'] = exp.metrics(config, force=force)
        except Exception as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
        print(f'{config.key}: ' + ', '.join(f'{k}={v}' for k, v in status.items()), flush=True)
        return config, status

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(one, enumerate(configs)))
