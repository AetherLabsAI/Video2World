"""Command line: `v2w setup | list | har | run | eval | score`."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from . import benchmark, paths, runner, scoring

DEFAULT_MODEL = 'claude-opus-5-5'


def _configs(a):
    configs = benchmark.configurations(families=a.family, samples=a.sample)
    if not configs:
        sys.exit('no configuration matches the selection')
    return configs


def _print_report(report, out=None):
    if out:
        scoring.write_report(report, out)
    if report['readiness'] != 'ready':
        print(f"V2WScore: blocked ({len(report['issues'])} issues)")
        for issue in report['issues'][:20]:
            print(f"  {issue.get('family')}/{issue.get('sample')}: {issue['reason']}")
    else:
        d = report['dimensions']
        print(f"V2WScore {report['v2w_score']:.2f}   F {d['F']:.3f}   G {d['G']:.3f}   D {d['D']:.3f}   "
              f"({report['eligible']} configurations, {report['families']} families)")
        for name, group in report['subdimensions'].items():
            print(f"  {name} {group['value']:.3f} ({group['families']} families)")
    if out:
        print(f'report: {out}')


def cmd_setup(a):
    config = paths.local_config()
    for key in ('data', 'python', 'isaac_python', 'twin_python', 'robodojo_source', 'graphics_libs', 'ffmpeg'):
        value = getattr(a, key)
        if value:
            # absolute but not resolved: an environment's python is often a symlink, and resolving it leaves the environment
            config[key] = str(Path(value).expanduser().absolute()) if key != 'graphics_libs' else value
    paths.LOCAL.write_text(json.dumps(config, indent=1) + '\n')
    data = Path(config.get('data') or paths.REPO / 'data')
    print(f'config: {paths.LOCAL}')
    print(f"data:   {data} ({'ok' if (data / 'manifest.json').exists() else 'missing manifest.json'})")
    for key in ('python', 'isaac_python', 'twin_python', 'robodojo_source', 'ffmpeg'):
        value = config.get(key) or paths.tool(key, required=False)
        state = 'ok' if value and (Path(value).exists() or shutil.which(value)) else 'not set'
        print(f'{key:16s} {value or "-"} ({state})')


def cmd_list(a):
    groups = benchmark.families()
    print(f"{sum(len(v) for v in groups.values())} configurations, {len(groups)} families")
    for family, configs in groups.items():
        if a.family and family not in a.family:
            continue
        evaluator = Counter(c.evaluator for c in configs).most_common(1)[0][0]
        print(f'  {family:44s} {len(configs):3d}  {benchmark.TRACKS[evaluator]}')
        if a.samples:
            for c in configs:
                print('      ' + c.sample)


def cmd_har(a):
    configs = benchmark.configurations()
    records = {(c.family, c.sample): runner.load(paths.DATA / 'har' / c.family / f'{c.sample}.json') for c in configs}
    config = scoring.load_config(a.config)
    report = scoring.score(records, scoring.make_scope(benchmark.scope_rows(configs), config), config)
    _print_report(report, a.out)


def _experiment(a):
    return Path(a.runs) / a.name


def cmd_run(a):
    configs = _configs(a)
    steps = ['agent', 'eval', 'metrics'] if not a.no_agent else ['eval', 'metrics']
    agent_args = []
    for flag, value in (('--agent', a.agent), ('--base-url', a.base_url), ('--api-key-env', a.api_key_env),
                        ('--reasoning-effort', a.reasoning_effort)):
        if value:
            agent_args += [flag, value]
    runner.run(_experiment(a), configs, steps, model=a.model, agent_cmd=a.agent_cmd, agent_args=agent_args, gpus=a.gpus,
               workers=a.workers, video=a.video, force=a.force)
    if not a.family and not a.sample:
        _score(a, configs)


def cmd_eval(a):
    configs = _configs(a)
    if a.packages:   # <packages>/<family>/<sample>/ holding protocol.json (or a pkg/ subdirectory)
        root = Path(a.packages)
        for c in configs:
            src = root / c.family / c.sample
            src = src / 'pkg' if (src / 'pkg').is_dir() else src
            dst = runner.Experiment(_experiment(a)).dir(c) / 'pkg'
            if (src / 'protocol.json').exists() and not dst.exists():
                shutil.copytree(src, dst, symlinks=True)
            elif not (src / 'protocol.json').exists():
                runner.dump(dst.parent / 'agent.json', dict(package=False, note='no package submitted'))
    runner.run(_experiment(a), configs, ['eval', 'metrics'], gpus=a.gpus, workers=a.workers, video=a.video, force=a.force)
    if not a.family and not a.sample:
        _score(a, configs)


def cmd_check(a):
    from v2w.agents.run import check
    errors = check(a.task, a.package)
    for e in errors:
        print('[FAIL] ' + e)
    print('OK' if not errors else f'{len(errors)} error(s)')
    sys.exit(1 if errors else 0)


def _score(a, configs=None):
    configs = configs or benchmark.configurations()
    config = scoring.load_config(a.config)
    records = runner.Experiment(_experiment(a)).records(configs)
    report = scoring.score(records, scoring.make_scope(benchmark.scope_rows(configs), config), config)
    _print_report(report, a.out or _experiment(a) / 'report')


def cmd_score(a):
    _score(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog='v2w', description='Video2World benchmark')
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('setup', help='record the data location and simulator interpreters in v2w.local.json')
    s.add_argument('--data')
    s.add_argument('--python', help='evaluator interpreter (SAPIEN/ManiSkill)')
    s.add_argument('--isaac-python', dest='isaac_python', help='Isaac Sim interpreter (RoboDojo and in-house tracks)')
    s.add_argument('--twin-python', dest='twin_python', help='MuJoCo interpreter for the reconstructed twins')
    s.add_argument('--robodojo-source', dest='robodojo_source', help='RoboDojo checkout (upstream commit 25691aa)')
    s.add_argument('--graphics-libs', dest='graphics_libs', help='extra LD_LIBRARY_PATH for headless Isaac rendering')
    s.add_argument('--ffmpeg')
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser('list', help='families and configurations')
    s.add_argument('--family', action='append')
    s.add_argument('--samples', action='store_true')
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser('har', help='score the human-assisted reference shipped with the data')
    s.add_argument('--out')
    s.add_argument('--config', help='score configuration (default: v2w/config/score.json)')
    s.set_defaults(fn=cmd_har)

    def common(s, packages=False):
        s.add_argument('--name', required=True, help='experiment name (runs/<name>)')
        s.add_argument('--runs', default=str(paths.REPO / 'runs'))
        s.add_argument('--family', action='append')
        s.add_argument('--sample', action='append')
        s.add_argument('--gpus', type=lambda x: [int(g) for g in x.split(',')], default=[0])
        s.add_argument('--workers', type=int, default=1)
        s.add_argument('--video', action='store_true', help='render rollout videos')
        s.add_argument('--force', action='store_true', help='re-run evaluation and metrics')
        s.add_argument('--config')
        s.add_argument('--out')

    s = sub.add_parser('run', help='run a coding agent on the benchmark, evaluate and score')
    common(s)
    s.add_argument('--agent', choices=('claude', 'codex', 'opencode'),
                   help='coding agent CLI (default: claude; opencode when --base-url is given)')
    s.add_argument('--model', help=f'model id (default for claude: {DEFAULT_MODEL})')
    s.add_argument('--base-url', help='OpenAI-compatible endpoint (vLLM, SGLang, hosted APIs), e.g. http://localhost:8000/v1')
    s.add_argument('--api-key-env', help='environment variable holding the endpoint key (default: OPENAI_API_KEY)')
    s.add_argument('--reasoning-effort', help='reasoning effort for Codex, e.g. high')
    s.add_argument('--agent-cmd', help='custom agent command; placeholders {task} {video} {brief} {out} {model} {profile} '
                                       '{family} (docs/custom_agent.md)')
    s.add_argument('--no-agent', action='store_true', help='only evaluate packages already in the experiment')
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser('eval', help='evaluate submitted packages and score them')
    common(s)
    s.add_argument('--packages', help='directory <family>/<sample>/ with protocol packages')
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser('check', help='check a package against its task (the feedback the bundled driver gives its agent)')
    s.add_argument('--task', required=True, help='task workspace written for a custom agent')
    s.add_argument('package', nargs='?', help='package directory (default: the task output)')
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser('score', help='score an experiment')
    s.add_argument('--name', required=True)
    s.add_argument('--runs', default=str(paths.REPO / 'runs'))
    s.add_argument('--config')
    s.add_argument('--out')
    s.set_defaults(fn=cmd_score)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, RuntimeError) as exc:   # missing data or an unconfigured simulator: say what to do
        sys.exit(f'v2w: {exc}')


if __name__ == '__main__':
    main()
