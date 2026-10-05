"""Run a coding agent (Claude Code, Codex or OpenCode) on one benchmark sample under a profile.

The sandbox holds exactly the sample's video (plus public task/robot handouts for native and twin profiles); hidden GT,
gt_pkg and meta.json never enter it, and the run log is audited for any path into the sample directory. Each round
ends with the profile's package checks; their errors are fed back to the agent.

Usage: python -m v2w.agents.run --profile P --sample SAMPLE_DIR --out PKG [--agent claude|codex|opencode] [--model M]
       [--base-url URL --api-key-env VAR] [--reasoning-effort E] [--max-rounds 2] [--dry-run]
--base-url points the agent at another endpoint: an OpenAI-compatible server (vLLM, SGLang, a hosted API) through
OpenCode or Codex, or an Anthropic-compatible one through Claude Code.
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from v2w import paths
from v2w.agents import checks, handouts

PROFILES = Path(__file__).resolve().parent / 'profiles'
# The claude CLI can die with SIGBUS when its cwd / output dir sit on a full network file system: work on local disk,
# then copy the package to --out.
LOCAL = Path(os.environ.get('V2W_AGENT_LOCAL') or Path(tempfile.gettempdir()) / 'v2w_agent')


def stage_library():
    """A private copy of the video2sim package for the agent, so its tools never point into the repository (and the data)."""
    lib = LOCAL / 'lib'
    if not (lib / 'video2sim').exists():
        tmp = LOCAL / f'lib.{os.getpid()}'
        shutil.copytree(paths.REPO / 'video2sim', tmp / 'video2sim', ignore=shutil.ignore_patterns('__pycache__'))
        try:
            tmp.rename(lib)
        except OSError:   # another run staged it first
            shutil.rmtree(tmp, ignore_errors=True)
    return lib
NO_COMMON = checks.NATIVE_PROFILES + checks.TWIN_PROFILES + checks.CLOTH_PROFILES
ROBOT_BRIEF = ('\nExecution backend: video2sim native rigid robot scene, robot.uid={robot}. The actual robot base pose is required for native '
               'self-replay in the camera frame; do not substitute an identity base for a differently oriented robot. Task scoring is benchmark-side.\n')
HAND_BRIEF = ('\nExecution robot: {robot}. This profile uses the benchmark hand adapter; generic bridge/WidowX replay is not its execution interface. '
              'Use the hand profile validator and public hand control contract.\n')


def profiles():
    return sorted(p.stem for p in PROFILES.glob('*.md') if p.stem != 'common')


def brief(profile):
    """Brief template of a profile (str.format fields: video, out, name, W, H, T, ours, repo, py_eval, py)."""
    text = (PROFILES / (profile + '.md')).read_text()
    if profile not in NO_COMMON:
        text += (PROFILES / 'common.md').read_text()
    robot = checks.ROBOTS.get(profile)
    if robot and profile.startswith('fb_'):
        text = text.replace('"uid": "widowx250s_bridge"', '"uid": "' + robot + '"').replace('// nominal: the robot is NOT part of this profile', '// actual execution robot')
        text += ROBOT_BRIEF.format(robot=robot)
    elif robot:
        text += HAND_BRIEF.format(robot=robot)
    return text


AGENTS = ('claude', 'codex', 'opencode')
PROVIDER = 'v2w'


def backend(a):
    """(adapter name, extra adapter options, model id the adapter receives)."""
    key = os.environ.get(a.api_key_env) or 'EMPTY'   # local servers usually accept any key
    if a.agent == 'claude':
        env = dict(ANTHROPIC_BASE_URL=a.base_url, ANTHROPIC_AUTH_TOKEN=key) if a.base_url else {}
        return 'claude', dict(env=env), a.model
    if a.agent == 'codex':
        config = []
        if a.base_url:
            config += [f'model_providers.{PROVIDER}={{name="{PROVIDER}", base_url="{a.base_url}", env_key="{a.api_key_env}", wire_api="responses"}}',
                       f'model_provider="{PROVIDER}"']
        if a.reasoning_effort:
            config.append(f'model_reasoning_effort="{a.reasoning_effort}"')
        return 'codex', dict(config=config, skip_permissions=True), a.model
    provider = None
    model = a.model
    if a.base_url:
        provider = {PROVIDER: dict(npm='@ai-sdk/openai-compatible', name=PROVIDER,
                                   options=dict(baseURL=a.base_url, apiKey=key), models={a.model: dict(name=a.model)})}
        model = f'{PROVIDER}/{a.model}'
    # opencode has no non-interactive permission prompt; isolation comes from the staged workspace and the log audit
    return 'opencode', dict(provider=provider, skip_permissions=True), model


def task_brief(profile, sample, video, out, workdir):
    """The brief the bundled agent receives, plus the public handouts staged next to the video."""
    meta = json.load(open(sample / 'meta.json'))
    if 'resolution' in meta and 'T' in meta:
        (W, H), T = meta['resolution'], meta['T']
    else:
        W, H, T = handouts.media_info(video)
    name = sample.name if not profile.startswith('ego_') else f'{sample.name}__{profile}'
    prompt = brief(profile).format(video=video, out=out, name=name, W=W, H=H, T=T, ours=stage_library(), repo=paths.REPO,
                                   py_eval=paths.tool('python'), py=paths.tool('twin_python'))
    if profile in checks.TWIN_PROFILES:
        public = handouts.stage_twin(profile, out)
        prompt += '\nPublic robot contract: ' + str(public) + '\n' + public.read_text()
    if profile in checks.NATIVE_PROFILES:
        public = handouts.stage_native(sample, workdir)
        prompt += '\nPublic task/robot specification: ' + str(public) + '\n' + public.read_text()
        if json.loads(public.read_text()).get('wallet_dynamics') == 'rigid_or_elastic':
            prompt += '\n' + (PROFILES / 'wallet_supplement.txt').read_text()
    return prompt


def prepare(a):
    """Task workspace for a custom agent: the video, the brief and the public handouts, nothing from the ground truth."""
    sample, out, task = Path(a.sample).resolve(), Path(a.out).resolve(), Path(a.prepare).resolve()
    if task.exists():
        shutil.rmtree(task)
    task.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    video = task / 'video.mp4'
    shutil.copy(sample / 'video.mp4', video)
    (out / 'source').mkdir(exist_ok=True)
    shutil.copy(video, out / 'source' / 'video.mp4')
    text = task_brief(a.profile, sample, video, out, task)
    (task / 'brief.md').write_text(text)
    check = f'v2w check --task {task} {out}'
    (task / 'task.json').write_text(json.dumps(dict(profile=a.profile, sample=sample.name, video=str(video), brief=str(task / 'brief.md'),
                                                    out=str(out), check=check), indent=1) + '\n')
    print(task)


def check(task, pkg=None):
    """Package checks of the task's profile; the same feedback the bundled driver gives its agent."""
    spec = json.loads((Path(task) / 'task.json').read_text())
    return checks.errors(spec['profile'], Path(pkg or spec['out']), paths.sample(spec['sample']))


def main(argv=None):
    from video2sim.bench.agent_run import audit, stage
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--profile', required=True, choices=profiles())
    ap.add_argument('--sample', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--agent', choices=AGENTS, help='coding agent CLI (default: claude, or opencode with --base-url)')
    ap.add_argument('--model', help='model id (default: claude-opus-5-5 for claude, the CLI default otherwise)')
    ap.add_argument('--base-url', help='model endpoint, e.g. http://localhost:8000/v1 for a vLLM server')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY', help='environment variable holding the endpoint key')
    ap.add_argument('--reasoning-effort', help='reasoning effort passed to Codex (e.g. high)')
    ap.add_argument('--max-rounds', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true', help='stage the sandbox and print the brief without running the agent')
    ap.add_argument('--video', default=None, help='pre-staged sandbox video; default: stage one from the sample')
    ap.add_argument('--prepare', metavar='TASK_DIR', help='only write the task workspace for a custom agent (see docs/custom_agent.md)')
    a = ap.parse_args(argv)
    if a.prepare:
        return prepare(a)
    sample, final_out = Path(a.sample).resolve(), Path(a.out).resolve()
    final_out.mkdir(parents=True, exist_ok=True)
    tag = final_out.name if final_out.name != 'pkg' else f'{sample.name}__{a.profile}__pkg'   # concurrent runs never share local dirs
    out, cwd = LOCAL / tag, LOCAL / f'cwd_{tag}'
    name = sample.name if not a.profile.startswith('ego_') else f'{sample.name}__{a.profile}'
    scratch = Path('/tmp') / name
    for d in (out, cwd, scratch):   # a fresh run never sees an earlier run's package or scratch
        if d.exists():
            shutil.rmtree(d)
    out.mkdir(parents=True)
    cwd.mkdir(parents=True)
    video = Path(a.video).resolve() if a.video else stage(sample, out)
    (out / 'source').mkdir(parents=True, exist_ok=True)
    shutil.copy(video, out / 'source' / 'video.mp4')
    ours = str(stage_library())
    prompt = task_brief(a.profile, sample, video, out, video.parent)
    if a.dry_run:
        print(prompt)
        return
    from video2sim import agents
    a.agent = a.agent or ('opencode' if a.base_url else 'claude')
    if a.agent == 'claude' and not a.model:
        a.model = 'claude-opus-5-5'
    if a.base_url and not a.model:
        raise SystemExit('--base-url needs --model (the model name served at that endpoint)')
    name, extra, model = backend(a)
    ad = agents.get(name)
    if not ad.available():
        raise SystemExit(f'the {ad.binary} CLI is not available on PATH')
    os.environ['PYTHONPATH'] = ours
    os.environ.pop('CLAUDECODE', None)
    opts = dict(model=model, skip_permissions=False, allowed_tools='Bash,Read,Edit,Write,Glob,Grep',
                add_dirs=[str(video.parent), str(out), str(scratch)])
    opts.update(extra)
    scratch.mkdir(exist_ok=True)
    rec = dict(profile=a.profile, sample=str(sample), agent=name, model=a.model, base_url=a.base_url, rounds=[], cost_usd=0.0,
               started=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    log, session, t0 = '', None, time.time()
    for r in range(a.max_rounds):
        res = agents.run_round(ad, prompt, opts, cwd=cwd, resume=session)
        log += res.log + '\n'
        rec['cost_usd'] += res.cost_usd or 0.0
        session = res.session
        errs = checks.errors(a.profile, out, sample)
        rec['rounds'].append(dict(round=r + 1, exit=res.exit_code, validate_errors=errs, cost_usd=res.cost_usd))
        (out / 'agent_log.txt').write_text(log)
        (out / 'agent_run.json').write_text(json.dumps(rec, indent=1, default=str))
        if not errs:
            break
        prompt = 'Your package at %s does not validate:\n- %s\nFix it (same rules apply) and re-run the validate command.' % (out, '\n- '.join(errs))
    rec.update(seconds_agent=round(time.time() - t0), leaks=audit(log, sample), finished=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    (out / 'agent_log.txt').write_text(log)
    (out / 'agent_run.json').write_text(json.dumps(rec, indent=1, default=str))
    if final_out != out:
        shutil.rmtree(final_out)
        shutil.copytree(out, final_out, symlinks=False)
    print(json.dumps({k: v for k, v in rec.items() if k != 'rounds'}, indent=1, default=str))


if __name__ == '__main__':
    main()
