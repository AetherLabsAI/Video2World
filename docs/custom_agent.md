# Evaluating your own agent

`v2w run --agent-cmd` runs any program as the coding agent: a research prototype, an agent framework, or a script that
calls a model API. Video2World prepares each task, launches your command, evaluates the package it writes and scores
the experiment, exactly as for the bundled Claude Code / Codex / OpenCode drivers.

## How a run works

For every instance (`<family>/<sample>`) `v2w run` does:

1. **prepare** a task workspace `runs/<name>/<family>/<sample>/task/` containing only public inputs;
2. **launch** your command (a shell template) with the placeholders below filled in;
3. **evaluate** the package your command left in `{out}`, then compute the metrics;
4. after all instances, **score** the experiment (`runs/<name>/report/`).

Each step is resumable: an instance whose `{out}/protocol.json` exists is not run again.

## Placeholders

| Placeholder | Value |
|---|---|
| `{task}` | task workspace directory |
| `{video}` | `{task}/video.mp4`, the source video (the only observation of the scene) |
| `{brief}` | `{task}/brief.md`, the task brief: what to reconstruct, the target robot and simulator, the delivery frame and format, and how to self-check |
| `{out}` | directory where the package must be written |
| `{model}` | value of `--model` (empty if not given) |
| `{profile}` | brief profile of the instance (e.g. `fb_furniture`, `ego_arm`, `robodojo_candidate_task_v1`) |
| `{family}` | task family |

The task workspace contains:

```
task/
  video.mp4      source video
  brief.md       the brief, identical to what the bundled agents receive
  task.json      {"profile", "sample", "video", "brief", "out", "check"}
  ...            public task / robot handouts for the RoboDojo and in-house profiles
```

`brief.md` is self-contained: hand it to your model as the task prompt. It refers to `{video}` and `{out}` by
absolute path and gives the validation command for the package.

## What your agent must deliver

A [video2sim protocol package](../video2sim/PROTOCOL.md) in `{out}`: `protocol.json` (scene, robot, camera frame,
task), the action stream, meshes, and a short `report.md`. The brief states the target robot, the action convention and
which entities are scored. Two commands help the agent check its work before it stops:

```bash
python -m video2sim.cli validate {out}     # the package is well formed
v2w check --task {task}                    # the same feedback the bundled drivers give their agents
```

Your command should exit with status 0 once the package is complete. A missing or invalid package counts as a build
failure for that instance (S = 0); it is never retried.

## Rules

- Use only the task workspace. The sample directories under `data/` contain the hidden ground truth and must not be
  read, listed or searched by the agent. The bundled drivers audit their logs for such accesses; a custom agent is
  responsible for enforcing the same isolation (e.g. by running it in a container that mounts only `{task}` and
  `{out}`).
- One attempt per instance, no selection over repeated runs, no human intervention.
- Wall-clock limit: 3 hours per instance; the process group is killed afterwards.
- The agent may run simulators, render its own rollouts and iterate; it may not use another instance's package.

## Example 1: a wrapper script

```bash
#!/bin/bash
# my_agent.sh TASK OUT MODEL
set -e
TASK=$1; OUT=$2; MODEL=$3
cd "$TASK"
my-agent --model "$MODEL" --prompt-file brief.md --workdir "$OUT"
python -m video2sim.cli validate "$OUT"
```

```bash
v2w run --name my_agent --model my-model-v1 --gpus 0,1 --workers 4 \
    --agent-cmd "bash my_agent.sh {task} {out} {model}"
```

## Example 2: a Python agent on an OpenAI-compatible API

A minimal tool-use loop: the model receives the brief, may run shell commands in the task workspace, and stops when the
package checks pass. Real agents add image viewing (the brief asks the agent to look at video frames), file editing
tools and context management.

```python
# my_agent.py
import json, subprocess, sys
from openai import OpenAI

task = json.load(open(sys.argv[1] + '/task.json'))
client = OpenAI()                                  # OPENAI_BASE_URL / OPENAI_API_KEY, e.g. a vLLM server
shell = {'type': 'function', 'function': {
    'name': 'shell', 'description': 'Run a bash command in the task workspace.',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}
messages = [{'role': 'user', 'content': open(task['brief']).read()}]

for _ in range(200):
    reply = client.chat.completions.create(model=sys.argv[2], messages=messages, tools=[shell]).choices[0].message
    messages.append(reply.model_dump(exclude_none=True))
    if not reply.tool_calls:
        check = subprocess.run(task['check'], shell=True, capture_output=True, text=True)
        if check.returncode == 0:
            break
        messages.append({'role': 'user', 'content': 'The package does not pass the checks:\n' + check.stdout})
        continue
    for call in reply.tool_calls:
        cmd = json.loads(call.function.arguments)['command']
        res = subprocess.run(cmd, shell=True, cwd=sys.argv[1], capture_output=True, text=True, timeout=1800)
        messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': (res.stdout + res.stderr)[-20000:]})
```

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
v2w run --name my_loop --model Qwen/Qwen3-Coder --agent-cmd "python my_agent.py {task} {model}"
```

## Packages produced elsewhere

If your system produces packages outside Video2World, lay them out as `<family>/<sample>/` (each containing
`protocol.json`) and evaluate them directly:

```bash
v2w eval --name submitted --packages /path/to/packages --gpus 0,1 --workers 2
```

Instances without a package count as build failures. Use the briefs from a prepared task workspace
(`python -m v2w.agents.run --profile P --sample data/samples/S --out OUT --prepare TASK`) so that packages use the
expected robot, frame and format.

## Outputs

```
runs/<name>/<family>/<sample>/
  task/          the task workspace your agent received
  pkg/           the package ({out})
  agent.log      stdout/stderr of your command
  agent.json     exit code, duration, package present, check errors
  eval/          simulator artifacts (scene, rollout, videos with --video)
  metrics.json   the metric record
runs/<name>/report/  summary.json, samples.json, samples.csv
```

Restrict a run with `--family` / `--sample`, re-evaluate with `v2w eval --name <name> --force`, and re-score with
`v2w score --name <name>`.
