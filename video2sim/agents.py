"""Coding-agent adapters: run one headless round, whoever the agent is.

The valuable parts of this project are already agent-agnostic — the brief is
markdown, the acceptance gates are CLI commands, the backends are libraries.
What is not portable is the ~60 lines that launch a specific CLI and read its
event stream. That lives here.

An adapter owes the driver three things:

  argv(prompt, opts, resume)   how to launch one round
  events(line) -> [Event]      normalise one stdout line
  session/usage from events    so the next round can resume

The normalised `Event.log_line` matters more than it looks. The video-only
audit greps the transcript for reads of restricted files, so every adapter
must render tool calls into the SAME shape (`  ▸ Tool: detail`). An adapter
that hides what the agent ran would silently turn the audit into a rubber
stamp — worse than having no audit, because the run still prints OK.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    """One normalised thing that happened during a round."""
    kind: str                       # "session" | "text" | "tool" | "result" | "raw"
    text: str = ""
    log_line: str = ""              # what goes into the transcript the audit reads
    session: str | None = None
    cost_usd: float | None = None
    usage: dict | None = None


@dataclass
class RunResult:
    session: str | None = None
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    exit_code: int = 0
    text: str = ""                  # the agent's last message
    log: str = ""                   # full transcript, what the audit reads


def _summ(v, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


class Adapter:
    name = "?"
    binary = "?"
    default_model: str | None = None
    #: adapters that cannot continue a conversation get a fresh round with the
    #: feedback prepended instead — correct, just more expensive
    supports_resume = True

    #: extra places to look when the binary is not on PATH. A batch worker
    #: inherits the environment of whatever launched it, and node-installed
    #: CLIs live under a version-specific bin that an interactive shell has
    #: but a subprocess usually does not.
    extra_bin_globs: tuple = (os.path.expanduser("~/.nvm/versions/node/*/bin"),)

    def resolve(self) -> str | None:
        found = shutil.which(self.binary)
        if found:
            return found
        import glob
        for pat in self.extra_bin_globs:
            for d in sorted(glob.glob(pat), reverse=True):
                cand = Path(d) / self.binary
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)
        return None

    def available(self) -> bool:
        return self.resolve() is not None

    def env(self, opts: dict) -> dict:
        """Extra environment for this agent's process (merged over os.environ)."""
        return {}

    def argv(self, prompt: str, opts: dict, resume: str | None) -> list[str]:
        raise NotImplementedError

    def events(self, line: str) -> list[Event]:
        raise NotImplementedError

    def count_image_views(self, session: str | None) -> int | None:
        """How many images the agent actually opened, when the transcript cannot say.

        Returning None means "no out-of-band record — count the transcript".
        An adapter whose CLI does not surface image views in its event stream
        MUST implement this, or the visual-inspection audit silently reports
        zero for a run that looked at every frame.
        """
        return None


class ClaudeAdapter(Adapter):
    name, binary = "claude", "claude"
    default_model = "claude-opus-5"

    def env(self, opts):
        return dict(opts.get("env") or {})

    def argv(self, prompt, opts, resume):
        cmd = [self.resolve() or self.binary, "-p", prompt, "--output-format", "stream-json",
               "--verbose", "--model", opts["model"]]
        if resume:
            cmd += ["--resume", resume]
        if opts.get("skip_permissions"):
            cmd += ["--dangerously-skip-permissions"]
        else:
            cmd += ["--permission-mode", "acceptEdits",
                    "--allowedTools", opts["allowed_tools"]]
        for d in opts.get("add_dirs", []):
            cmd += ["--add-dir", d]
        return cmd

    def events(self, line):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return [Event("raw", log_line=line)]
        t, out = ev.get("type"), []
        if t == "system" and ev.get("subtype") == "init":
            out.append(Event("session", session=ev.get("session_id"),
                             log_line=f"[session {ev.get('session_id')}] "
                                      f"model={ev.get('model')}"))
        elif t == "assistant":
            for blk in (ev.get("message") or {}).get("content", []):
                if blk.get("type") == "text" and blk.get("text", "").strip():
                    out.append(Event("text", text=blk["text"].strip(),
                                     log_line=f"\n{blk['text'].strip()}\n"))
                elif blk.get("type") == "tool_use":
                    inp = blk.get("input", {})
                    detail = inp.get("command") or inp.get("file_path") or inp
                    out.append(Event("tool",
                                     log_line=f"  ▸ {blk.get('name')}: {_summ(detail)}"))
        elif t == "result":
            out.append(Event("result", text=ev.get("result", "") or "",
                             cost_usd=ev.get("total_cost_usd"),
                             usage=ev.get("usage"), log_line=""))
        return out


class CodexAdapter(Adapter):
    name, binary = "codex", "codex"
    default_model = None            # whatever the user's codex config selects

    def env(self, opts):
        return dict(opts.get("env") or {})

    def argv(self, prompt, opts, resume):
        cmd = [self.resolve() or self.binary, "exec", "--json", "--skip-git-repo-check"]
        if opts.get("model"):
            cmd += ["-m", opts["model"]]
        # Codex's own sandbox is bubblewrap; inside an unprivileged container
        # it fails with "bwrap: Failed to make / slave" and EVERY shell command
        # the agent runs comes back exit 1. The run then looks like a confused
        # agent rather than a broken sandbox.
        #
        # The same helper backs view_image and apply_patch, so under `-s
        # read-only` / `-s workspace-write` an image opens only where bwrap
        # actually works: it fails with "fs sandbox helper failed" here and on
        # hosts with no bwrap at all. Under the bypass flag below the helper is
        # out of the path and view_image works everywhere we have measured, so
        # a skip-permissions run CAN see. Do not infer blindness from the
        # transcript: codex's event stream carries no item for an image view
        # (see count_image_views) — a run that opened twelve frames prints the
        # same empty log as one that opened none.
        for item in opts.get("config", []):   # `-c key=value` overrides: provider, reasoning effort
            cmd += ["-c", item]
        cmd += (["--dangerously-bypass-approvals-and-sandbox"]
                if opts.get("skip_permissions")
                else ["-s", "workspace-write"])
        for d in opts.get("add_dirs", []):
            cmd += ["--add-dir", d]
        if resume:
            cmd += ["resume", resume, prompt]
        else:
            cmd += [prompt]
        return cmd

    def events(self, line):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return [Event("raw", log_line=line)] if line.strip() else []
        t, out = ev.get("type"), []
        if t == "thread.started":
            sid = ev.get("thread_id")
            out.append(Event("session", session=sid, log_line=f"[session {sid}]"))
        elif t in ("item.completed", "item.started"):
            it = ev.get("item") or {}
            kind = it.get("type")
            if kind == "agent_message" and t == "item.completed":
                txt = (it.get("text") or "").strip()
                if txt:
                    out.append(Event("text", text=txt, log_line=f"\n{txt}\n"))
            elif kind == "command_execution" and t == "item.started":
                # same shape the Claude adapter emits, so the audit's patterns
                # match a shell read regardless of which agent ran it
                out.append(Event("tool",
                                 log_line=f"  ▸ Bash: {_summ(it.get('command', ''))}"))
            elif kind == "file_change" and t == "item.completed":
                for ch in it.get("changes") or []:
                    out.append(Event("tool",
                                     log_line=f"  ▸ {ch.get('kind', 'edit').title()}: "
                                              f"{ch.get('path')}"))
        elif t == "turn.completed":
            out.append(Event("result", usage=ev.get("usage"), log_line=""))
        return out

    def count_image_views(self, session: str | None) -> int | None:
        """Count image views from codex's own rollout, not from our transcript.

        `codex exec --json` emits items for command_execution and file_change
        and for nothing else, so an image view leaves NO trace in the stream we
        parse — measured directly: a run whose model correctly described a
        probe image produced only agent_message items. The rollout file does
        record it, as a Code Mode `custom_tool_call` whose script calls
        `tools.view_image({path})`, or as a plain `view_image` function_call on
        older builds. Both forms are counted here.

        Reading the transcript instead is what produced the claim that GPT-6
        "never opened a frame" while it was in fact opening four to twelve per
        run.
        """
        if not session:
            return None
        import glob as _glob
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        rolls = _glob.glob(str(home / "sessions" / "*" / "*" / "*" /
                               f"rollout-*{session}*.jsonl"))
        if not rolls:
            return None
        n = 0
        for r in rolls:
            try:
                for line in open(r, errors="ignore"):
                    if "view_image" not in line:
                        continue
                    try:
                        pay = json.loads(line).get("payload") or {}
                    except json.JSONDecodeError:
                        continue
                    if pay.get("type") == "custom_tool_call":
                        n += (pay.get("input") or "").count("tools.view_image(")
                    elif (pay.get("type") == "function_call"
                          and pay.get("name") == "view_image"):
                        n += 1
            except OSError:
                continue
        return n


class OpenCodeAdapter(Adapter):
    """opencode (MIT). The provider-agnostic one: `-m provider/model` points at
    OpenRouter, a local vLLM/ollama server, or anything OpenAI-compatible, so a
    run need not depend on any vendor's CLI subscription.

    Models verified end to end on one episode (ep087670, can pick-and-place;
    every one delivered, both gates 0, in-distribution, object left upright):

        openrouter/~deepseek/deepseek-v4-flash-latest   2 rds  $0.58  z 2.50
        openrouter/~anthropic/claude-sonnet-latest      5 rds  $6.93  z 2.98
        openrouter/~moonshotai/kimi-latest             13 rds $24.04  z 2.94

    The open-weight DeepSeek is the default: cheapest, fewest rounds, and the
    distribution score closest to the benchmark's own data — and it leaves the
    chain with no proprietary component. `z` is p95 |z| against RoboDojo's
    published statistics, where their own data scores ~1.70.
    """
    name, binary = "opencode", "opencode"
    default_model = "openrouter/~deepseek/deepseek-v4-flash-latest"
    #: strings proven on a real episode; anything else is untested, not invalid
    verified_models = ("openrouter/~deepseek/deepseek-v4-flash-latest",
                       "openrouter/~anthropic/claude-sonnet-latest",
                       "openrouter/~moonshotai/kimi-latest")

    def argv(self, prompt, opts, resume):
        cmd = [self.resolve() or self.binary, "run", "--format", "json"]
        if opts.get("model"):
            cmd += ["-m", opts["model"]]
        if resume:
            cmd += ["-s", resume]
        return cmd + [prompt]

    def env(self, opts):
        """opencode has no permission flag, and headless it still stops to ask
        ("Permission required: external_directory") — with stdin closed that is
        a hang, not a refusal. Config comes in through the environment so the
        user's own opencode.json is left alone."""
        # per-tool, never the top-level string: `{"permission": "allow"}` is
        # accepted by the schema but this build silently runs NO tools under it
        # — the agent narrates what it would do and every file stays unwritten
        act = "allow" if opts.get("skip_permissions") else "ask"
        cfg = {"permission": {k: act for k in
                              ("read", "list", "glob", "grep", "edit", "bash",
                               "task", "external_directory", "todowrite",
                               "webfetch", "websearch")}}
        # `question` is denied even in skip-permissions mode: there is no user
        # to answer one. An agent that decides to "check with the user" blocks
        # on a closed stdin forever — one run sat at 0% CPU for 101 minutes
        # after announcing it wanted to raise a concern. Denying it makes the
        # agent state the concern in its report and carry on, which is what
        # `status: failure|infeasible` is for.
        cfg["permission"]["question"] = "deny"
        if opts.get("provider"):   # e.g. an OpenAI-compatible endpoint (vLLM, SGLang, a hosted API)
            cfg["provider"] = opts["provider"]
        return {"OPENCODE_CONFIG_CONTENT": json.dumps(cfg)}

    def events(self, line):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return [Event("raw", log_line=line)] if line.strip() else []
        t, part, out = ev.get("type"), ev.get("part") or {}, []
        sid = ev.get("sessionID")
        if sid and t == "step_start":
            out.append(Event("session", session=sid, log_line=""))
        if t == "text":
            txt = (part.get("text") or "").strip()
            if txt:
                out.append(Event("text", text=txt, log_line=f"\n{txt}\n"))
        elif t == "tool_use":
            st = part.get("state") or {}
            inp = st.get("input") or {}
            detail = (inp.get("command") or inp.get("filePath")
                      or st.get("title") or inp)
            # same transcript shape as the other adapters, so the video-only
            # audit's patterns apply unchanged
            out.append(Event("tool",
                             log_line=f"  ▸ {part.get('tool')}: {_summ(detail)}"))
        elif t == "step_finish":
            out.append(Event("result", cost_usd=part.get("cost"),
                             usage=part.get("tokens"), log_line=""))
        elif t == "error":
            err = json.dumps(ev.get("error"), ensure_ascii=False)[:300]
            out.append(Event("raw", log_line=f"  !! opencode error: {err}"))
        return out


ADAPTERS = {a.name: a for a in (ClaudeAdapter(), CodexAdapter(), OpenCodeAdapter())}


def get(name: str) -> Adapter:
    if name not in ADAPTERS:
        raise SystemExit(f"unknown agent {name!r}; have: {', '.join(ADAPTERS)}")
    a = ADAPTERS[name]
    if not a.available():
        raise SystemExit(f"{a.binary!r} is not on PATH — install it or pick "
                         f"another --agent")
    return a


def _kill_group(pid: int) -> None:
    """Reap anything the agent CLI left behind (it spawns its own binary)."""
    import os
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_round(adapter: Adapter, prompt: str, opts: dict, cwd: Path,
              resume: str | None = None) -> RunResult:
    """One headless round. Streams normalised progress; returns the transcript."""
    import os
    cmd = adapter.argv(prompt, opts, resume)
    env = {**os.environ, **adapter.env(opts)}
    # stdin must be closed: codex blocks on "Reading additional input from
    # stdin" when it is a pipe that never reaches EOF, and the run just hangs
    # Own process group, and tear the whole group down on the way out. Killing
    # the driver alone leaves the agent CLI running: two such orphans survived
    # five hours here, still billing the API key, because pkill matched only
    # the parent.
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, env=env, start_new_session=True)
    res = RunResult()
    lines: list[str] = []
    for line in proc.stdout:
        for e in adapter.events(line.rstrip("\n")):
            if e.log_line:
                print(e.log_line, flush=True)
                lines.append(e.log_line)
            if e.session:
                res.session = e.session
            if e.cost_usd:
                res.cost_usd += e.cost_usd
            if e.usage:
                for k, v in e.usage.items():
                    if isinstance(v, (int, float)):
                        res.usage[k] = res.usage.get(k, 0) + v
            if e.kind in ("text", "result") and e.text:
                res.text = e.text
    try:
        proc.wait()
    finally:
        # also on KeyboardInterrupt / a driver that dies mid-round
        _kill_group(proc.pid)
    res.exit_code = proc.returncode
    res.log = "\n".join(lines)
    return res
