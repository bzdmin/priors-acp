"""Re-run the coordination demonstration with a stated number of observations.

One kept promise followed by N broken ones, against a counterparty told to fail
deterministically. Four things here are correctness rather than convenience.

**Collection has to bypass the gate, and this is not a workaround.** After one
kept and two broken promises the counterparty's reliability sits at 0.4168,
below the 0.4669 bar, so Priors refuses to create any further job for it and
`hire.ts` returns without touching a wallet. Once a counterparty carries any
adverse record this applies to the first hire of a new run as well, so every
collection hire bypasses, not only the ones meant to break.

That is the mechanism working. It also means evidence about a blocked
counterparty cannot be gathered through the gated path, because only a job
produces an observation and the gate stops jobs.
So the collection hires run with `PRIORS_NO_MEMORY=1`, which is the flag
`hire.ts` already carries for the deletion condition. The observations are
unaffected: each one records what Digest actually did after declaring itself
available. What is bypassed is Priors' own refusal to keep asking.

**The counterparty's lifecycle is owned.** npm spawns tsx as a child, and on
Windows killing the wrapper leaves the child alive. This kills the tree. Four
orphaned Digest instances answering one job is what produced a double BudgetSet
during manual testing.

**A faulted hire never exits.** `hire.ts` stops on job.completed, job.rejected
or job.expired. A job nobody answers emits none of the three, so the buyer would
listen forever. This takes the job id and stops the buyer, which changes
nothing: the job is already on chain and silence is the point.

**Output is read on a thread.** Reading a pipe inline blocks, so a timeout
checked in the loop body only fires while output keeps arriving, which is
precisely not the case when something has hung.

    python scripts/rerun_coordination.py --broken 14 --dry-run
    python scripts/rerun_coordination.py --broken 14

State is saved after every step. A run that dies re-observes the job it already
created rather than creating another one.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.chain import DEFAULT_RPC, FALLBACK_RPCS, _Rpc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUYER = ROOT / "buyer"
DIGEST = ROOT / "digest"
RESUME = ROOT / ".rerun_coordination.json"
PROVIDER = "0x5043147b8b666ac070e01ff659e1fbbbc2462bc7"

#: The SDK floor is 5 minutes. Wait past it before asking whether a promise was
#: kept, or the observer correctly refuses to score a job still in play.
SLA_WAIT_SECONDS = 5 * 60 + 45

#: How long to let a kept promise run to completion before giving up.
KEPT_TIMEOUT_SECONDS = 420

#: How long to wait for a job id before deciding the buyer is stuck.
CREATE_TIMEOUT_SECONDS = 240

JOB_CREATED_RE = re.compile(r"\[job (\d+)\]\s+created on Base")
IN_FLIGHT_RE = re.compile(r"resuming (\d+) in-flight job")
VERDICT_RE = re.compile(r"verdict\s*:\s*declaration (\w+)")
FATAL_RE = re.compile(r"\[error\] fatal|ECONNRESET|ETIMEDOUT|not valid JSON")

#: observe_response.py reports whether the declaration held, in its own words.
VERDICT = {"confirmed": "kept", "contradicted": "broken"}
REFUSED_RE = re.compile(r"no job created, nothing spent")

WINDOWS = os.name == "nt"


class Stalled(RuntimeError):
    """The buyer declined to create a job, and said why."""


class HireFailed(RuntimeError):
    """The buyer died before creating anything. Safe to try again."""


@dataclass
class Observation:
    index: int
    intended: str            # "kept" or "broken"
    from_block: int
    job_id: int | None = None
    recorded: str | None = None


@dataclass
class RunState:
    broken_target: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def done_kept(self) -> bool:
        return any(o.intended == "kept" and o.recorded for o in self.observations)

    @property
    def done_broken(self) -> int:
        return sum(1 for o in self.observations
                   if o.intended == "broken" and o.recorded)

    def unfinished(self) -> Observation | None:
        """A job that was created but never scored, from an interrupted run."""
        for o in self.observations:
            if o.job_id is not None and o.recorded is None:
                return o
        return None


def load_state(target: int) -> RunState:
    if not RESUME.exists():
        return RunState(target)
    try:
        raw = json.loads(RESUME.read_text())
    except (json.JSONDecodeError, OSError):
        print("resume file unreadable, starting fresh")
        return RunState(target)
    if raw.get("broken_target") != target:
        print("resume file is for a different target, ignoring it")
        return RunState(target)
    st = RunState(target)
    fields = set(Observation.__dataclass_fields__)
    for o in raw.get("observations", []):
        kept = {k: v for k, v in o.items() if k in fields}
        if kept.get("recorded") in VERDICT:      # an older run's wording
            kept["recorded"] = VERDICT[kept["recorded"]]
        st.observations.append(Observation(**kept))
    print(f"resuming: {len(st.observations)} observation(s) on record")
    return st


def save_state(st: RunState) -> None:
    RESUME.write_text(json.dumps(
        {"broken_target": st.broken_target,
         "observations": [asdict(o) for o in st.observations]}, indent=2))


def head_block() -> int:
    rpc = _Rpc(urls=(DEFAULT_RPC,) + FALLBACK_RPCS)
    return int(rpc.call("eth_blockNumber", []), 16)


# -- process handling -------------------------------------------------------

def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def strays() -> list[str]:
    """Agent processes this script did not start. Never fatal."""
    try:
        if not WINDOWS:
            out = subprocess.run(["pgrep", "-af", "node|tsx"],
                                 capture_output=True, text=True).stdout
        else:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" "
                 "| Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.splitlines()
            if "seller" in l.lower() or "hire" in l.lower()]


def npm(args: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        ["npm", *args], cwd=cwd, env=env, shell=WINDOWS, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def reader_thread(proc: subprocess.Popen) -> queue.Queue:
    """Drain stdout on a thread so the caller can enforce a real timeout."""
    q: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line.rstrip())
        except Exception:
            pass
        finally:
            q.put(None)

    threading.Thread(target=pump, daemon=True).start()
    return q


def node_env() -> dict:
    env = os.environ.copy()
    # An IPv6 DNS stall killed Digest at startup more than once on this machine.
    env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "")
                           + " --dns-result-order=ipv4first").strip()
    return env


def start_digest(faults: int) -> subprocess.Popen:
    env = node_env()
    env["DIGEST_FAULT_NEXT"] = str(faults)
    print(f"  starting Digest, DIGEST_FAULT_NEXT={faults}")
    proc = npm(["start"], DIGEST, env)
    reader_thread(proc)          # drain, or the pipe fills and Digest blocks
    time.sleep(12)
    if proc.poll() is not None:
        raise RuntimeError("Digest exited immediately. Run `npm start` in "
                           "digest/ by hand to see why.")
    return proc


# -- one hire ---------------------------------------------------------------

def run_hire(intended: str) -> int:
    """Create one job and return its id."""
    env = node_env()
    # Every collection hire bypasses the gate, the kept one included. The
    # counterparty already carries a record from an earlier run, so the gate is
    # shut before this even starts: it read 0.4587 against the 0.4669 bar and
    # refused the first job. There is no clean slate to begin from, which is
    # the selective labels problem in its most literal form. Collection runs
    # through the `--no-memory` path in coordinate.py; the verdict afterwards
    # uses the full record.
    env["PRIORS_NO_MEMORY"] = "1"

    proc = npm(["run", "hire"], BUYER, env)
    q = reader_thread(proc)

    job_id: int | None = None
    waiting_for = "a job id"
    deadline = time.time() + CREATE_TIMEOUT_SECONDS
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                if job_id is None:
                    raise HireFailed(f"timed out waiting for {waiting_for}")
                raise RuntimeError(f"timed out waiting for {waiting_for}")
            try:
                line = q.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if line is None:                       # the process ended
                break
            if line:
                print("      " + line[:150])

            if job_id is None:
                stall = IN_FLIGHT_RE.search(line)
                if stall:
                    raise Stalled(
                        f"hire.ts found {stall.group(1)} in-flight job(s) and "
                        f"created none. If those jobs are past their deadline, "
                        f"the guard in buyer/src/hire.ts is not excluding them.")
                if REFUSED_RE.search(line):
                    raise Stalled(
                        "hire.ts refused to create a job. If this says "
                        "coordination declined, the response gate has closed "
                        "on this counterparty and PRIORS_NO_MEMORY is not "
                        "reaching it.")
                m = JOB_CREATED_RE.search(line)
                if m:
                    job_id = int(m.group(1))
                    if intended == "broken":
                        break                      # it will never resolve
                    waiting_for = "the job to complete"
                    deadline = time.time() + KEPT_TIMEOUT_SECONDS
                    continue

            if job_id is None and FATAL_RE.search(line):
                # Public RPC endpoints rate-limit and occasionally serve an
                # HTML error page. Nothing was created, so this is retryable.
                raise HireFailed(f"buyer died before creating a job: {line[:120]}")

            if intended == "kept" and ("done - " in line or "COMPLETED" in line):
                break
    finally:
        kill_tree(proc)

    if job_id is None:
        raise HireFailed("hire.ts exited without creating a job")
    return job_id


def hire_with_retry(intended: str, attempts: int = 3) -> int:
    for attempt in range(1, attempts + 1):
        try:
            return run_hire(intended)
        except HireFailed as e:
            if attempt == attempts:
                raise
            print(f"    {e}")
            print(f"    retrying ({attempt}/{attempts - 1}) in 30s")
            time.sleep(30)
    raise HireFailed("unreachable")


def observe(job_id: int, from_block: int, attempts: int = 5) -> str:
    """Ask the resolver for a verdict, waiting out PENDING and RPC lag."""
    last = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "observe_response.py"),
             "--job", str(job_id), "--from-block", str(from_block)],
            capture_output=True, text=True, timeout=600)
        last = proc.stdout + proc.stderr
        m = VERDICT_RE.search(last)
        if m:
            word = m.group(1).lower()
            return VERDICT.get(word, word)
        transient = "PENDING" in last or "no events found" in last
        if transient and attempt < attempts:
            print(f"      not scoreable yet, waiting 90s "
                  f"({attempt}/{attempts})")
            time.sleep(90)
            continue
        break
    print(last[-900:])
    raise RuntimeError(f"no verdict for job {job_id} after {attempts} attempts")


# -- the run ----------------------------------------------------------------

def finish_unfinished(st: RunState) -> None:
    """Score a job an interrupted run created but never observed."""
    obs = st.unfinished()
    if obs is None:
        return
    print(f"\n[{obs.index}] job {obs.job_id} was created but never scored, "
          f"observing it rather than creating another")
    if obs.intended == "broken":
        print(f"    waiting {SLA_WAIT_SECONDS}s in case its deadline is ahead")
        time.sleep(SLA_WAIT_SECONDS)
    obs.recorded = observe(obs.job_id, obs.from_block)
    save_state(st)
    print(f"    recorded {obs.recorded}")


def one_observation(st: RunState, intended: str, index: int) -> None:
    print(f"\n[{index}] intended {intended}")
    obs = Observation(index=index, intended=intended,
                      from_block=head_block() - 5)
    st.observations.append(obs)
    obs.job_id = hire_with_retry(intended)
    print(f"    job {obs.job_id}")
    save_state(st)

    if intended == "broken":
        print(f"    waiting {SLA_WAIT_SECONDS}s for the deadline")
        time.sleep(SLA_WAIT_SECONDS)

    obs.recorded = observe(obs.job_id, obs.from_block)
    save_state(st)
    flag = "" if obs.recorded == intended else "   <-- NOT AS INTENDED"
    print(f"    recorded {obs.recorded}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broken", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.broken < 0:
        print("--broken cannot be negative")
        return 1

    total = args.broken + 1
    mins = (120 + args.broken * (SLA_WAIT_SECONDS + 60)) / 60

    print("=" * 68)
    print(f"COORDINATION RE-RUN  1 kept + {args.broken} broken "
          f"= {total} observations")
    print(f"roughly {mins:.0f} minutes, sequential")
    print("=" * 68)

    found = strays()
    if found:
        print("\nSTRAY PROCESSES. Stop these first:")
        for s in found:
            print("   ", s[:150])
        return 1
    print("no stray agent processes")

    if args.dry_run:
        print("\nplan:")
        print("  every collection hire runs with PRIORS_NO_MEMORY=1, since the")
        print("  gate is already shut on this counterparty at 0.4587")
        print("  1. Digest DIGEST_FAULT_NEXT=0, one hire run to completion,")
        print("     expect 'kept'")
        print(f"  2. Digest DIGEST_FAULT_NEXT={args.broken}, {args.broken} "
              f"hires, each stopped once")
        print("     created, each observed after the SLA")
        print("  3. coordination_report.py, with memory on, for the verdict")
        print("\nnothing created (dry run)")
        return 0

    st = load_state(args.broken)
    digest = None
    try:
        if st.unfinished() is not None:
            # Whatever fault budget it had is gone with the old process.
            digest = start_digest(0 if st.unfinished().intended == "kept" else 1)
            finish_unfinished(st)
            kill_tree(digest)
            digest = None

        if not st.done_kept:
            digest = start_digest(0)
            one_observation(st, "kept", 1)
            kill_tree(digest)
            digest = None
            time.sleep(5)
        else:
            print("kept observation already on record, skipping")

        remaining = args.broken - st.done_broken
        if remaining > 0:
            digest = start_digest(remaining)
            for i in range(st.done_broken, args.broken):
                one_observation(st, "broken", i + 2)
    except Stalled as e:
        print(f"\nSTALLED\n{e}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted. State is saved, rerun to continue.")
        return 130
    finally:
        kill_tree(digest)
        left = strays()
        if left:
            print("\nWARNING: processes still alive, kill them by hand:")
            for s in left:
                print("   ", s[:150])

    print("\n" + "=" * 68)
    subprocess.run([sys.executable,
                    str(ROOT / "scripts" / "coordination_report.py"),
                    "--provider", PROVIDER])
    print("\nBefore committing, regenerate and update:")
    print("  python scripts/ablation.py")
    print("  README.md, docs/index.html, and the card")
    print("  and state in the README that the collection hires ran with the")
    print("  response gate bypassed, and why")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
