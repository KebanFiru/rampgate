"""Check the L5 real-hardware gates on a run log of `docker_gpu_queue.py` (SPEC §12 L5).

Usage::

    python examples/check_run_log.py docs/real-run/runs/<run>/run.jsonl [--gates]

Without ``--gates`` (an unstaged run) it only reports. With ``--gates`` it exits
1 unless every gate holds:

- **Probation:** an untrusted job is held (``WaitingForWarmup``) while another
  untrusted job warms on the GPU, and starts only after that job is STABLE.
- **Rule V:** ``Stop(vacate=True)`` → the job exits → ``Requeue(VACATED)`` → a
  later ``Start`` of the same job with ``resume=True``.
- **OOM → Rule O:** ``JobExited(OOM)`` → ``Requeue(OOM)`` whose ``new_estimate``
  is larger than the estimate at the failed ``Start`` → a later attempt of that
  job SUCCEEDS.

It also reports the measured items: who died on each OOM and how it was
classified, whose estimate grew, Rule V vacates, ``Overcommitted`` warnings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rampgate import ExitKind, JobExited, Overcommitted, Requeue, RequeueReason, Start, Stop, Warn
from rampgate.explain import WaitingForWarmup
from rampgate.serialization import from_data


@dataclass(frozen=True)
class Record:
    t: float
    kind: str
    data: Any


def read(path: Path) -> list[Record]:
    out: list[Record] = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        data = raw["data"]
        if raw["kind"] in ("event", "action"):
            data = from_data(data)
        elif raw["kind"] == "explanation":
            data = {"job_id": data["job_id"], "explanation": from_data(data["explanation"])}
        out.append(Record(float(raw["t"]), raw["kind"], data))
    return out


def actions(log: list[Record]) -> Iterator[tuple[float, Any]]:
    for r in log:
        if r.kind == "action":
            yield r.t, r.data


def states(log: list[Record]) -> Iterator[tuple[float, str, str, int]]:
    for r in log:
        if r.kind == "state":
            yield r.t, r.data["job_id"], r.data["state"], int(r.data["attempt"])


def starts(log: list[Record]) -> list[tuple[float, Start]]:
    return [(t, a) for t, a in actions(log) if isinstance(a, Start)]


def _vram(value: Any) -> float:
    return 0.0 if value is None else float(value["vram"])


def gate_probation(log: list[Record]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    all_starts = starts(log)
    first_start: dict[str, tuple[float, Start]] = {}
    for t, s in all_starts:
        first_start.setdefault(s.job_id, (t, s))
    stable_at: dict[tuple[str, int], float] = {}
    for t, job, state, attempt in states(log):
        if state == "STABLE":
            stable_at.setdefault((job, attempt), t)
    for r in log:
        if r.kind != "explanation":
            continue
        expl = r.data["explanation"]
        if not isinstance(expl, WaitingForWarmup):
            continue
        held, warming = r.data["job_id"], expl.warming_job_id
        warm_start = next((s for t, s in all_starts if s.job_id == warming and t <= r.t), None)
        if warm_start is None or expl.device_id in warm_start.trusted:
            continue  # the warming job must be untrusted
        held_start = first_start.get(held)
        stable = stable_at.get((warming, warm_start.attempt))
        if held_start is None or stable is None:
            continue
        held_untrusted = expl.device_id not in held_start[1].trusted
        if held_untrusted and held_start[0] >= stable:
            notes.append(
                f"{held} held at t={r.t:.0f} (WaitingForWarmup on {warming}#{warm_start.attempt}); "
                f"{warming} STABLE at t={stable:.0f}; {held} started at t={held_start[0]:.0f}"
            )
            return True, notes
    notes.append("no untrusted job was held behind an untrusted warming job")
    return False, notes


def gate_vacate(log: list[Record]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    events = [(r.t, r.data) for r in log if r.kind == "event"]
    acts = list(actions(log))
    for t, stop in acts:
        if not (isinstance(stop, Stop) and stop.vacate):
            continue
        exited = next(
            (
                (te, e)
                for te, e in events
                if isinstance(e, JobExited)
                and e.job_id == stop.job_id
                and e.attempt == stop.attempt
                and te >= t
            ),
            None,
        )
        requeue = next(
            (
                (tr, a)
                for tr, a in acts
                if isinstance(a, Requeue)
                and a.job_id == stop.job_id
                and a.reason is RequeueReason.VACATED
                and tr >= t
            ),
            None,
        )
        restart = next(
            (
                (ts, s)
                for ts, s in acts
                if isinstance(s, Start) and s.job_id == stop.job_id and s.attempt > stop.attempt
            ),
            None,
        )
        exit_note = "none" if exited is None else f"{exited[1].kind.value} at t={exited[0]:.0f}"
        requeue_note = "none" if requeue is None else f"t={requeue[0]:.0f}"
        restart_note = (
            "none"
            if restart is None
            else f"#{restart[1].attempt} at t={restart[0]:.0f} resume={restart[1].resume}"
        )
        notes.append(
            f"Stop(vacate=True) {stop.job_id}#{stop.attempt} at t={t:.0f}; exit: {exit_note}; "
            f"Requeue(VACATED): {requeue_note}; restart: {restart_note}"
        )
        if exited and requeue and restart and restart[1].resume and requeue[0] >= exited[0] - 1e-9:
            return True, notes
    if not notes:
        notes.append("no Stop(vacate=True) in the log")
    return False, notes


def gate_oom(log: list[Record]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    events = [(r.t, r.data) for r in log if r.kind == "event"]
    acts = list(actions(log))
    start_of = {(s.job_id, s.attempt): s for _, s in acts if isinstance(s, Start)}
    succeeded = {job for _, job, state, _ in states(log) if state == "SUCCEEDED"}
    ok = False
    for t, e in events:
        if not (isinstance(e, JobExited) and e.kind is ExitKind.OOM):
            continue
        requeue = next(
            (
                a
                for _, a in acts
                if isinstance(a, Requeue)
                and a.job_id == e.job_id
                and a.reason is RequeueReason.OOM
                and a.next_attempt == e.attempt + 1
            ),
            None,
        )
        start = start_of.get((e.job_id, e.attempt))
        before = None if start is None else start.estimate.get("gpu0")
        after = None if requeue is None else requeue.new_estimate.get("gpu0")
        grew = requeue is not None and _vram(after) > _vram(before)
        notes.append(
            f"OOM exit {e.job_id}#{e.attempt} at t={t:.0f}: estimate at Start "
            f"{'unknown' if before is None else f'{_vram(before):.0f} MiB'}"
            f"{' (trusted)' if start is not None and 'gpu0' in start.trusted else ''}, "
            f"Requeue(OOM) new_estimate "
            f"{'none' if requeue is None else f'{_vram(after):.0f} MiB'}; "
            f"later attempt SUCCEEDED: {e.job_id in succeeded}"
        )
        ok = ok or (grew and e.job_id in succeeded)
    if not notes:
        notes.append("no OOM exit in the log")
    return ok, notes


def report(log: list[Record]) -> list[str]:
    lines: list[str] = []
    meta = next(r.data for r in log if r.kind == "meta")
    lines.append(f"run {meta['run_id']}, commit {meta['commit']}, rampgate {meta['rampgate']}")
    finished = next((r for r in log if r.kind == "finished"), None)
    lines.append(
        f"finished: {finished.data['reason'] if finished else 'no'} at t={finished.t:.0f}"
        if finished
        else "finished: no"
    )
    final: dict[str, tuple[str, int]] = {}
    for _, job, state, attempt in states(log):
        final[job] = (state, attempt)
    lines.append(
        "final states: " + ", ".join(f"{j}={s}#{a}" for j, (s, a) in sorted(final.items()))
    )
    acts = list(actions(log))
    vacates = sum(isinstance(a, Stop) and a.vacate for _, a in acts)
    over = sum(isinstance(a, Warn) and isinstance(a.warning, Overcommitted) for _, a in acts)
    requeues = [a for _, a in acts if isinstance(a, Requeue)]
    lines.append(
        f"Rule V vacates: {vacates}; Overcommitted warnings: {over}; requeues: "
        + (
            ", ".join(f"{a.job_id}->{a.next_attempt} ({a.reason.value})" for a in requeues)
            or "none"
        )
    )
    for r in log:
        if r.kind == "exit_classified":
            d = r.data
            if d["kind"] in ("OOM", "FAILURE"):
                tail = [ln for ln in d["log_tail"].splitlines() if "out of memory" in ln.lower()]
                lines.append(
                    f"exit {d['job_id']}#{d['attempt']} at t={r.t:.0f}: code {d['exit_code']}, "
                    f"OOMKilled {d['oom_killed']}, classified {d['kind']}"
                    + (f"; log: {tail[-1]}" if tail else "")
                )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--gates", action="store_true", help="exit 1 unless every gate holds")
    args = parser.parse_args(argv)
    log = read(args.log)
    for line in report(log):
        print(line)
    print()
    results = []
    for name, gate in (
        ("Probation", gate_probation),
        ("Rule V", gate_vacate),
        ("OOM -> Rule O", gate_oom),
    ):
        ok, notes = gate(log)
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for note in notes:
            print(f"    {note}")
    if args.gates and not all(results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
