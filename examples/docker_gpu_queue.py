"""Run memory-hog containers on one real GPU through rampgate's Engine (SPEC §12 L5).

Usage (inside ``./dev integration``, which mounts the Docker socket and the GPU)::

    ./dev integration python examples/docker_gpu_queue.py examples/scenarios/staged.json

Each tick (every ``--tick`` seconds of wall time):

1. container events (`DockerEventWatcher`) → ``engine.apply``;
2. an NVML sample (`NvmlProbe`): the GPU's total usage → ``DeviceObserved``, and
   per-process usage mapped to managed containers (`ContainerUsageMapper`) →
   ``JobObserved``;
3. due submissions → ``engine.submit``;
4. ``engine.tick(now)``; its actions → `DockerExecutor`.

Everything goes to a JSONL log (``--out``/``run.jsonl``): every event, action,
explanation change and job state change, the raw probe samples, the executor's
own log (when a SIGTERM was actually sent) and each exit's classification. The
hogs' full logs are saved next to it. `check_run_log.py` checks the L5 gates on it.

The Engine runs with the default `Config` of the commit under test. A scenario
may stage conditions only through inputs: the hogs' workload profiles, submit
times or conditions, and a pre-filled history store; the scenario file is
copied into the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rampgate import (
    Config,
    DeviceObserved,
    Engine,
    HistoryEstimator,
    InMemoryHistoryStore,
    JobExited,
    JobObserved,
    Resources,
    Start,
    __version__,
)
from rampgate.model import Job, JobState
from rampgate.policies import FifoPolicy
from rampgate.serialization import to_data
from rampgate_docker import ContainerSpec, ContainerUsageMapper, DockerEventWatcher, DockerExecutor
from rampgate_nvml import NvmlProbe, pid_to_container_id

DEVICE = "gpu0"
RUN_LABEL = "rampgate.run"
TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.PREEMPTED}


class Log:
    def __init__(self, path: Path, clock: Any) -> None:
        self._file = path.open("w")
        self._clock = clock

    def write(self, kind: str, data: Any, t: float | None = None) -> None:
        record = {
            "t": round(self._clock() if t is None else t, 3),
            "wall": round(time.time(), 3),
            "kind": kind,
            "data": data,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _git_commit() -> str:
    # ./dev integration passes the host's commit (the dev image has no git).
    if os.environ.get("RAMPGATE_GIT_COMMIT"):
        return os.environ["RAMPGATE_GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _submission_due(spec: Mapping[str, Any], now: float, engine: Engine) -> bool:
    if "submit_when" in spec:
        cond = spec["submit_when"]
        rt = engine.state.jobs.get(cond["job"])
        return (
            rt is not None
            and rt.state.value == cond["state"]
            and rt.attempt >= int(cond.get("min_attempt", 1))
        )
    return now >= float(spec.get("submit_at", 0))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - the loop, top to bottom
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tick", type=float, default=1.0)
    parser.add_argument("--image", default="rampgate-hog:dev")
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    run_id = f"{scenario['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = args.out or Path("docs/real-run/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "hog-logs").mkdir(exist_ok=True)
    shutil.copy(args.scenario, out / "scenario.json")
    ckpt = out / "ckpt"
    ckpt.mkdir(exist_ok=True)
    # The hogs are started by the host's Docker daemon: bind mounts need host paths.
    host_root = Path(os.environ.get("RAMPGATE_HOST_ROOT", Path.cwd()))
    host_ckpt = host_root / ckpt.resolve().relative_to(Path.cwd().resolve())

    t0 = time.monotonic()

    def clock() -> float:
        return time.monotonic() - t0

    log = Log(out / "run.jsonl", clock)
    config = Config()
    store = InMemoryHistoryStore()
    estimator = HistoryEstimator(store)
    probe = NvmlProbe({0: DEVICE}, unit="MiB")
    (device,) = probe.devices()

    # Staged input: a pre-filled history store (disclosed in the scenario file).
    for i, item in enumerate(scenario.get("history", [])):
        for n in range(int(item.get("records", 1))):
            store.put(
                item["key"],
                estimator.device_kind(device),
                Resources({"vram": float(item["peak_mib"])}),
                lower_bound=False,
                record_id=f"prefill-{i}-{n}",
            )

    jobs_spec = {j["id"]: j for j in scenario["jobs"]}
    jobs = {
        jid: Job(
            jid,
            {DEVICE: None if j.get("hint_mib") is None else Resources(vram=float(j["hint_mib"]))},
            fingerprint=tuple(j.get("fingerprint", [])),
        )
        for jid, j in jobs_spec.items()
    }

    def spec_for(job: Job, start: Start) -> ContainerSpec:
        env = {
            k: (v if isinstance(v, str) else json.dumps(v))
            for k, v in jobs_spec[job.id]["env"].items()
        }
        env["HOG_CKPT_DIR"] = "/ckpt"
        return ContainerSpec(
            args.image,
            environment=env,
            gpus="all",
            labels={RUN_LABEL: run_id},
            extra={
                "user": f"{os.getuid()}:{os.getgid()}",
                "volumes": {str(host_ckpt): {"bind": "/ckpt", "mode": "rw,z"}},
            },
        )

    last_used: dict[str, float] = {}

    def pressure(job_id: str, attempt: int) -> bool:
        return last_used.get(DEVICE, 0.0) >= 0.95 * device.capacity["vram"]

    executor = DockerExecutor(
        spec_for,
        log=lambda name, fields: log.write("executor", {"name": name, **fields}),
        clock=clock,
    )
    watcher = DockerEventWatcher(
        executor.client,
        clock,
        stop_reason=executor.stop_reason,
        memory_pressure=pressure,
        on_exit=lambda record: log.write("exit_classified", dict(record)),
    )
    mapper = ContainerUsageMapper(executor.client, pid_to_container_id)
    engine = Engine([device], FifoPolicy(strict=False), estimator, config)

    log.write(
        "meta",
        {
            "run_id": run_id,
            "scenario": scenario,
            "rampgate": __version__,
            "commit": _git_commit(),
            "python": sys.version,
            "config": to_data_config(config),
            "device": {"id": device.id, "capacity_mib": device.capacity["vram"]},
            "tick_s": args.tick,
        },
        t=0.0,
    )

    pending = dict(jobs_spec)
    explanations: dict[str, Any] = {}
    states: dict[str, tuple[str, int]] = {}
    max_time = float(scenario.get("max_time", 1800))
    watcher.start()
    exit_code = 0
    try:
        next_tick = 0.0
        while True:
            now = clock()
            for event in watcher.drain():
                log.write("event", to_data(event), t=event.t)
                engine.apply(event)
                if isinstance(event, JobExited):
                    _save_hog_log(executor, event.job_id, event.attempt, out)

            observed, processes = probe.sample(now)
            for obs in observed:
                last_used[obs.device_id] = obs.used_total["vram"]
            job_obs = mapper.map(processes, now)
            log.write(
                "probe",
                {
                    "used_total_mib": {o.device_id: o.used_total["vram"] for o in observed},
                    "processes": [
                        {
                            "pid": p.pid,
                            "mib": p.usage["vram"],
                            "container": pid_to_container_id(p.pid),
                        }
                        for p in processes
                    ],
                    "jobs": {f"{o.job_id}#{o.attempt}": o.usage["vram"] for o in job_obs},
                },
                t=now,
            )
            for event2 in [*observed, *job_obs]:
                assert isinstance(event2, DeviceObserved | JobObserved)
                engine.observe(event2)

            for jid in [j for j, spec in pending.items() if _submission_due(spec, now, engine)]:
                del pending[jid]
                engine.submit(jobs[jid], t=now)
                log.write("submitted", {"job_id": jid}, t=now)

            decision = engine.tick(now)
            for action in decision.actions:
                log.write("action", to_data(action), t=now)
            for ignored in decision.ignored_events:
                log.write("ignored", to_data(ignored), t=now)
            for jid, expl in engine.explanations.items():
                data = to_data(expl)
                if explanations.get(jid) != data:
                    explanations[jid] = data
                    log.write("explanation", {"job_id": jid, "explanation": data}, t=now)
            for jid, rt in engine.state.jobs.items():
                key = (rt.state.value, rt.attempt)
                if states.get(jid) != key:
                    states[jid] = key
                    log.write(
                        "state",
                        {"job_id": jid, "state": rt.state.value, "attempt": rt.attempt},
                        t=now,
                    )
            executor.apply(decision.actions, {j: rt.job for j, rt in engine.state.jobs.items()})

            done = not pending and all(rt.state in TERMINAL for rt in engine.state.jobs.values())
            if done:
                log.write("finished", {"reason": "all jobs terminal"})
                break
            if now >= max_time:
                log.write("finished", {"reason": "max_time"})
                exit_code = 1
                break
            next_tick += args.tick
            time.sleep(max(0.0, next_tick - clock()))
    finally:
        watcher.stop()
        for c in executor.client.containers.list(
            all=True, filters={"label": f"{RUN_LABEL}={run_id}"}
        ):
            c.remove(force=True)
        executor.close()
        probe.close()
        log.write("errors", {"watcher": [repr(e) for e in watcher.errors]})
        log.close()
    print(out)
    return exit_code


def to_data_config(config: Config) -> dict[str, Any]:
    from dataclasses import asdict  # noqa: PLC0415

    return {
        k: (v if isinstance(v, int | float | str | bool) else repr(v))
        for k, v in asdict(config).items()
    }


def _save_hog_log(executor: DockerExecutor, job_id: str, attempt: int, out: Path) -> None:
    container_id = executor.container_id(job_id, attempt)
    if container_id is None:
        return
    try:
        logs = executor.client.containers.get(container_id).logs()
    except Exception:
        return
    (out / "hog-logs" / f"{job_id}-{attempt}.log").write_bytes(logs)


if __name__ == "__main__":
    raise SystemExit(main())
