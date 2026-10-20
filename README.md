# rampgate

> *ramp* + *gate*: hold each job's warm-up ramp at the gate, measure what it really uses, then reserve exactly that.

rampgate is an **observation-based admission scheduler** packaged as a small, embeddable, pure-Python library
(zero runtime dependencies in the core). It decides **when** a job may start on a device the caller chose:

- **Probation:** at most one warming job per device, so ramp-ups never race.
- **Observed-peak reservation:** reserve `observed_peak × safety_margin`, growing if the peak grows.
- **OOM feedback:** an OOM grows the estimate and re-queues the job to resume from its checkpoint.

**Status:** pre-alpha, under construction (phase L2 of [SPEC.md](SPEC.md) done: `decide`, policies, estimators and a minimal simulator). Not on PyPI yet.

## Packages

| Distribution | Purpose |
|---|---|
| `rampgate` | Core: model, reducer, decide, policies, estimators, Engine, simulator. Stdlib only. |
| `rampgate-nvml` | NVIDIA GPU probe (planned) |
| `rampgate-psutil` | Host RAM/CPU probe (planned) |
| `rampgate-docker` | Docker executor and event watcher (planned) |

## Development

Everything runs in Docker; nothing needs to be installed on the host.

```sh
./dev build          # build the dev image (uv + CPython 3.11–3.13)
./dev ci             # ruff, mypy --strict, pytest+coverage on 3.11, 3.12, 3.13
./dev ci 3.12        # a single version
./dev zero-deps      # verify the core wheel has no runtime dependencies
./dev shell          # interactive shell with a synced venv
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
