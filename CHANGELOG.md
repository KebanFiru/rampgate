# Changelog — repository

## Unreleased

### Added
- 0.1.0 packaging for all four distributions: version 0.1.0, adapters pinned to `rampgate>=0.1,<0.2`, project URLs, classifiers, PyPI-ready READMEs; `examples/quickstart.py` (the README quickstart); `CONTRIBUTING.md`; `.github/workflows/release.yml` (tag-driven, one distribution per tag, PyPI trusted publishing; not run yet); `./dev release-check` (builds all four, `twine check`, installs from the wheels in clean venvs, runs the quickstart) (L6).
- Docs site (mkdocs-material, `./dev docs`) and `benchmarks/trace_digest.py` (L6).
- L5 real-hardware demonstration: `examples/docker_gpu_queue.py` (one real GPU through the Engine, JSONL log), `examples/check_run_log.py` (the L5 gates), `examples/measure_preempt.py` (start-up and SIGTERM → exit / → memory-freed latency), staged and unstaged scenarios, and the recorded runs and `docs/real-run/assumptions.md`.
- `./dev integration`: Docker/GPU integration tests or a command, with the Docker socket, the host PID and cgroup namespaces and the GPUs (never used by `./dev ci`); `./dev build` also builds the GPU memory-hog image `docker/hog` (CUDA driver API via ctypes, no toolkit).
- Dev dependencies `types-docker`, `types-psutil`; pytest marker `docker`.
- Benchmark: OOM kills per job (max, distribution) and a `--check` gate: no full-rampgate job OOMs twice on the same device.
- Benchmark: first-contact OOMs split into `unknown` and `under-declared hint` (SPEC §12 L4 clarification, written after results); `--check` treats every non-first-contact OOM of a full-rampgate row as a bug in every scenario.
- `benchmarks/` (L4): scenarios (a)-(e), `run.py` (parallel grid, warm runs, median/IQR with censoring, `results.json`/`results.md`, `--check` for SPEC §12 L4), `plots.py`; a `bench` dependency group (matplotlib); a CI smoke test and a nightly benchmark job.
- GitHub Actions CI workflow wrapping `./dev ci` on Python 3.11–3.13, plus the zero-dependency check (L2).
- Nightly workflow running the thorough `nightly` hypothesis profile; `./dev` forwards `HYPOTHESIS_PROFILE` (L2 follow-up).
- `docs/reconciliation.md`: restarting and reconciling an Engine (L3).
