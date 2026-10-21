# Changelog — repository

## Unreleased

### Added
- `benchmarks/` (L4): scenarios (a)-(e), `run.py` (parallel grid, warm runs, median/IQR with censoring, `results.json`/`results.md`, `--check` for SPEC §12 L4), `plots.py`; a `bench` dependency group (matplotlib); a CI smoke test and a nightly benchmark job.
- GitHub Actions CI workflow wrapping `./dev ci` on Python 3.11–3.13, plus the zero-dependency check (L2).
- Nightly workflow running the thorough `nightly` hypothesis profile; `./dev` forwards `HYPOTHESIS_PROFILE` (L2 follow-up).
- `docs/reconciliation.md`: restarting and reconciling an Engine (L3).
