# Contributing

Thanks for helping. rampgate is small on purpose; the bar for changes is that the scheduling stays explainable and tested.

## Setup

Everything runs in Docker; nothing is installed on the host.

```sh
./dev build          # the dev image (uv + CPython 3.11–3.13) and the GPU memory-hog image
./dev ci             # ruff, mypy --strict, pytest + coverage on 3.11, 3.12, 3.13, zero-dependency check
./dev ci 3.13        # one version
./dev docs           # the docs site, strict
./dev integration    # Docker/GPU integration tests (mounts the Docker socket: root-equivalent)
```

## Rules

- **The core stays pure:** no I/O, threads, wall clock, randomness or imports outside the standard library in `packages/rampgate`. Time and randomness are passed in. The zero-dependency check enforces the last part.
- **Behaviour is specified first.** `SPEC.md` describes every rule and invariant; change it in the same pull request as the code, and say why.
- **Invariants are tested with hypothesis** (`packages/rampgate/tests/test_invariants.py`). A bug found by hypothesis becomes an explicit `@example` and a regression test.
- **Benchmark results are not tuned.** Criteria are fixed before measuring; a held-out seed set is run once per frozen configuration; changes after results were seen are listed in the README's Methodology. A change that should not affect the benchmark is proven with `benchmarks/trace_digest.py` (identical per-run action sequences).
- **Conventional commits**, one logical change per commit. Update the package's `CHANGELOG.md`.
- `mypy --strict` and ruff are part of CI; coverage of the core stays ≥ 95 %.

## Releases

Each distribution is released from its own tag: `rampgate-v0.1.0`, `rampgate-nvml-v0.1.0`, `rampgate-psutil-v0.1.0`, `rampgate-docker-v0.1.0`. The release workflow builds that package, checks that the tag matches its version, and publishes it to PyPI with trusted publishing. Run `./dev release-check` first.
