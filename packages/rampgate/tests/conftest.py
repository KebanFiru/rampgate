"""Hypothesis profiles: ``ci`` (default, fast) and ``nightly`` (thorough).

Select with ``HYPOTHESIS_PROFILE=nightly``. Expensive tests scale their
example counts from the active profile through ``rampgate_testkit.examples``.
"""

import os

from hypothesis import HealthCheck, settings

_COMMON = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}
settings.register_profile("ci", max_examples=50, **_COMMON)  # type: ignore[arg-type]
settings.register_profile("nightly", max_examples=300, **_COMMON)  # type: ignore[arg-type]
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
