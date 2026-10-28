# rampgate-psutil

Host RAM/CPU probe for rampgate (psutil).

Part of the [rampgate](../../README.md) monorepo.

```python
from rampgate_psutil import HostProbe

probe = HostProbe("cpu", {"ram": "GiB", "cpu": "cores"})
engine.add_device(probe.device())           # capacity: total RAM, logical cores
engine.observe(probe.sample(t))             # DeviceObserved: total - available RAM, busy cores
usage = probe.process_tree_usage(pid)       # RSS sum and cores of pid and its descendants
```

CPU usage is measured between two calls, so the first sample of the host or of a process reports 0 cores.
