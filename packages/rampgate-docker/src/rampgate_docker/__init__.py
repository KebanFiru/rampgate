"""Docker executor, exit classification and event watcher for rampgate (SPEC §9)."""

from rampgate_docker.classify import DEFAULT_OOM_PATTERNS, classify_exit
from rampgate_docker.executor import (
    ATTEMPT_LABEL,
    JOB_LABEL,
    ContainerLike,
    ContainerSpec,
    DockerClientLike,
    DockerExecutor,
)
from rampgate_docker.mapper import ContainerUsageMapper, ProcessSample
from rampgate_docker.watcher import DockerEventWatcher

__version__ = "0.1.0"

__all__ = [
    "ATTEMPT_LABEL",
    "DEFAULT_OOM_PATTERNS",
    "JOB_LABEL",
    "ContainerLike",
    "ContainerSpec",
    "ContainerUsageMapper",
    "DockerClientLike",
    "DockerEventWatcher",
    "DockerExecutor",
    "ProcessSample",
    "classify_exit",
]
