"""A fake Docker client: just enough of docker-py for the adapter's unit tests."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

_ids = itertools.count(1)


@dataclass
class FakeContainer:
    image: str
    command: Any
    kwargs: dict[str, Any]
    id: str = field(default_factory=lambda: f"{next(_ids):064x}")
    started: bool = False
    removed: bool = False
    stop_timeouts: list[float | None] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    log: bytes = b""

    @property
    def labels(self) -> dict[str, str]:
        labels: dict[str, str] = self.kwargs.get("labels", {})
        return labels

    @property
    def attrs(self) -> dict[str, Any]:
        return {"State": self.state}

    def start(self) -> None:
        self.started = True

    def stop(self, *, timeout: float | None = None) -> None:
        self.stop_timeouts.append(timeout)

    def remove(self, *, v: bool = False, link: bool = False, force: bool = False) -> None:
        self.removed = True

    def logs(self, *, tail: int | Literal["all"] = "all") -> bytes:
        if tail == "all":
            return self.log
        return b"\n".join(self.log.split(b"\n")[-tail:])


class NotFoundError(Exception):
    pass


@dataclass
class FakeContainers:
    all: list[FakeContainer] = field(default_factory=list)

    def create(self, image: str, command: Any = None, **kwargs: Any) -> FakeContainer:
        container = FakeContainer(image, command, kwargs)
        self.all.append(container)
        return container

    def get(self, container_id: str) -> FakeContainer:
        for c in self.all:
            if c.id == container_id and not c.removed:
                return c
        raise NotFoundError(container_id)

    def list(
        self, *, all: bool = False, filters: dict[str, Any] | None = None
    ) -> list[FakeContainer]:
        wanted = (filters or {}).get("label", [])
        wanted = [wanted] if isinstance(wanted, str) else wanted
        out = []
        for c in self.all:
            if c.removed:
                continue
            pairs = [w.split("=", 1) for w in wanted]
            if all_match(c.labels, pairs):
                out.append(c)
        return out


def all_match(labels: dict[str, str], pairs: list[list[str]]) -> bool:
    for pair in pairs:
        if pair[0] not in labels:
            return False
        if len(pair) == 2 and labels[pair[0]] != pair[1]:
            return False
    return True


@dataclass
class FakeClient:
    containers: FakeContainers = field(default_factory=FakeContainers)
    stream: list[dict[str, Any]] = field(default_factory=list)
    event_filters: list[dict[str, Any] | None] = field(default_factory=list)

    def events(self, *, decode: Literal[True], filters: dict[str, Any] | None = None) -> Any:
        self.event_filters.append(filters)
        return iter(self.stream)


def docker_event(action: str, container: FakeContainer, **attributes: str) -> dict[str, Any]:
    return {
        "Type": "container",
        "Action": action,
        "id": container.id,
        "Actor": {"ID": container.id, "Attributes": {**container.labels, **attributes}},
    }
