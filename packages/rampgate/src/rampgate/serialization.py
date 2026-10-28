"""Lossless, versioned JSON for the reducer state (SPEC §4, I8).

Only classes from rampgate's own model modules can be decoded (an explicit
allowlist built at import time); nothing is imported dynamically from a dump.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from enum import Enum
from types import ModuleType
from typing import Any

from rampgate import actions, events, explain, model, reducer
from rampgate.model import State, frozen_map
from rampgate.resources import Resources

__all__ = [
    "SCHEMA_VERSION",
    "STATE_FORMAT",
    "UnsupportedSchema",
    "dump_state",
    "from_data",
    "load_state",
    "to_data",
]

STATE_FORMAT = "rampgate.state"
SCHEMA_VERSION = 5


def _v1_to_v2(payload: Any) -> Any:
    """v2 adds ``JobRuntime.usage_samples`` (Rule O's room at OOM); v1 dumps start empty."""
    if isinstance(payload, dict):
        out = {k: _v1_to_v2(v) for k, v in payload.items()}
        if out.get("$t") == "JobRuntime":
            out["f"].setdefault("usage_samples", {"$m": []})
        return out
    if isinstance(payload, list):
        return [_v1_to_v2(v) for v in payload]
    return payload


def _v2_to_v3(payload: Any) -> Any:
    """v3 adds the observation counts of rule A: ``JobRuntime.admitted_observations`` and
    ``Start.observations``; v2 dumps start empty (0 observations: ``safety_margin``)."""
    if isinstance(payload, dict):
        out = {k: _v2_to_v3(v) for k, v in payload.items()}
        if out.get("$t") == "JobRuntime":
            out["f"].setdefault("admitted_observations", {"$m": []})
        if out.get("$t") == "Start":
            out["f"].setdefault("observations", {"$m": []})
        return out
    if isinstance(payload, list):
        return [_v2_to_v3(v) for v in payload]
    return payload


def _v3_to_v4(payload: Any) -> Any:
    """v4 adds Rule V and the isolation retry: ``JobRuntime.admit_seq``, ``vacating`` and
    ``isolate_seq``, ``State.next_order`` and ``Stop.vacate``; v3 dumps start without them."""
    if isinstance(payload, dict):
        out = {k: _v3_to_v4(v) for k, v in payload.items()}
        if out.get("$t") == "JobRuntime":
            out["f"].setdefault("admit_seq", None)
            out["f"].setdefault("vacating", False)
            out["f"].setdefault("isolate_seq", None)
        if out.get("$t") == "State":
            out["f"].setdefault("next_order", 0)
        if out.get("$t") == "Stop":
            out["f"].setdefault("vacate", False)
        return out
    if isinstance(payload, list):
        return [_v3_to_v4(v) for v in payload]
    return payload


def _v4_to_v5(payload: Any) -> Any:
    """v5 adds the Rule S usage floor ``Device.stable_min_usage`` (off), the sample
    counters ``DeviceState.samples`` and ``JobRuntime.obs_sample``; v4 dumps start
    with the floor off and no samples counted."""
    if isinstance(payload, dict):
        out = {k: _v4_to_v5(v) for k, v in payload.items()}
        if out.get("$t") == "Device":
            out["f"].setdefault("stable_min_usage", {"$r": {}})
        if out.get("$t") == "DeviceState":
            out["f"].setdefault("samples", 0)
        if out.get("$t") == "JobRuntime":
            out["f"].setdefault("obs_sample", {"$m": []})
        return out
    if isinstance(payload, list):
        return [_v4_to_v5(v) for v in payload]
    return payload


# MIGRATIONS[v] upgrades a version-v ``state`` payload to version v + 1.
MIGRATIONS: dict[int, Callable[[Any], Any]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
    3: _v3_to_v4,
    4: _v4_to_v5,
}


class UnsupportedSchema(ValueError):
    """The dump has an unknown format, a newer schema, or no migration path."""


def _registry(*modules: ModuleType) -> dict[str, type]:
    out: dict[str, type] = {}
    for module in modules:
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and (dataclasses.is_dataclass(obj) or issubclass(obj, Enum))
            ):
                if name in out:
                    raise RuntimeError(
                        f"duplicate serializable class name {name!r}"
                    )  # pragma: no cover
                out[name] = obj
    return out


_CLASSES = _registry(model, events, actions, explain, reducer)


def to_data(value: Any) -> Any:
    """Encode a value of the state graph as JSON-compatible data."""
    if isinstance(value, Enum):  # before str: StrEnum members are also str
        return {"$e": type(value).__name__, "v": value.value}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Resources):
        return {"$r": value.to_json()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _CLASSES.get(name) is not type(value):
            raise TypeError(f"{name} is not a serializable rampgate class")
        fields = {f.name: to_data(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {"$t": name, "f": fields}
    if isinstance(value, Mapping):
        return {"$m": [[to_data(k), to_data(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"$tu": [to_data(v) for v in value]}
    if isinstance(value, frozenset):
        items = [to_data(v) for v in value]
        return {"$s": sorted(items, key=lambda x: json.dumps(x, sort_keys=True))}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def from_data(data: Any) -> Any:
    """Decode data produced by `to_data`."""
    if data is None or isinstance(data, bool | int | float | str):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"unexpected JSON value {data!r}")
    if "$r" in data:
        return Resources.from_json(data["$r"])
    if "$e" in data:
        return _cls(data["$e"])(data["v"])
    if "$t" in data:
        cls = _cls(data["$t"])
        fields = {k: from_data(v) for k, v in data["f"].items()}
        return cls(**fields)
    if "$m" in data:
        return frozen_map({from_data(k): from_data(v) for k, v in data["$m"]})
    if "$tu" in data:
        return tuple(from_data(v) for v in data["$tu"])
    if "$s" in data:
        return frozenset(from_data(v) for v in data["$s"])
    raise ValueError(f"unexpected JSON object with keys {sorted(data)}")


def _cls(name: str) -> type:
    cls = _CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown class {name!r} in dump")
    return cls


def dump_state(state: State) -> dict[str, Any]:
    """``{"format", "schema_version", "state"}``, ready for ``json.dumps``."""
    return {"format": STATE_FORMAT, "schema_version": SCHEMA_VERSION, "state": to_data(state)}


def load_state(data: Mapping[str, Any]) -> State:
    """Inverse of `dump_state`; upgrades older schema versions through `MIGRATIONS`."""
    if not isinstance(data, Mapping) or data.get("format") != STATE_FORMAT:
        raise UnsupportedSchema("not a rampgate state dump")
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise UnsupportedSchema(f"invalid schema_version {version!r}")
    if version > SCHEMA_VERSION:
        raise UnsupportedSchema(
            f"dump has schema_version {version}, this rampgate reads up to {SCHEMA_VERSION}"
        )
    payload = data["state"]
    while version < SCHEMA_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise UnsupportedSchema(f"no migration from schema_version {version}")
        payload = migrate(payload)
        version += 1
    state = from_data(payload)
    if not isinstance(state, State):
        raise UnsupportedSchema("dump does not contain a State")
    return state
