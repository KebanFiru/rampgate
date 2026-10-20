"""`Resources`: an immutable, non-negative vector over free-form dimension names."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeAlias

__all__ = ["GiB", "KiB", "MiB", "Number", "Resources", "TiB"]

Number: TypeAlias = int | float

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4

_EPS_REL = 1e-9


def _check_value(dim: object, value: object) -> Number:
    if not isinstance(dim, str) or not dim:
        raise TypeError(f"dimension names must be non-empty strings, got {dim!r}")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"value for {dim!r} must be int or float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"value for {dim!r} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"value for {dim!r} must be non-negative, got {value!r}")
    return value


class Resources(Mapping[str, Number]):
    """Immutable mapping from dimension name to a non-negative number.

    A missing dimension means 0: ``r["anything"]`` returns 0 for unknown
    dimensions, and zero-valued dimensions are dropped on construction, so
    ``Resources(vram=0) == Resources()``. Iteration, ``len`` and ``in`` only
    see the non-zero dimensions, in sorted order.
    """

    __slots__ = ("_hash", "_items", "_values")

    _items: tuple[tuple[str, Number], ...]
    _values: dict[str, Number]
    _hash: int

    def __init__(
        self,
        mapping: Mapping[str, Number] | Iterable[tuple[str, Number]] | None = None,
        /,
        **dims: Number,
    ) -> None:
        pairs: list[tuple[str, Number]] = []
        if mapping is not None:
            pairs.extend(mapping.items() if isinstance(mapping, Mapping) else mapping)
        pairs.extend(dims.items())
        values: dict[str, Number] = {}
        for dim, value in pairs:
            checked = _check_value(dim, value)
            if dim in values:
                raise ValueError(f"dimension {dim!r} given twice")
            values[dim] = checked
        self._items = tuple(sorted((d, v) for d, v in values.items() if v != 0))
        self._values = dict(self._items)
        self._hash = hash(self._items)

    def __getitem__(self, dim: str) -> Number:
        return self._values.get(dim, 0)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, dim: object) -> bool:
        return dim in self._values

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Resources):
            return self._items == other._items
        return super().__eq__(other)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        if all(d.isidentifier() for d, _ in self._items):
            inner = ", ".join(f"{d}={v!r}" for d, v in self._items)
            return f"Resources({inner})"
        return f"Resources({self._values!r})"

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, "_hash"):
            raise AttributeError("Resources is immutable")
        object.__setattr__(self, name, value)

    def __reduce__(self) -> tuple[type[Resources], tuple[dict[str, Number]]]:
        return (Resources, (dict(self._values),))

    def _dims(self, other: Resources) -> list[str]:
        return sorted(self._values.keys() | other._values.keys())

    def __add__(self, other: Resources) -> Resources:
        if not isinstance(other, Resources):
            return NotImplemented
        return Resources({d: self[d] + other[d] for d in self._dims(other)})

    def __sub__(self, other: Resources) -> Resources:
        if not isinstance(other, Resources):
            return NotImplemented
        return self.sub(other)

    def sub(self, other: Resources, *, clamp: bool = False) -> Resources:
        """Component-wise ``self - other``.

        A result in ``[-eps, 0)`` with ``eps = 1e-9 * max(|a|, |b|, 1)`` snaps
        to 0. Anything more negative raises ``ValueError``, or is clamped to 0
        when ``clamp=True``.
        """
        out: dict[str, Number] = {}
        for d in self._dims(other):
            a, b = self[d], other[d]
            diff = a - b
            if diff < 0:
                eps = _EPS_REL * max(abs(a), abs(b), 1)
                if diff < -eps and not clamp:
                    raise ValueError(
                        f"Resources subtraction would go negative in {d!r}: "
                        f"{a!r} - {b!r} = {diff!r}"
                    )
                diff = 0
            out[d] = diff
        return Resources(out)

    def scale(self, k: Number) -> Resources:
        """Multiply every dimension by ``k`` (finite, non-negative)."""
        if isinstance(k, bool) or not isinstance(k, int | float):
            raise TypeError(f"scale factor must be int or float, got {type(k).__name__}")
        if not math.isfinite(k) or k < 0:
            raise ValueError(f"scale factor must be finite and non-negative, got {k!r}")
        return Resources({d: v * k for d, v in self._items})

    def max(self, other: Resources) -> Resources:
        """Component-wise maximum. Also usable as ``Resources.max(a, b)``."""
        return Resources({d: max(self[d], other[d]) for d in self._dims(other)})

    def min(self, other: Resources) -> Resources:
        """Component-wise minimum. Also usable as ``Resources.min(a, b)``."""
        return Resources({d: min(self[d], other[d]) for d in self._dims(other)})

    def fits_in(self, other: Resources) -> bool:
        """True iff every dimension of ``self`` is ``<=`` the same dimension of ``other``."""
        return all(v <= other[d] for d, v in self._items)

    def exceeds(self, other: Resources) -> list[str]:
        """Sorted dimensions in which ``self`` is strictly greater than ``other``."""
        return [d for d, v in self._items if v > other[d]]

    @property
    def is_zero(self) -> bool:
        return not self._items

    def to_json(self) -> dict[str, Number]:
        """A plain ``dict`` (sorted, zeros omitted) suitable for ``json.dumps``."""
        return dict(self._items)

    @classmethod
    def from_json(cls, data: object) -> Resources:
        if not isinstance(data, Mapping):
            raise TypeError(f"expected a JSON object, got {type(data).__name__}")
        return cls(data)
