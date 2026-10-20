import copy
import json
import math
import pickle

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rampgate import GiB, Resources

DIMS = st.sampled_from(["vram", "ram", "cpu", "bw", "a/b"])
VALUES = st.one_of(
    st.integers(min_value=0, max_value=10**15),
    st.floats(min_value=0, max_value=1e15, allow_nan=False, allow_infinity=False),
)
RES = st.dictionaries(DIMS, VALUES, max_size=5).map(Resources)


def test_construct_kwargs_and_mapping() -> None:
    r = Resources({"nvidia.com/gpu": 1}, vram=24 * GiB)
    assert r["vram"] == 24 * GiB
    assert r["nvidia.com/gpu"] == 1
    assert Resources([("cpu", 2)]) == Resources(cpu=2)


def test_missing_dimension_is_zero() -> None:
    r = Resources(vram=1)
    assert r["ram"] == 0
    assert "ram" not in r
    assert "vram" in r


def test_zero_dims_are_dropped() -> None:
    assert Resources(vram=0) == Resources()
    assert hash(Resources(vram=0, cpu=1)) == hash(Resources(cpu=1))
    assert len(Resources(vram=0, cpu=1)) == 1
    assert list(Resources(b=1, a=2, c=0)) == ["a", "b"]


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"vram": -1}, ValueError),
        ({"vram": math.nan}, ValueError),
        ({"vram": math.inf}, ValueError),
        ({"vram": True}, TypeError),
        ({"vram": "1"}, TypeError),
    ],
)
def test_invalid_values(kwargs: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        Resources(kwargs)  # type: ignore[arg-type]


def test_invalid_dimension_names() -> None:
    with pytest.raises(TypeError):
        Resources({"": 1})
    with pytest.raises(TypeError):
        Resources({1: 1})  # type: ignore[arg-type]


def test_duplicate_dimension() -> None:
    with pytest.raises(ValueError, match="twice"):
        Resources({"vram": 1}, vram=2)


def test_immutable() -> None:
    r = Resources(vram=1)
    with pytest.raises(AttributeError):
        r._items = ()


def test_repr() -> None:
    assert repr(Resources(vram=1, cpu=2.5)) == "Resources(cpu=2.5, vram=1)"
    assert repr(Resources({"a/b": 1})) == "Resources({'a/b': 1})"
    assert repr(Resources()) == "Resources()"


def test_equality_with_plain_mapping() -> None:
    assert Resources(vram=1) == {"vram": 1}
    assert Resources(vram=1) != {"vram": 2}
    assert Resources(vram=1) != 1


def test_add() -> None:
    assert Resources(vram=1, cpu=2) + Resources(vram=3, ram=4) == Resources(vram=4, cpu=2, ram=4)


def test_sub_exact() -> None:
    assert Resources(vram=5, cpu=2) - Resources(vram=3) == Resources(vram=2, cpu=2)
    assert Resources(vram=5) - Resources(vram=5) == Resources()


def test_sub_raises_naming_dimension_and_values() -> None:
    with pytest.raises(ValueError, match=r"'vram'.*3.*5"):
        Resources(vram=3) - Resources(vram=5)
    with pytest.raises(ValueError, match="'cpu'"):
        Resources(vram=3) - Resources(cpu=1)


def test_sub_clamp() -> None:
    assert Resources(vram=3, cpu=4).sub(Resources(vram=5, cpu=1), clamp=True) == Resources(cpu=3)


def test_sub_float_tolerance_snaps_to_zero() -> None:
    a = Resources(vram=1.0)
    b = Resources(vram=1.0 + 1e-12)
    assert a - b == Resources()
    big = 80 * GiB
    assert Resources(vram=big) - Resources(vram=big * (1 + 5e-10)) == Resources()
    with pytest.raises(ValueError):
        Resources(vram=big) - Resources(vram=big * (1 + 1e-8))


def test_sub_tolerance_floor_is_absolute_for_small_values() -> None:
    assert Resources(x=0.0) - Resources(x=5e-10) == Resources()
    with pytest.raises(ValueError):
        Resources() - Resources(x=2e-9)


def test_sub_operator_rejects_non_resources() -> None:
    with pytest.raises(TypeError):
        Resources(vram=1) - {"vram": 1}  # type: ignore[operator]
    with pytest.raises(TypeError):
        Resources(vram=1) + {"vram": 1}  # type: ignore[operator]


def test_scale() -> None:
    assert Resources(vram=10, cpu=2).scale(1.5) == Resources(vram=15.0, cpu=3.0)
    assert Resources(vram=10).scale(0) == Resources()
    with pytest.raises(ValueError):
        Resources(vram=1).scale(-1)
    with pytest.raises(ValueError):
        Resources(vram=1).scale(math.inf)
    with pytest.raises(TypeError):
        Resources(vram=1).scale(True)


def test_max_min() -> None:
    a, b = Resources(vram=1, cpu=5), Resources(vram=3, ram=2)
    assert a.max(b) == Resources(vram=3, cpu=5, ram=2)
    assert Resources.max(a, b) == a.max(b)
    assert a.min(b) == Resources(vram=1)


def test_fits_in_and_exceeds() -> None:
    cap = Resources(vram=10, cpu=4)
    assert Resources(vram=10).fits_in(cap)
    assert Resources().fits_in(cap)
    assert not Resources(vram=11).fits_in(cap)
    assert not Resources(ram=1).fits_in(cap)
    assert Resources(vram=11, ram=1, cpu=4).exceeds(cap) == ["ram", "vram"]
    assert Resources(vram=1).exceeds(cap) == []


def test_is_zero() -> None:
    assert Resources().is_zero
    assert Resources(vram=0).is_zero
    assert not Resources(vram=1).is_zero


def test_json_round_trip_preserves_int_and_float() -> None:
    r = Resources(vram=24 * GiB, cpu=2.5)
    data = json.loads(json.dumps(r.to_json()))
    back = Resources.from_json(data)
    assert back == r
    assert isinstance(back["vram"], int)
    assert isinstance(back["cpu"], float)
    with pytest.raises(TypeError):
        Resources.from_json([1, 2])


def test_pickle_and_copy() -> None:
    r = Resources(vram=1, cpu=2.5)
    assert pickle.loads(pickle.dumps(r)) == r
    assert copy.deepcopy(r) == r


@given(RES, RES)
def test_add_commutative(a: Resources, b: Resources) -> None:
    assert a + b == b + a


INT_RES = st.dictionaries(DIMS, st.integers(0, 10**15), max_size=5).map(Resources)


@given(INT_RES, INT_RES, INT_RES)
def test_add_associative_on_ints(a: Resources, b: Resources, c: Resources) -> None:
    assert (a + b) + c == a + (b + c)


@given(RES, RES)
def test_add_then_sub_round_trips(a: Resources, b: Resources) -> None:
    back = (a + b) - b
    for d in a.keys() | back.keys():
        scale = max(a[d], b[d], 1)
        assert math.isclose(back[d], a[d], rel_tol=1e-9, abs_tol=1e-12 * scale)


@given(RES, RES)
def test_max_commutative_idempotent(a: Resources, b: Resources) -> None:
    assert a.max(b) == b.max(a)
    assert a.max(a) == a
    assert a.fits_in(a.max(b))
    assert b.fits_in(a.max(b))
    assert a.min(b).fits_in(a)


@given(RES, RES)
def test_clamped_sub_never_negative_and_fits(a: Resources, b: Resources) -> None:
    d = a.sub(b, clamp=True)
    assert all(v >= 0 for v in d.values())
    assert d.fits_in(a)


@given(RES, RES)
def test_exceeds_iff_not_fits(a: Resources, b: Resources) -> None:
    assert (a.exceeds(b) == []) == a.fits_in(b)


@given(RES)
def test_json_round_trip(a: Resources) -> None:
    assert Resources.from_json(json.loads(json.dumps(a.to_json()))) == a


@given(RES, RES)
def test_hash_consistent_with_eq(a: Resources, b: Resources) -> None:
    if a == b:
        assert hash(a) == hash(b)
    assert hash(Resources(dict(a))) == hash(a)


@given(RES, st.sampled_from([1.15, 1.25, 1.1, 0.9, 1 / 3]))
def test_scale_round_off_never_raises(a: Resources, k: float) -> None:
    scaled = a.scale(k).scale(1 / k)
    scaled - a
    a - scaled
