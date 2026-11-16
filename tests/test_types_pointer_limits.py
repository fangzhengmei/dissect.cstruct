from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest

from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.exception import (
    NullPointerDereference,
    PointerBudgetExceeded,
    PointerCycleError,
    PointerError,
    PointerWindowError,
)
from dissect.cstruct.types import BaseType
from tests.util import verify_compiled

if TYPE_CHECKING:
    from typing import BinaryIO

    from dissect.cstruct.cstruct import Endianness

# A self-referential node and a root pointing at two nodes that share a target.
#
# pointer size is uint16; node = uint16 next + uint32 value -> 6 bytes (next@0, value@2)
#   root @0:  n1 -> 6, n2 -> 6 (shared), label -> 22
#   node A @6:  next -> 14, value 0x44434241 (@8..12), 2 pad bytes
#   node B @14: next -> 0 (null), value 0x45454545 (@16..20), 2 pad bytes
#   label @22: b"hi\0"
CDEF = """
struct node {
    node    *next;
    uint32  value;
};
struct root {
    node    *n1;
    node    *n2;
    char    *label;
};
"""

BUF = (
    b"\x06\x00"  # 0: root.n1 -> 6
    b"\x06\x00"  # 2: root.n2 -> 6
    b"\x16\x00"  # 4: root.label -> 22
    b"\x0e\x00"  # 6: A.next -> 14
    b"ABCD"  # 8: A.value
    b"\x00\x00"  # 12: A pad
    b"\x00\x00"  # 14: B.next -> 0 (null)
    b"EEEE"  # 16: B.value
    b"\x00\x00"  # 20: B pad
    b"hi\x00"  # 22: label
)


def _make_cs(compiled: bool, **limits: object) -> cstruct:
    cs = cstruct(pointer="uint16", **limits)
    cs.load(CDEF, compiled=compiled)
    assert verify_compiled(cs.root, compiled)
    assert verify_compiled(cs.node, compiled)
    return cs


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_pointer_limits_disabled_by_default(compiled: bool) -> None:
    cs = cstruct(pointer="uint16")
    cs.load(CDEF, compiled=compiled)
    obj = cs.root(io.BytesIO(BUF))

    # No resolver is attached when no limits are configured, so dereferencing is unrestricted
    assert obj.n1._resolver is None
    assert obj.n1.dereference().value == 0x44434241


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_window_allows_targets_inside(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    assert obj.n1.dereference().value == 0x44434241
    assert obj.n1.dereference().next.dereference().value == 0x45454545
    assert obj.label.dereference() == b"hi"


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_window_rejects_target_past_end(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 22), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    # label target is 22, which is at the (exclusive) window end
    with pytest.raises(PointerWindowError):
        obj.label.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_window_rejects_target_extending_past_end(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 19), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    # node B starts at 14 and its 6 bytes extend to 20, past the window end of 19
    with pytest.raises(PointerWindowError):
        obj.n1.dereference().next.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_window_rejects_target_before_start(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(10, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    with pytest.raises(PointerWindowError):
        obj.n1.dereference()  # target 6 is before the window start of 10


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_negative_pointer_offset_rejected(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    negative = obj.n1 - 100  # 6 - 100 is a negative offset
    assert int(negative) < 0
    with pytest.raises(PointerWindowError):
        negative.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_window_bounds_char_string_reads(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 24), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    # The string "hi\0" spans 22..25 but the window ends at 24, so the read is truncated and fails
    with pytest.raises(EOFError):
        obj.label.dereference()

    # The stream position is restored even though the read was truncated
    assert obj.label._stream.tell() == 6


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_null_pointer_still_raises_null_error(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    with pytest.raises(NullPointerDereference):
        obj.n1.dereference().next.dereference().next.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_budget_limits_dereference_chain(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=1)
    obj = cs.root(io.BytesIO(BUF))

    node_a = obj.n1.dereference()  # first dereference is allowed
    assert node_a.value == 0x44434241

    # Following the next pointer would be the second dereference, exceeding the budget
    with pytest.raises(PointerBudgetExceeded):
        node_a.next.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_budget_zero_rejects_first_dereference(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=0)
    obj = cs.root(io.BytesIO(BUF))

    with pytest.raises(PointerBudgetExceeded):
        obj.n1.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_budget_shared_target_does_not_spend_budget_twice(compiled: bool) -> None:
    # Both root pointers target node A. Resolving the second one must reuse the cached value instead
    # of spending another dereference, so a budget of 1 still allows both to be resolved.
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=1)
    obj = cs.root(io.BytesIO(BUF))

    first = obj.n1.dereference()
    second = obj.n2.dereference()
    assert first is second


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_shared_target_returns_identical_object(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    assert obj.n1.dereference() is obj.n2.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_resolver_is_shared_across_whole_graph(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=8)
    obj = cs.root(io.BytesIO(BUF))

    node_a = obj.n1.dereference()
    # The nested pointer read while dereferencing n1 must reuse the very same resolver
    assert node_a.next._resolver is obj.n1._resolver


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_stream_position_restored_on_window_rejection(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 22), pointer_budget=8)
    stream = io.BytesIO(BUF)
    obj = cs.root(stream)

    stream.seek(7)
    with pytest.raises(PointerWindowError):
        obj.label.dereference()
    assert stream.tell() == 7


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_stream_position_restored_on_budget_rejection(compiled: bool) -> None:
    cs = _make_cs(compiled, pointer_window=(0, 32), pointer_budget=1)
    stream = io.BytesIO(BUF)
    obj = cs.root(stream)
    node_a = obj.n1.dereference()

    stream.seek(11)
    with pytest.raises(PointerBudgetExceeded):
        node_a.next.dereference()
    assert stream.tell() == 11


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_resolver_propagates_through_embedded_struct(compiled: bool) -> None:
    # The resolver must reach pointers nested inside an embedded (non-pointer) structure, so the budget
    # is shared across the whole parse rather than restarting at the nested structure.
    cdef = """
    struct inner {
        uint32 *p1;
        uint32 *p2;
    };
    struct outer {
        inner nested;
    };
    """
    buf = b"\x04\x00" b"\x08\x00" b"ABCD" b"EFEF"  # p1 -> 4, p2 -> 8
    cs = cstruct(pointer="uint16", pointer_window=(0, 12), pointer_budget=1)
    cs.load(cdef, compiled=compiled)

    obj = cs.outer(io.BytesIO(buf))
    assert obj.nested.p1.dereference() == 0x44434241
    assert obj.nested.p1._resolver is obj.nested.p2._resolver

    with pytest.raises(PointerBudgetExceeded):
        obj.nested.p2.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_union_pointer_limits(compiled: bool) -> None:
    # Pointers inside a union dereference against the union's own buffer, and must honor the limits too.
    cdef = """
    union choice {
        uint8 *p;
        uint8 bytes[2];
    };
    struct holder {
        choice c;
    };
    """
    buf = b"\x01\x42"  # p -> 1, byte at offset 1 in the union buffer is 0x42

    cs = cstruct(pointer="uint8", pointer_window=(0, 2), pointer_budget=2)
    cs.load(cdef, compiled=compiled)
    assert cs.holder(io.BytesIO(buf)).c.p.dereference() == 0x42

    cs_window = cstruct(pointer="uint8", pointer_window=(0, 1))
    cs_window.load(cdef, compiled=compiled)
    with pytest.raises(PointerWindowError):
        cs_window.holder(io.BytesIO(buf)).c.p.dereference()

    cs_budget = cstruct(pointer="uint8", pointer_budget=0)
    cs_budget.load(cdef, compiled=compiled)
    with pytest.raises(PointerBudgetExceeded):
        cs_budget.holder(io.BytesIO(buf)).c.p.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_array_of_pointers_shares_resolver_and_limits(compiled: bool) -> None:
    cdef = """
    struct mainargs {
        uint8_t argc;
        char *args[4];
    };
    """
    buf = b"\x02\x09\x00\x16\x00\x00\x00\x00\x00argument one\x00argument two\x00"

    cs = cstruct(pointer="uint16", pointer_window=(0, len(buf)), pointer_budget=4)
    cs.load(cdef, compiled=compiled)
    obj = cs.mainargs(io.BytesIO(buf))
    assert obj.args[0].dereference() == b"argument one"
    assert obj.args[1].dereference() == b"argument two"
    resolvers = {id(p._resolver) for p in obj.args if p._resolver is not None}
    assert len(resolvers) == 1

    cs_budget = cstruct(pointer="uint16", pointer_window=(0, len(buf)), pointer_budget=1)
    cs_budget.load(cdef, compiled=compiled)
    obj_budget = cs_budget.mainargs(io.BytesIO(buf))
    assert obj_budget.args[0].dereference() == b"argument one"
    with pytest.raises(PointerBudgetExceeded):
        obj_budget.args[1].dereference()

    cs_window = cstruct(pointer="uint16", pointer_window=(0, 0x10), pointer_budget=4)
    cs_window.load(cdef, compiled=compiled)
    obj_window = cs_window.mainargs(io.BytesIO(buf))
    with pytest.raises(PointerWindowError):
        obj_window.args[1].dereference()  # target 0x16 is outside the window


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_pointer_arithmetic_inherits_limits(compiled: bool) -> None:
    cs = cstruct(pointer="uint16", pointer_window=(0, 8))
    cs.load("struct r { uint32 *p; };", compiled=compiled)
    obj = cs.r(io.BytesIO(b"\x02\x00ABCD"))

    assert obj.p.dereference() == 0x44434241

    moved = obj.p + 6  # target 8, at the exclusive window end
    assert moved._resolver is obj.p._resolver
    with pytest.raises(PointerWindowError):
        moved.dereference()

    negative = obj.p - 10  # target 2 - 10 is negative
    assert int(negative) < 0
    assert negative._resolver is obj.p._resolver
    with pytest.raises(PointerWindowError):
        negative.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_void_pointer_counts_against_budget_and_window(compiled: bool) -> None:
    cs = cstruct(pointer="uint16", pointer_window=(0, 8), pointer_budget=1)
    cs.load("struct r { void *p; uint32 *q; };", compiled=compiled)
    obj = cs.r(io.BytesIO(b"\x02\x00" b"\x04\x00" b"ABCD"))

    # A void pointer resolves to None but still consumes a dereference
    assert obj.p.dereference() is None
    with pytest.raises(PointerBudgetExceeded):
        obj.q.dereference()

    cs_window = cstruct(pointer="uint16", pointer_window=(0, 2))
    cs_window.load("struct r { void *p; };", compiled=compiled)
    with pytest.raises(PointerWindowError):
        cs_window.r(io.BytesIO(b"\x05\x00")).p.dereference()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_nested_pointer_cycle_rejected(compiled: bool) -> None:
    class Hop(BaseType):
        """A one-byte value whose read immediately dereferences that byte as a pointer to another Hop."""

        __size__ = 1

        @classmethod
        def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Any:
            val = cls.__cs__.uint8._read(stream, context=context, endian=endian)
            ptr_type = cls.__cs__._make_pointer(cls)
            ptr = ptr_type.__new__(ptr_type, val, stream, context=context, endian=endian)
            return ptr.dereference()

    cs = cstruct(pointer="uint8", pointer_budget=100)
    cs.add_custom_type("hop", Hop)
    cs.load("struct root { hop h; };", compiled=compiled)

    # Self loop at a non-zero address: offset 0 holds 1, offset 1 holds 1 -> 1 -> 1
    with pytest.raises(PointerCycleError):
        cs.root(io.BytesIO(b"\x01\x01"))

    # Two-hop cycle 0 -> 2 -> 1 -> 2
    with pytest.raises(PointerCycleError):
        cs.root(io.BytesIO(b"\x02\x02\x01"))


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_pointer_of_pointer_budget(compiled: bool) -> None:
    cs = cstruct(pointer="uint8", pointer_budget=1)
    cs.load("struct test { uint32 **ptr; };", compiled=compiled)

    obj = cs.test(io.BytesIO(b"\x01\x02AAAA"))
    inner = obj.ptr.dereference()  # first dereference returns the inner pointer
    assert inner == 2
    with pytest.raises(PointerBudgetExceeded):
        inner.dereference()


def test_pointer_exceptions_are_distinguishable() -> None:
    for exc in (PointerWindowError, PointerBudgetExceeded, PointerCycleError, NullPointerDereference):
        assert issubclass(exc, PointerError)


def test_pointer_window_validation() -> None:
    with pytest.raises(ValueError):
        cstruct(pointer="uint16", pointer_window=(5, 2))
    with pytest.raises(ValueError):
        cstruct(pointer="uint16", pointer_window=(-1, 2))


def test_pointer_budget_validation() -> None:
    with pytest.raises(ValueError):
        cstruct(pointer="uint16", pointer_budget=-1)


def test_pointer_limits_context_manager_restores_state() -> None:
    cs = cstruct(pointer="uint16", pointer_budget=9)
    assert cs.pointer_window is None

    with cs.pointer_limits(window=(0, 4)):
        assert cs.pointer_window == (0, 4)
        assert cs.pointer_budget == 9
        with cs.pointer_limits(budget=1):
            assert cs.pointer_window == (0, 4)
            assert cs.pointer_budget == 1
        assert cs.pointer_budget == 9

    assert cs.pointer_window is None
    assert cs.pointer_budget == 9


def test_pointer_limits_survive_copy() -> None:
    cs = cstruct(pointer="uint16", pointer_window=(1, 9), pointer_budget=3).copy()
    assert cs.pointer_window == (1, 9)
    assert cs.pointer_budget == 3


def test_per_pointer_dereference_cache_preserved() -> None:
    cs = _make_cs(True, pointer_window=(0, 32), pointer_budget=1)
    obj = cs.root(io.BytesIO(BUF))

    first = obj.n1.dereference()
    second = obj.n1.dereference()  # cached: must not consume the budget again
    assert first is second


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_runtime_limit_changes_apply_to_already_loaded_types(compiled: bool) -> None:
    cs = cstruct(pointer="uint16")
    cs.load("struct r { uint32 *p; };", compiled=compiled)
    buf = b"\x02\x00ABCD"

    # No limits yet: dereferencing is unrestricted
    assert cs.r(io.BytesIO(buf)).p.dereference() == 0x44434241

    # Limits applied at runtime via the context manager must take effect on the already loaded type
    with cs.pointer_limits(budget=0):
        with pytest.raises(PointerBudgetExceeded):
            cs.r(io.BytesIO(buf)).p.dereference()

    # After the context exits, dereferencing is unrestricted again
    assert cs.r(io.BytesIO(buf)).p.dereference() == 0x44434241
