from __future__ import annotations

import io

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.exception import (
    DereferenceBudgetExceeded,
    NegativePointerAddress,
    NullPointerDereference,
    PointerCycleError,
    PointerDereferenceError,
    PointerOutOfWindow,
)
from dissect.cstruct.types.pointer import DereferenceBudget, DereferenceResult, PointerWindow
from tests.util import verify_compiled

GRAPH = """
struct node {
    uint16  value;
    node    *next;
};

struct graph {
    node    *a;
    node    *b;
};
"""

# graph { a: 0x04, b: 0x08 }, node @0x04 { value: 1, next: 0x08 }, node @0x08 { value: 2, next: 0 }
GRAPH_DATA = b"\x04\x00\x08\x00" + b"\x01\x00\x08\x00" + b"\x02\x00\x00\x00"


def make_cs(compiled: bool, **kwargs) -> cstruct:
    cs = cstruct(pointer="uint16", **kwargs)
    cs.load(GRAPH, compiled=compiled)
    return cs


def test_window_limits_pointer_visibility(compiled: bool) -> None:
    """A window hides everything outside of it, while the pointer graph inside of it stays intact."""
    cs = make_cs(compiled)

    obj = cs.graph(GRAPH_DATA)
    assert obj.a.dereference().value == 1
    assert obj.b.dereference().value == 2

    # A window that only covers the first 8 bytes hides the second node at 0x08
    cs.read_window = (0, 8)
    obj = cs.graph(GRAPH_DATA)
    assert obj.a.dereference().value == 1
    assert obj.a.dereference().next.try_dereference().reason == "out_of_window"

    # The same window given per parse has the same result
    obj = cs.graph.reads(GRAPH_DATA, read_window=(0, 8))
    assert obj.b.try_dereference().reason == "out_of_window"

    # A window that starts after the first node hides it instead
    obj = cs.graph(GRAPH_DATA)
    assert obj.a.try_dereference(window=PointerWindow.from_bounds(0x04, 0xC)).ok
    assert obj.a.try_dereference(window=PointerWindow.from_bounds(0x08, 0xC)).reason == "out_of_window"


def test_window_rejects_partial_value(compiled: bool) -> None:
    """A pointer that points inside the window, but whose value extends past it, is rejected."""
    cs = cstruct(pointer="uint16")
    cs.load("struct t { uint32 *a; uint32 *b; };", compiled=compiled)

    # a = 0x08, b = 0x0A, value of 4 bytes at 0x08
    buf = b"\x08\x00\x0a\x00" + b"\xff" * 4 + b"\x01\x02\x03\x04"

    for window, expected in (((0, 10), ["out_of_window", "out_of_window"]), ((0, 12), [None, "out_of_window"])):
        obj = cs.t.reads(buf, read_window=window)
        assert [obj.a.try_dereference().reason, obj.b.try_dereference().reason] == expected

    obj = cs.t.reads(buf, read_window=(0, 16))
    assert obj.a.dereference() == 0x04030201


def test_window_rejects_partial_string(compiled: bool) -> None:
    """A null terminated string that runs past the window is rejected, one that ends inside it is not."""
    cs = cstruct(pointer="uint16")
    cs.load("struct t { char *a; char *b; };", compiled=compiled)

    # a = 0x04 (string "ab", terminated at 0x06), b = 0x07 (string "cdef", terminated at 0x0B)
    buf = b"\x04\x00\x07\x00" + b"ab\x00" + b"cdef\x00"

    obj = cs.t.reads(buf, read_window=(0, 8))
    assert obj.a.dereference() == b"ab"
    assert obj.b.try_dereference().reason == "out_of_window"

    # The window ends exactly on the terminator
    obj = cs.t.reads(buf, read_window=(0, 7))
    assert obj.a.dereference() == b"ab"

    obj = cs.t.reads(buf, read_window=(0, len(buf)))
    assert obj.a.dereference() == b"ab"
    assert obj.b.dereference() == b"cdef"


def test_negative_pointer_address(compiled: bool) -> None:
    """A negative address is rejected, with or without a window."""
    cs = make_cs(compiled)
    obj = cs.graph(GRAPH_DATA)

    negative = obj.a - 0x10
    assert int(negative) < 0

    with pytest.raises(NegativePointerAddress) as exc:
        negative.dereference()
    assert exc.value.reason == "negative_address"

    result = negative.try_dereference()
    assert not result
    assert result.reason == "negative_address"
    assert isinstance(result.error, PointerOutOfWindow)

    # Also when the pointer belongs to a resolution with a window
    cs.read_window = (0, len(GRAPH_DATA))
    obj = cs.graph(GRAPH_DATA)
    assert (obj.a - 0x10).try_dereference().reason == "negative_address"


def test_null_pointer_is_distinguishable(compiled: bool) -> None:
    """A null pointer is told apart from a rejected address and keeps raising for existing callers."""
    cs = make_cs(compiled)
    obj = cs.graph(b"\x00\x00\x00\x00" + GRAPH_DATA[4:])

    with pytest.raises(NullPointerDereference):
        obj.a.dereference()

    assert obj.a.try_dereference().reason == "null_pointer"
    assert obj.a.try_dereference(window=(0, 16)).reason == "null_pointer"


def test_cycle_is_rejected(compiled: bool) -> None:
    """A structure that needs itself to be resolved is rejected instead of recursing forever."""
    cs = cstruct(pointer="uint16")
    cs.load("struct rec { uint16 size; rec *next; uint8 data[next.size]; };", compiled=compiled)

    buf = b"\x00\x00" + b"\x02\x00" + b"\x02\x00" + b"ab"

    with pytest.raises(PointerCycleError) as exc:
        cs.rec.reads(buf)
    assert exc.value.reason == "cycle"

    # It is rejected the same way with limits configured
    cs.read_window = (0, len(buf))
    cs.dereference_budget = 10
    with pytest.raises(PointerCycleError):
        cs.rec.reads(buf)


def test_cyclic_chain_terminates(compiled: bool) -> None:
    """A cyclic linked list terminates and keeps returning the same object for the same node."""
    cs = cstruct(pointer="uint16", dereference_budget=8)
    cs.load("struct node { uint16 value; node *next; };", compiled=compiled)
    cs.load("struct holder { node *p; };", compiled=compiled)

    # holder { p: 0x04 }, node @0x04 { value: 7, next: 0x04 }
    holder = cs.holder(b"\x04\x00" + b"\x00\x00" + b"\x07\x00\x04\x00")

    first = holder.p.dereference()
    second = first.next.dereference()
    assert first.value == 7
    assert second is first

    # Walking further keeps the result stable instead of growing the chain
    assert second.next.dereference() is first


def test_shared_object_semantics(compiled: bool) -> None:
    """Multiple pointers to the same object share it, without costing extra budget."""
    cs = cstruct(pointer="uint16", dereference_budget=1)
    cs.load("struct node { uint16 value; };", compiled=compiled)
    cs.load("struct pair { node *a; node *b; };", compiled=compiled)

    obj = cs.pair(b"\x04\x00\x04\x00" + b"\x2a\x00")
    a = obj.a.dereference()
    b = obj.b.dereference()
    assert a is b
    assert a.value == 0x2A

    # Without a resolution the values of two pointers are equal, but not shared
    cs2 = cstruct(pointer="uint16")
    cs2.load("struct node { uint16 value; };", compiled=compiled)
    cs2.load("struct pair { node *a; node *b; };", compiled=compiled)
    obj = cs2.pair(b"\x04\x00\x04\x00" + b"\x2a\x00")
    assert obj.a.dereference() is not obj.b.dereference()
    assert obj.a.dereference().value == obj.b.dereference().value == 0x2A


def test_dereference_budget(compiled: bool) -> None:
    """The budget limits how many dereferences a parse traces, equally on both paths."""
    expected = {0: ["budget_exceeded"] * 2, 1: [None, "budget_exceeded"], 2: [None, None]}

    for budget, reasons in expected.items():
        cs = make_cs(compiled, dereference_budget=budget)
        obj = cs.graph(GRAPH_DATA)
        assert [obj.a.try_dereference().reason, obj.b.try_dereference().reason] == reasons

    # The budget can also be given per parse
    cs = make_cs(compiled)
    obj = cs.graph.reads(GRAPH_DATA, dereference_budget=1)
    assert obj.a.try_dereference().ok
    assert obj.b.try_dereference().reason == "budget_exceeded"

    # And per dereference
    cs = make_cs(compiled)
    obj = cs.graph(GRAPH_DATA)
    assert obj.a.try_dereference(budget=0).reason == "budget_exceeded"
    assert obj.a.dereference(budget=1).value == 1


def test_budget_after_parse_is_still_shared(compiled: bool) -> None:
    """Pointers created by a parse keep the budget of that parse when they are dereferenced later."""
    cs = make_cs(compiled, dereference_budget=2)
    obj = cs.graph(GRAPH_DATA)

    # The budget only counts unique objects, so resolving the same node again is free
    assert obj.a.dereference().value == 1
    assert obj.a.dereference().value == 1
    assert obj.a.dereference().next.dereference().value == 2

    # Both pointers now share the same node, which was already resolved
    assert obj.b.dereference() is obj.a.dereference().next.dereference()

    # A budget that was already exhausted rejects the next object it sees
    cs.read_window = None
    obj = make_cs(compiled, dereference_budget=1).graph(GRAPH_DATA)
    assert obj.a.dereference().value == 1
    assert obj.b.try_dereference().reason == "budget_exceeded"


def test_stream_position_restored_on_rejection(compiled: bool) -> None:
    """A rejected dereference restores the stream position, like an accepted one does."""
    cs = cstruct(pointer="uint16")
    cs.load(GRAPH, compiled=compiled)

    stream = io.BytesIO(GRAPH_DATA)
    obj = cs.graph(stream)
    assert stream.tell() == len(GRAPH_DATA) - 8

    obj.a.dereference()
    assert stream.tell() == len(GRAPH_DATA) - 8

    cs.read_window = (0, 8)
    obj = cs.graph(stream)
    obj.a.dereference()
    position = stream.tell()

    assert obj.a.dereference().next.try_dereference().reason == "out_of_window"
    assert stream.tell() == position

    obj.a.try_dereference(budget=0)
    assert stream.tell() == position


@pytest.mark.parametrize(
    "window",
    [(0, 16), [0, 16], range(16), slice(0, 16), {"offset": 0, "size": 16}, {"start": 0, "end": 16}],
)
def test_window_formats(window: object) -> None:
    """A window can be given as an offset and size, as a start and end, or as a slice."""
    cs = make_cs(True, read_window=window)
    obj = cs.graph(GRAPH_DATA)
    assert obj.a.dereference().value == 1

    assert cs.read_window == PointerWindow(0, 16)
    assert cs.read_window == PointerWindow(start=0, end=16)
    assert cs.read_window.end == 16
    assert hash(cs.read_window) == hash(PointerWindow(0, 16))


def test_window_validation() -> None:
    with pytest.raises(ValueError, match="offset must be non-negative"):
        PointerWindow(-1, 10)
    with pytest.raises(ValueError, match="size must be non-negative"):
        PointerWindow(0, -1)
    with pytest.raises(ValueError, match="size must be non-negative"):
        PointerWindow.from_bounds(10, 5)
    with pytest.raises(TypeError, match="Cannot combine 'end'"):
        PointerWindow(0, 10, end=20)
    with pytest.raises(TypeError, match="Invalid read window"):
        cstruct(read_window="nope")

    with pytest.raises(ValueError, match="budget must be non-negative"):
        DereferenceBudget(-1)
    with pytest.raises(TypeError, match="Invalid dereference budget"):
        cstruct(dereference_budget="nope")


def test_copy_keeps_limits() -> None:
    cs = cstruct(pointer="uint16", read_window=(0, 16), dereference_budget=3)
    copied = cs.copy()

    assert copied.read_window == PointerWindow(0, 16)
    assert copied.dereference_budget == DereferenceBudget(3)


def test_dereference_result() -> None:
    cs = make_cs(True, dereference_budget=1)
    obj = cs.graph(GRAPH_DATA)

    result = obj.a.try_dereference()
    assert isinstance(result, DereferenceResult)
    assert result.ok
    assert result.reason is None
    assert result.value.value == 1
    assert bool(result)

    rejected = obj.b.try_dereference()
    assert not rejected.ok
    assert not rejected
    assert rejected.value is None
    assert rejected.reason == "budget_exceeded"
    assert isinstance(rejected.error, DereferenceBudgetExceeded)
    assert isinstance(rejected.error, PointerDereferenceError)


def test_unlimited_by_default(compiled: bool) -> None:
    """Without limits, pointer behavior is unchanged."""
    cs = make_cs(compiled)
    assert cs.read_window is None
    assert cs.dereference_budget is None

    obj = cs.graph(GRAPH_DATA)
    assert obj.a.dereference().value == 1
    assert obj.a.dereference().next.dereference().value == 2
    assert obj.b.dereference().value == 2

    # Even a large pointer value isn't rejected, it simply fails to read
    far = cs.graph(b"\xff\x00\x00\x00" + GRAPH_DATA[4:])
    with pytest.raises(EOFError):
        far.a.dereference()


def test_compiled_and_interpreted_agree(compiled: bool) -> None:
    """The compiled and interpreted readers honour the same constraints."""
    cdef = """
    struct node {
        uint16  value;
        node    *next;
    };

    struct graph {
        node    *a;
        node    *b;
    };
    """

    results = {}
    for variant in (True, False):
        cs = cstruct(pointer="uint16")
        cs.load(cdef, compiled=variant)
        assert verify_compiled(cs.graph, variant)

        for window, budget in ((None, None), ((0, 12), None), ((0, 8), None), (None, 1), (None, 2)):
            cs.read_window = window
            cs.dereference_budget = budget
            obj = cs.graph(GRAPH_DATA)
            results[(variant, window, budget)] = (
                obj.a.try_dereference().reason,
                obj.b.try_dereference().reason,
            )

        cs.read_window = None
        cs.dereference_budget = None
        results[(variant, None, None)] = (
            len(cs.graph(GRAPH_DATA).dumps()),
            cs.graph(GRAPH_DATA).a.dereference().value,
        )

    for window, budget in ((None, None), ((0, 12), None), ((0, 8), None), (None, 1), (None, 2)):
        assert results[(True, window, budget)] == results[(False, window, budget)]


def test_window_does_not_change_bytes_consumed(compiled: bool) -> None:
    """A window only restricts dereferences, the parse itself consumes the same bytes."""
    for window in (None, (0, 4), (0, len(GRAPH_DATA))):
        cs = make_cs(compiled, read_window=window)
        stream = io.BytesIO(GRAPH_DATA)
        cs.graph(stream)
        assert stream.tell() == 4
        assert cs.graph(GRAPH_DATA).dumps() == GRAPH_DATA[:4]


def test_pointer_struct_write_unaffected(compiled: bool) -> None:
    """Writing pointers is unaffected by the limits."""
    cs = make_cs(compiled, dereference_budget=0, read_window=(0, 4))
    assert cs.graph(a=0x04, b=0x08).dumps() == b"\x04\x00\x08\x00"


def test_empty_window(compiled: bool) -> None:
    """An empty window rejects every dereference."""
    cs = make_cs(compiled, read_window=PointerWindow(0, 0))
    obj = cs.graph(GRAPH_DATA)
    assert obj.a.try_dereference().reason == "out_of_window"


def test_per_call_limits_cover_the_chain(compiled: bool) -> None:
    """A limit given to a single dereference applies to the whole chain that is walked from it."""
    cs = cstruct(pointer="uint32")
    cs.load("struct node { uint32 value; node *next; };", compiled=compiled)

    # Nodes at 0x00, 0x08, 0x10 and 0x18, with values 1, 2, 3 and 4
    data = (
        b"\x01\x00\x00\x00\x08\x00\x00\x00"
        b"\x02\x00\x00\x00\x10\x00\x00\x00"
        b"\x03\x00\x00\x00\x18\x00\x00\x00"
        b"\x04\x00\x00\x00\x10\x00\x00\x00"
    )
    obj = cs.node(data)

    node2 = obj.next.dereference(budget=2)
    assert node2.value == 2
    assert node2.next.dereference().value == 3
    assert node2.next.dereference().next.try_dereference().reason == "budget_exceeded"

    node2 = obj.next.dereference(window=(0, len(data)))
    assert node2.next.dereference().value == 3
    assert node2.next.dereference().next.dereference().value == 4

    node2 = obj.next.dereference(window=(0, 0x18))
    assert node2.value == 2
    assert node2.next.dereference().value == 3
    assert node2.next.dereference().next.try_dereference().reason == "out_of_window"


def test_pointer_that_is_read_while_limited_inherits_limits(compiled: bool) -> None:
    """Pointers that are read while a limited dereference resolves a value belong to that resolution."""
    cs = cstruct(pointer="uint32")
    cs.load(
        """
        struct inner {
            uint32  value;
            inner   *next;
        };

        struct outer {
            inner   *first;
        };
        """,
        compiled=compiled,
    )

    # outer { first: 0x04 }, inner @0x04 { value: 1, next: 0x0C }, inner @0x0C { value: 2, next: 0 }
    data = b"\x04\x00\x00\x00" + b"\x01\x00\x00\x00\x0c\x00\x00\x00" + b"\x02\x00\x00\x00\x00\x00\x00\x00"
    obj = cs.outer(data)

    # The window of the dereference of ``first`` also applies to the ``next`` pointer inside the value
    first = obj.first.dereference(window=(0, 0x10))
    assert first.value == 1
    assert first.next.try_dereference().reason == "out_of_window"

    first = obj.first.dereference(window=(0, len(data)))
    assert first.next.dereference().value == 2

    # The budget of the dereference of ``first`` also covers the ``next`` pointer inside the value
    first = obj.first.dereference(budget=1)
    assert first.value == 1
    assert first.next.try_dereference().reason == "budget_exceeded"


def test_pointer_arithmetic_keeps_the_resolution(compiled: bool) -> None:
    """Arithmetic on a pointer keeps its resolution, so the budget and window of a parse still apply."""
    cs = make_cs(compiled, dereference_budget=1)
    obj = cs.graph(GRAPH_DATA)

    pointer = obj.a + 4
    assert pointer == 8
    assert pointer.dereference().value == 2

    # The new pointer shares the budget of the parse, and the object it resolved is shared with ``b``
    assert obj.b.dereference() is pointer.dereference()

    # The window of the parse applies too
    cs.read_window = (0, 8)
    shifted = cs.graph(GRAPH_DATA).a + 4
    assert shifted.try_dereference().reason == "out_of_window"


def test_dereference_does_not_change_pointer_identity(compiled: bool) -> None:
    """Dereferencing doesn't change the pointer or the stream, so repeated reads give the same result."""
    cs = cstruct(pointer="uint16", read_window=(0, 10))
    cs.load("struct node { uint16 value; };", compiled=compiled)
    cs.load("struct pair { node *a; node *b; };", compiled=compiled)

    # a = 0x04 -> 1, b = 0x08 -> 2
    obj = cs.pair(b"\x04\x00\x08\x00" + b"\x01\x00" + b"\x00\x00" + b"\x02\x00")

    assert obj.a.dereference().value == 1
    assert obj.b.dereference().value == 2
    assert obj.a == 4
    assert obj.b == 8
    assert (obj.a.dereference().value, obj.b.dereference().value) == (1, 2)


def test_window_slice_is_not_cached_by_the_pointer(compiled: bool) -> None:
    """Different windows on the same unrestricted pointer each have their own result."""
    cs = cstruct(pointer="uint16")
    cs.load("struct node { uint16 value; node *next; };", compiled=compiled)

    node = cs.node(b"\x01\x00\x04\x00" + b"\x02\x00\x00\x00")

    assert node.next.dereference().value == 2
    assert node.next.try_dereference(window=(0, 4)).reason == "out_of_window"
    assert node.next.dereference().value == 2
