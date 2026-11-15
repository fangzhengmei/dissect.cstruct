"""Tests for the length and budget semantics of arrays in untrusted structures.

An array length in a forensic structure is usually taken from the evidence itself, through a length
expression like ``char data[length]``. Such a length is untrusted: it can be negative, be the product
of two header fields, or claim far more entries than the evidence can hold. A parser that trusts it
can be driven into allocating a huge array, or into reading past the end of the record, before it
notices that the input is malformed.

The tests below pin down the intended semantics:

1. A length is validated before it is used: negative and non-integer lengths are rejected with an
   ``ArraySizeError``, while a legal zero length and normal arrays keep parsing.
2. A length can't be turned into a large allocation. Reads are bounded by what the stream can
   actually deliver, so an over-budget length fails with an ``EOFError`` at the array's evidence
   offset instead of pre-allocating the declared capacity or raising ``MemoryError``/``OverflowError``.
3. Element types that consume no bytes get an explicit entry budget, because no amount of evidence
   can ever satisfy their declared length.
4. The compiled and the interpreted reader reject the same malformed input at the same offset with
   the same exception, and a failed read never yields a partially filled array as if it were complete.
"""

from __future__ import annotations

import io
import struct

import pytest

from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.exception import ArraySizeError
from dissect.cstruct.types.base import MAX_ZERO_WIDTH_ARRAY_ENTRIES
from tests.util import verify_compiled


class GuardedStream(io.BytesIO):
    """A seekable stream that records read requests and rejects unusually large ones.

    A bounded parser must never ask the stream for the full declared length of an untrusted array,
    because that is what makes a stream allocate the declared capacity up front. Any request above
    the guard is a finding: it means the declared length was trusted before the evidence was checked.
    """

    def __init__(self, data: bytes, limit: int = 1 << 16):
        super().__init__(data)
        self.limit = limit
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        if size is not None and size >= 0:
            self.requests.append(size)
            assert size <= self.limit, f"stream asked for {size} bytes, guard is {self.limit}"
        return super().read(size)


def canonical(value: object) -> object:
    """Return a comparable representation of a parsed value, independent of its concrete class."""
    if hasattr(value, "__dict__") and hasattr(value.__class__, "__fields__"):
        return (value.__class__.__name__, tuple((key, canonical(val)) for key, val in value.__dict__.items()))
    if isinstance(value, list):
        return ("list", tuple(canonical(item) for item in value))
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    return (type(value).__name__, value)


def observe(type_: type, data: bytes, *, guard: int = 1 << 16) -> tuple:
    """Parse ``data`` and return a comparable outcome of the read attempt.

    The outcome contains the exception class, the offset the parser gave up at and the exception
    message, or the parsed value and the final stream position on success. Both read paths must
    produce the same outcome for the same input.
    """
    stream = GuardedStream(data, limit=guard)
    try:
        value = type_(stream)
    except Exception as e:
        return (type(e).__name__, stream.tell(), str(e))

    return ("ok", stream.tell(), canonical(value))


def assert_parity(cdef: str, data: bytes, *, guard: int = 1 << 16) -> tuple:
    """Assert that both read paths observe the same outcome, and return that outcome."""
    outcomes = []
    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        assert verify_compiled(cs.test, compiled)
        outcomes.append(observe(cs.test, data, guard=guard))

    assert outcomes[0] == outcomes[1], f"compiled and interpreted paths diverged: {outcomes}"
    return outcomes[0]


# ---------------------------------------------------------------------------
# Negative and non-integer lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cdef", "data"),
    [
        # A negative length field, signed so it can be negative
        ("struct test { int8 n; uint8 data[n]; };", b"\xff\x01\x02"),
        # A length expression that goes below zero, which used to be clamped to an empty array
        ("struct test { uint8 n; char data[n - 4]; };", b"\x01\x41\x42"),
        # A negative length on a zero-width element type
        ("struct empty { }; struct test { int8 n; empty data[n]; };", b"\xff"),
    ],
    ids=["negative-field", "negative-expression", "negative-zero-width"],
)
def test_negative_length_rejected(cdef: str, data: bytes) -> None:
    """A negative array length is a malformed record, not an empty array."""
    outcome = assert_parity(cdef, data)
    assert outcome[0] == "ArraySizeError"
    assert "negative" in outcome[2]


def test_negative_length_is_not_an_empty_array() -> None:
    """A negative length must not silently produce a complete looking empty array."""
    for compiled in (True, False):
        cs = cstruct().load("struct test { int8 n; uint8 data[n]; uint16 tail; };", compiled=compiled)

        with pytest.raises(ArraySizeError, match="a length can't be negative"):
            cs.test(b"\xff\x01\x00")


@pytest.mark.parametrize(
    "cdef",
    ["#define N -1\nstruct test { uint8 data[N]; };", "struct test { uint8 data[-1]; };"],
    ids=["define", "literal"],
)
def test_static_negative_length_rejected(cdef: str) -> None:
    """A static array length is validated when the definition is loaded."""
    with pytest.raises(ArraySizeError, match="a length can't be negative"):
        cstruct().load(cdef)


# ---------------------------------------------------------------------------
# Over-budget lengths: reject at the evidence offset, don't amplify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cdef", "count", "expected"),
    [
        # A byte array reports the declared entry count
        ("struct test { uint32 n; char data[n]; };", 1 << 30, 1 << 30),
        # A multi-byte integer array reports the byte length the read reached for
        ("struct test { uint32 n; uint16 data[n]; };", 1 << 30, 1 << 31),
        # A product of two header fields, the classic amplification expression
        ("struct test { uint32 a; uint32 b; uint8 data[a * b]; };", 1 << 16, (1 << 16) * (1 << 16)),
    ],
    ids=["char", "uint16", "product"],
)
def test_overbudget_length_fails_at_evidence_offset(cdef: str, count: int, expected: int) -> None:
    """An over-budget length must fail at the array's offset, not after pre-allocating it."""
    product = "a * b" in cdef
    headers = count.to_bytes(4, "little") * (2 if product else 1)
    data = headers + b"\x01\x02\x03"

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        stream = GuardedStream(data)

        with pytest.raises(EOFError, match=rf"expected {expected}$") as exc_info:
            cs.test(stream)

        # The untrusted declared size is reported, the actual amount read is what the evidence held
        assert str(exc_info.value).startswith("Read 3 bytes, but expected ")
        # The parser consumed the header and every available evidence byte, never past the buffer
        assert stream.tell() == len(data)
        # The array is only ever read against the evidence, never in the declared size
        assert stream.requests[-1] == 3
        assert max(stream.requests) <= max(len(headers), 3)


def test_extreme_length_fails_with_eof_not_memory_error() -> None:
    """A length that can't even be an index must be a truncation error, not an implementation leak."""
    for compiled in (True, False):
        cs = cstruct().load("struct test { uint64 n; uint32 data[n]; };", compiled=compiled)

        with pytest.raises(EOFError, match="expected 36893488147419103232"):
            cs.test(GuardedStream((2**63).to_bytes(8, "little")))


def test_large_static_array_is_not_read_in_one_request() -> None:
    """A huge static array is also read against the evidence, not in a single allocation."""
    cdef = "struct test { char data[100000000]; uint8 tail; };"
    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        stream = GuardedStream(b"\x01\x02\x03")

        with pytest.raises(EOFError, match=r"expected 10000000[01]"):
            cs.test(stream)

        assert stream.tell() == 3
        assert stream.requests == [3]


# ---------------------------------------------------------------------------
# Zero-width elements: an explicit budget
# ---------------------------------------------------------------------------


def test_zero_width_array_budget() -> None:
    """An element type that consumes no bytes can't be amplified by a length field."""
    cdef = "struct empty { }; struct test { uint32 n; empty data[n]; };"
    over = MAX_ZERO_WIDTH_ARRAY_ENTRIES + 1

    outcome = assert_parity(cdef, over.to_bytes(4, "little"))
    assert outcome[0] == "ArraySizeError"
    assert "zero-width" in outcome[2]
    # Only the header is read, no element was allocated
    assert outcome[1] == 4


def test_zero_width_array_within_budget() -> None:
    """A zero-width element array within the budget still parses, on both paths."""
    cdef = "struct empty { }; struct test { uint16 n; empty data[n]; };"
    data = struct.pack("<H", 3)

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        obj = cs.test(data)

        assert len(obj.data) == 3
        assert obj.dumps() == data


def test_void_array_is_not_amplified(cs: cstruct) -> None:
    """A void element consumes no bytes, and a void array never materialises its entries."""
    cs.load("struct test { uint32 n; void data[n]; };")

    assert cs.test((0).to_bytes(4, "little")).data == []
    assert cs.test((1 << 24).to_bytes(4, "little")).data == []


# ---------------------------------------------------------------------------
# Legal zero length and normal arrays keep parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cdef", "data", "expected"),
    [
        ("struct test { uint8 n; uint8 data[n]; uint16 tail; };", b"\x00\x01\x00", ("data", [])),
        ("struct test { uint8 n; char data[n]; uint16 tail; };", b"\x00\x01\x00", ("data", b"")),
        ("struct test { uint8 n; uint16 data[n]; uint16 tail; };", b"\x00\x01\x00", ("data", [])),
    ],
    ids=["uint8", "char", "uint16"],
)
def test_zero_length_array_parses(cdef: str, data: bytes, expected: tuple) -> None:
    """A legal zero-length array consumes no bytes and doesn't shift the following fields."""
    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        obj = cs.test(data)

        assert getattr(obj, expected[0]) == expected[1]
        assert obj.tail == 1
        assert obj.dumps() == data


@pytest.mark.parametrize(
    ("cdef", "count", "width"),
    [
        ("struct test { uint8 n; uint8 data[n]; uint16 tail; };", 5, 1),
        ("struct test { uint8 n; uint16 data[n]; uint16 tail; };", 5, 2),
        ("struct test { uint8 n; uint32 data[n]; uint16 tail; };", 5, 4),
    ],
    ids=["uint8", "uint16", "uint32"],
)
def test_normal_array_parses(cdef: str, count: int, width: int) -> None:
    """A normal, evidence backed array still parses on both paths."""
    payload = bytes(range(count * width))
    data = bytes([count]) + payload + b"\xbe\xef"

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        stream = GuardedStream(data)
        obj = cs.test(stream)

        assert len(obj.data) == count
        assert obj.tail == 0xEFBE
        assert stream.tell() == len(data)


# ---------------------------------------------------------------------------
# Failure never yields a partial array, nor pollutes the next field
# ---------------------------------------------------------------------------


def test_short_read_does_not_yield_partial_array() -> None:
    """A short read must fail, not return the entries that happened to fit."""
    cdef = "struct test { uint8 n; uint16 data[n]; uint16 tail; };"

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)

        # 4 entries of 2 bytes are declared, only 3 bytes are available
        with pytest.raises(EOFError, match="expected 8"):
            cs.test(GuardedStream(b"\x04\x01\x02\x03"))


def test_failure_leaves_stream_replayable() -> None:
    """After a failed array read, the stream is left where the failure was detected."""
    cdef = "struct test { uint32 n; char data[n]; uint16 tail; };"
    malformed = (1 << 30).to_bytes(4, "little") + b"abc"

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)

        first = io.BytesIO(malformed)
        with pytest.raises(EOFError):
            cs.test(first)
        first_offset = first.tell()

        replay = io.BytesIO(malformed)
        with pytest.raises(EOFError):
            cs.test(replay)

        assert replay.tell() == first_offset == len(malformed)


def test_failed_record_does_not_consume_next_record() -> None:
    """A failed read of one record must not read the start of the next record."""
    cdef = "struct test { uint32 n; char data[n]; };"
    valid = (3).to_bytes(4, "little") + b"abc"
    truncated = (10).to_bytes(4, "little") + b"xy"
    next_record = b"\xa5\xa5\xa5\xa5"

    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        stream = io.BytesIO(valid + truncated + next_record)

        first = cs.test(stream)
        assert first.n == 3
        assert first.data == b"abc"
        assert stream.tell() == len(valid)

        with pytest.raises(EOFError, match="but expected 10"):
            cs.test(stream)

        # The stream is where the truncation was found, so re-reading fails identically
        offset = stream.tell()
        stream.seek(len(valid))
        with pytest.raises(EOFError, match="but expected 10"):
            cs.test(stream)
        assert stream.tell() == offset


# ---------------------------------------------------------------------------
# Nested dynamic structures share the same semantics
# ---------------------------------------------------------------------------


def test_nested_dynamic_structure_lengths() -> None:
    """A nested dynamic structure uses the same length semantics as a top-level one."""
    cdef = """
    struct inner {
        int8    n;
        uint8   data[n];
    };
    struct test {
        uint8   m;
        inner   entries[m];
    };
    """

    # A negative length inside a nested entry is rejected too
    for compiled in (True, False):
        cs = cstruct().load(cdef, compiled=compiled)
        assert verify_compiled(cs.test, compiled)

        with pytest.raises(ArraySizeError, match="a length can't be negative"):
            cs.test(b"\x01\xff")


def test_nested_dynamic_structure_normal() -> None:
    """A normal nested dynamic structure parses identically on both paths."""
    cdef = """
    struct inner {
        uint8   n;
        uint8   data[n];
    };
    struct test {
        uint8   m;
        inner   entries[m];
    };
    """
    data = b"\x03" + b"\x01\x41" + b"\x02\x42\x43" + b"\x00"

    outcome = assert_parity(cdef, data)
    assert outcome[0] == "ok"

    # Validate the parsed evidence with a concrete reader too
    cs = cstruct().load(cdef)
    value = cs.test(data)
    assert [bytes(entry.data) for entry in value.entries] == [b"A", b"BC", b""]


# ---------------------------------------------------------------------------
# Every element type shares the same length semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("type_name", ["char", "wchar", "uint8", "uint16", "uint32", "float"])
def test_negative_length_rejected_for_every_element_type(type_name: str) -> None:
    """Every array element type rejects a negative length, instead of some reading until EOF."""
    cs = cstruct()

    with pytest.raises(ArraySizeError, match="a length can't be negative"):
        getattr(cs, type_name)[-1]


@pytest.mark.parametrize("dtype", ["char", "wchar", "uint16", "uint32"])
def test_dynamic_negative_length_rejected_for_every_element_type(dtype: str) -> None:
    """A negative dynamic length is rejected at the evidence offset for every element type."""
    cdef = f"struct test {{ int8 n; {dtype} data[n]; uint16 tail; }};"

    outcome = assert_parity(cdef, b"\xff\x01\x00")
    assert outcome[0] == "ArraySizeError"
    assert outcome[1] == 1
    assert "negative" in outcome[2]


def test_multidimensional_array_shares_semantics() -> None:
    """Multi-dimensional arrays use the same length semantics on both paths."""
    # A normal nested array parses
    outcome = assert_parity("struct test { uint8 a[2][3]; uint8 tail; };", bytes(range(1, 8)))
    assert outcome[0] == "ok"

    # A zero-length dimension is legal and consumes no bytes
    outcome = assert_parity("struct test { uint8 a[2][0]; uint8 tail; };", b"\x01")
    assert outcome[0] == "ok"

    # A zero-width element in a nested array is also budgeted, not amplified
    cdef = "struct empty { }; struct test { uint32 n; empty a[n][5]; };"
    outcome = assert_parity(cdef, (MAX_ZERO_WIDTH_ARRAY_ENTRIES + 1).to_bytes(4, "little"))
    assert outcome[0] == "ArraySizeError"
    assert "zero-width" in outcome[2]
