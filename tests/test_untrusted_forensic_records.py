"""Tests for parsing untrusted, length-prefixed forensic records from finite input.

Forensic record formats commonly combine three unbounded-looking constructs:

* variable-length integers (``uleb128``/``ileb128``) whose length is signalled
  by continuation bytes instead of a fixed width,
* zero-terminated fields (``char name[]`` / ``wchar name[]``) whose length is
  signalled by a null terminator, and
* dynamic arrays (``char data[length]``) whose length is declared by a
  previously parsed field.

A hostile or simply truncated record can try to drive any of these into a read
that never terminates, or into consuming bytes past the end of a record so that
the start of the next record is lost. The tests below feed deliberately finite
malformed samples to the parser and assert that:

1. every malformed sample terminates with an ``EOFError`` on both the compiled
   and the interpreted code path,
2. the exception message and the evidence offset (the stream position where the
   read gave up) are identical on both paths,
3. the stream is left at a deterministic position so a second attempt fails at
   the same offset and a valid record preceding the malformed one survives, and
4. an over-budget length declaration never produces a partial, "complete"
   looking field.

Valid multi-byte integers and well-formed adjacent records are included as
controls, so the malformed samples exercise genuine parsing logic rather than
an always-failing stub.
"""

from __future__ import annotations

import io
import signal

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.types.base import BaseType
from tests.util import NonSeekableStream

# A bound far larger than any read in these tests needs: if any of the samples
# below can force an unbounded read, the alarm fires instead of hanging pytest.
TERMINATION_DEADLINE_SECONDS = 5

# Lengths that vastly exceed the bytes that follow, encoded as little endian.
OVERSIZE_UINT32 = (1 << 30).to_bytes(4, "little")
OVERSIZE_UINT16 = (1 << 15).to_bytes(2, "little")


class _Deadline:
    """Raise :class:`AssertionError` if a parser call does not return in time."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._previous = None

    def _handler(self, signum: int, frame: object) -> None:
        raise AssertionError("Parser did not terminate on finite input")

    def __enter__(self) -> _Deadline:
        self._previous = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc: object) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._previous)


def _uleb128(value: int) -> bytes:
    """Encode an unsigned integer as an unsigned LEB128 byte string."""
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(0x80 | byte)
        else:
            result.append(byte)
            return bytes(result)


def _parse_at(type_: type, data: bytes) -> tuple[object, type[BaseException], str, int]:
    """Parse ``data`` and return ``(result_or_None, exception_type, message, offset)``.

    The evidence offset is the position of the stream when the parser gives up.
    A successful parse returns ``None`` for the exception fields.
    """
    stream = io.BytesIO(data)
    try:
        result = type_(stream)
    except EOFError as e:
        return None, EOFError, str(e), stream.tell()
    return result, type(None), "", stream.tell()


# ---------------------------------------------------------------------------
# Variable-length integers: continuation bytes must not enable an unbounded read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("type_name", ["uleb128", "ileb128"])
@pytest.mark.parametrize("continuation_run", [1, 2, 4, 16, 64])
def test_varint_continuation_run_terminates(
    cs: cstruct, type_name: str, continuation_run: int
) -> None:
    """A finite run of continuation bytes with no final byte must raise ``EOFError``."""
    sample = b"\x80" * continuation_run

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError, match="EOF reached, while final LEB128 byte was not yet read"):
            getattr(cs, type_name)(sample)

    # The read must have consumed exactly the available bytes, no more and no less
    stream = io.BytesIO(sample)
    with pytest.raises(EOFError):
        getattr(cs, type_name)(stream)
    assert stream.tell() == continuation_run


@pytest.mark.parametrize("type_name", ["uleb128", "ileb128"])
def test_varint_empty_input(cs: cstruct, type_name: str) -> None:
    """An empty stream is an immediate, bounded failure."""
    with pytest.raises(EOFError, match="EOF reached, while final LEB128 byte was not yet read"):
        getattr(cs, type_name)(b"")


@pytest.mark.parametrize("type_name", ["uleb128", "ileb128"])
def test_varint_valid_long_integer_control(cs: cstruct, type_name: str) -> None:
    """Control: legal multi-byte integers still decode, proving the failure samples hit real logic."""
    # 13371337 takes four bytes; this is the "legal longer integer" counterpart of the runs above.
    assert getattr(cs, type_name)(b"\xc9\x8f\xb0\x06") == 13371337

    # A legal multi-byte integer followed by a truncated one must not swallow the boundary between them
    stream = io.BytesIO(b"\x8b\x25" + b"\x80\x80")
    assert getattr(cs, type_name)(stream) == 4747
    with pytest.raises(EOFError):
        getattr(cs, type_name)(stream)
    assert stream.tell() == 4


# ---------------------------------------------------------------------------
# Zero-terminated fields: a missing terminator must not enable an unbounded read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream_type", [io.BytesIO, NonSeekableStream])
@pytest.mark.parametrize(
    ("type_name", "sample", "expected_message"),
    [
        ("char", b"abcdef", "Read 0 bytes, but expected 1"),
        ("wchar", "abc".encode("utf-16-le"), "Read 0 bytes, but expected 2"),
        # A single dangling byte for a wide string is a partial item
        ("wchar", "ab".encode("utf-16-le") + b"\x41", "Read 1 bytes, but expected 2"),
    ],
)
def test_zero_terminated_field_missing_terminator(
    cs: cstruct, stream_type: type, type_name: str, sample: bytes, expected_message: str
) -> None:
    """Both seekable chunked reading and non-seekable bytewise reading must terminate."""
    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError, match=expected_message):
            getattr(cs, type_name)[None](stream_type(sample))


@pytest.mark.parametrize("type_name", ["char", "wchar"])
def test_zero_terminated_field_control(cs: cstruct, type_name: str) -> None:
    """Control: a present terminator ends the field and leaves the following bytes untouched."""
    if type_name == "char":
        payload, terminator, expected = b"name", b"\x00", b"name"
    else:
        payload, terminator, expected = "name".encode("utf-16-le"), b"\x00\x00", "name"

    stream = io.BytesIO(payload + terminator + b"trailing")
    assert getattr(cs, type_name)[None](stream) == expected
    assert stream.read() == b"trailing"


# ---------------------------------------------------------------------------
# Dynamic arrays: an over-budget declared length must not fabricate a full field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "length_bytes", "expected_in_message", "reported_bytes"),
    [
        # Byte arrays report the declared element count directly
        ("uint32 length; char data[length];", OVERSIZE_UINT32, 1 << 30, 3),
        ("uint16 length; char data[length];", OVERSIZE_UINT16, 1 << 15, 3),
        # Integer arrays report the byte length the read reached for: count * element width
        ("uint32 count; uint16 values[count];", OVERSIZE_UINT32, 1 << 31, 3),
    ],
)
def test_dynamic_array_overbudget_length(
    cs: cstruct,
    declaration: str,
    length_bytes: bytes,
    expected_in_message: int,
    reported_bytes: int,
) -> None:
    """An array length that exceeds the remaining bytes must fail, never return a partial array."""
    cs.load(f"struct record {{ {declaration} }};")

    trailing = b"\x01\x02\x03"
    sample = length_bytes + trailing

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError) as exc_info:
            cs.record(sample)

    # The expected figure is the untrusted declared size, the reported count what the stream held
    assert f"expected {expected_in_message}" in str(exc_info.value)
    assert f"Read {reported_bytes} bytes" in str(exc_info.value)

    # No structure object with a "complete" field may escape the failed parse
    stream = io.BytesIO(sample)
    with pytest.raises(EOFError):
        cs.record(stream)
    # The parser consumed the length declaration and every available byte, but never reached past
    # the finite buffer, so no following record can be silently overrun
    assert stream.tell() == len(sample)


def test_dynamic_array_zero_length_control(cs: cstruct) -> None:
    """Control: a zero-length dynamic array consumes no bytes and parsing continues normally."""
    cs.load(
        """
        struct record {
            uint32 length;
            char   data[length];
            uint8  marker;
        };
        """
    )
    obj = cs.record((0).to_bytes(4, "little") + b"\x2a")
    assert obj.length == 0
    assert obj.data == b""
    assert obj.marker == 0x2A


# ---------------------------------------------------------------------------
# Combined forensic records and parity between the compiled and interpreted paths
# ---------------------------------------------------------------------------

FORENSIC_CDEF = """
struct forensic_record {
    uint32 length;
    char   data[length];
    char   label[];
};
"""

# A finite malformed sample and the offset where a correct parser must give up.
# Each tuple is (sample, expected evidence offset, substring of the EOFError).
MALFORMED_DYNAMIC_RECORDS = [
    pytest.param(
        OVERSIZE_UINT32 + b"abc",
        7,
        "Read 3 bytes, but expected 1073741824",
        id="overbudget-array-length",
    ),
    pytest.param(
        (3).to_bytes(4, "little") + b"abcxy",
        9,
        "Read 0 bytes, but expected 1",
        id="missing-null-terminator",
    ),
    pytest.param(
        (3).to_bytes(4, "little") + b"ab",
        6,
        "Read 2 bytes, but expected 3",
        id="truncated-array-body",
    ),
    pytest.param(
        b"\x01\x02",
        2,
        "Read 2 bytes, but expected 4",
        id="truncated-length-field",
    ),
    pytest.param(
        b"",
        0,
        "Read 0 bytes, but expected 4",
        id="empty-input",
    ),
]


@pytest.mark.parametrize("sample, expected_offset, expected_message", MALFORMED_DYNAMIC_RECORDS)
def test_compiled_and_interpreted_path_parity(
    cs: cstruct, compiled: bool, sample: bytes, expected_offset: int, expected_message: str
) -> None:
    """Both paths must terminate with the same exception source, message and evidence offset."""
    cs.load(FORENSIC_CDEF, compiled=compiled)
    assert cs.forensic_record.__compiled__ is compiled

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        result, error_type, message, offset = _parse_at(cs.forensic_record, sample)

    assert result is None
    assert error_type is EOFError
    assert message == expected_message
    assert offset == expected_offset


def test_parity_between_paths_is_actually_observed(cs: cstruct) -> None:
    """Explicitly compare both paths against each other for every malformed sample."""
    samples = [param.values[0] for param in MALFORMED_DYNAMIC_RECORDS]
    outcomes = {}
    for compiled in (True, False):
        parser = cstruct()
        parser.load(FORENSIC_CDEF, compiled=compiled)
        assert parser.forensic_record.__compiled__ is compiled
        outcomes[compiled] = [_parse_at(parser.forensic_record, sample) for sample in samples]

    for interpreted, compiled in zip(outcomes[False], outcomes[True], strict=True):
        assert interpreted == compiled


def test_failed_record_does_not_lose_next_record_start(cs: cstruct, compiled: bool) -> None:
    """A malformed record must leave the stream at a deterministic, replayable position."""
    cs.load(
        """
        struct record {
            uint32 length;
            char   data[length];
        };
        """,
        compiled=compiled,
    )

    valid = (3).to_bytes(4, "little") + b"abc"
    truncated = (10).to_bytes(4, "little") + b"xy"
    buffer = valid + truncated

    stream = io.BytesIO(buffer)
    first = cs.record(stream)
    assert first.length == 3 and first.data == b"abc"
    assert stream.tell() == len(valid)

    with pytest.raises(EOFError, match="Read 2 bytes, but expected 10"):
        cs.record(stream)
    first_failure_offset = stream.tell()

    # Replaying the whole buffer must fail at exactly the same evidence offset
    replay = io.BytesIO(buffer)
    cs.record(replay)
    with pytest.raises(EOFError, match="Read 2 bytes, but expected 10"):
        cs.record(replay)
    assert replay.tell() == first_failure_offset == len(buffer)


# ---------------------------------------------------------------------------
# LEB128-based records: continuation runs combined with declared array lengths
# ---------------------------------------------------------------------------

LEB_FORENSIC_CDEF = """
struct leb_forensic_record {
    uleb128 length;
    char    data[length];
    char    label[];
};
"""

MALFORMED_LEB_RECORDS = [
    pytest.param(
        b"\x80\x80\x80\x80", 4, "EOF reached, while final LEB128 byte was not yet read",
        id="continuation-run-no-final-byte",
    ),
    pytest.param(
        b"\x80", 1, "EOF reached, while final LEB128 byte was not yet read", id="single-continuation-byte",
    ),
    pytest.param(b"", 0, "EOF reached, while final LEB128 byte was not yet read", id="empty-input"),
    pytest.param(b"\x05ab", 3, "Read 2 bytes, but expected 5", id="valid-length-truncated-array"),
    pytest.param(b"\x03abcxy", 6, "Read 0 bytes, but expected 1", id="valid-array-missing-terminator"),
]


@pytest.mark.parametrize("sample, expected_offset, expected_message", MALFORMED_LEB_RECORDS)
def test_leb_record_termination(
    cs: cstruct, compiled: bool, sample: bytes, expected_offset: int, expected_message: str
) -> None:
    """Continuation bytes and missing terminators must terminate on finite LEB128 records."""
    cs.load(LEB_FORENSIC_CDEF, compiled=compiled)
    # LEB128 is not supported by the compiler, so this structure always uses the interpreted reader
    assert cs.leb_forensic_record.__compiled__ is False

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        result, error_type, message, offset = _parse_at(cs.leb_forensic_record, sample)

    assert result is None
    assert error_type is EOFError
    assert message == expected_message
    assert offset == expected_offset


def test_leb_overbudget_length_is_not_completed(cs: cstruct) -> None:
    """A huge LEB128 length declaration must not be turned into a complete forensic field."""
    cs.load("struct leb_record { uleb128 length; char data[length]; };")

    declared = 1 << 40
    sample = _uleb128(declared) + b"abc"

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError, match=rf"Read 3 bytes, but expected {declared}"):
            cs.leb_record(sample)


def test_leb_adjacent_records_control(cs: cstruct) -> None:
    """Control: a legal long LEB128 integer and a normal adjacent record parse back-to-back."""
    cs.load(
        """
        struct leb_record {
            uleb128 length;
            char    data[length];
        };
        struct frame {
            leb_record first;
            leb_record second;
        };
        """
    )

    # 4747 needs the two-byte 0x8b 0x25 encoding, followed by a short second record
    buffer = b"\x8b\x25" + b"A" * 4747 + b"\x02BC"
    frame = cs.frame(buffer)
    assert frame.first.length == 4747
    assert frame.first.data == b"A" * 4747
    assert frame.second.length == 2
    assert frame.second.data == b"BC"
    assert frame.dumps() == buffer


# ---------------------------------------------------------------------------
# Stacked adversities and null-terminated arrays of structured values
# ---------------------------------------------------------------------------


def test_stacked_continuation_and_missing_terminator(cs: cstruct) -> None:
    """Continuation bytes, a length declaration and a missing terminator stacked in one finite sample."""
    cs.load(
        """
        struct entry {
            uint16 value;
            char   tag[];
        };
        struct container {
            uleb128 count;
            entry   entries[count];
        };
        """
    )

    # Declare two entries, provide one full entry and a second entry with an unterminated tag
    sample = _uleb128(2) + (7).to_bytes(2, "little") + b"first\x00" + (8).to_bytes(2, "little") + b"se"

    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError, match="Read 0 bytes, but expected 1"):
            cs.container(sample)


def test_null_terminated_integer_array_missing_terminator(cs: cstruct, compiled: bool) -> None:
    """A null-terminated integer array must stop at EOF instead of scanning for ever."""
    cs.load("struct table { uint32 values[]; };", compiled=compiled)

    sample = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError, match="Read 0 bytes, but expected 4"):
            cs.table(sample)

    # The control: a real zero terminator ends the array and trailing data remains available
    stream = io.BytesIO((1).to_bytes(4, "little") + (0).to_bytes(4, "little") + b"rest")
    parsed = cs.table(stream)
    assert list(parsed.values) == [1]
    assert stream.read() == b"rest"


# ---------------------------------------------------------------------------
# Non-seekable streams and allocation bounds
# ---------------------------------------------------------------------------


def test_zero_terminated_field_non_seekable_parity(cs: cstruct) -> None:
    """Null-terminated byte and wide fields must fail identically on non-seekable streams."""
    samples = [
        (cs.char[None], b"abcdef", "Read 0 bytes, but expected 1"),
        (cs.wchar[None], "abc".encode("utf-16-le"), "Read 0 bytes, but expected 2"),
        (cs.uleb128, b"\x80" * 8, "EOF reached, while final LEB128 byte was not yet read"),
    ]
    for type_, sample, message in samples:
        with _Deadline(TERMINATION_DEADLINE_SECONDS):
            with pytest.raises(EOFError, match=message):
                type_(NonSeekableStream(sample))


def test_overbudget_declaration_does_not_pre_allocate(cs: cstruct, compiled: bool) -> None:
    """A gigantic declared count must fail immediately instead of allocating the declared capacity."""
    cs.load(
        """
        struct record {
            uint32 count;
            uint8  values[count];
        };
        """,
        compiled=compiled,
    )

    # 2**31 entries would need gigabytes if the parser pre-allocated the array; a bounded parser
    # reads against the finite stream and raises before any such allocation can matter
    impossible = (1 << 31).to_bytes(4, "little")
    with _Deadline(TERMINATION_DEADLINE_SECONDS):
        with pytest.raises(EOFError):
            cs.record(io.BytesIO(impossible + b"\x00"))
