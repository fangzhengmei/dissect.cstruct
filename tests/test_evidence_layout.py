"""Tests that pin down the byte level layout of a mixed endianness, bit field and union based structure.

The sample definition :data:`EVIDENCE_CDEF` models a forensic "evidence record". It deliberately combines the
constructs that make a struct definition fragile:

* bit fields that share a storage unit (``version`` and ``flags`` in one ``uint16``, ``tag`` and ``kind`` in another
  with unused boundary bits, and ``window``/``span`` sharing one ``uint32``),
* bit fields that span multiple bytes (``tag`` is 10 bits wide, ``window`` is 20 bits wide),
* consecutive bit field units of different base types (``uint8``, ``uint16`` and ``uint32``) that must each start a
  new storage unit, also when the previous unit still has room,
* a union with overlapping interpretations, including a bit field view of the first two bytes,
* signed fields, and enum and flag fields with explicitly signed or unsigned base types,
* explicit padding fields and, in aligned mode, compiler inserted padding.

The tests verify:

1. Layout stability: field offsets, sizes and bit positions are what the definition says, and the interpreted and
   compiled readers/writers agree on values, positions and the bytes they write back.
2. Failure behaviour: truncated input and non-seekable streams never yield a partially interpreted record, and a
   framing mistake is never silently completed from the bytes of the next record.
3. Error attribution: a small diagnosis helper distinguishes a layout error from an endianness error and from a bit
   width error, using the same sample for all three.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.structure import StructureMetaType, UnionMetaType, UnionProxy
from tests.util import NonSeekableStream, verify_compiled

EVIDENCE_CDEF = """
#define EVIDENCE_MAGIC 0x4556

flag EvidenceFlags : uint16 {
    NONE = 0,
    FILE = 1,
    DIRECTORY = 2,
    SYMLINK = 4,
    DELETED = 8,
    CARVED = 0x10
};

enum EvidenceKind : int16 {
    UNKNOWN = 0,
    CONTENT = 1,
    METADATA = 2,
    REDACTED = -2
};

union EvidenceValue {
    uint32 raw;
    int32  signed_raw;
    struct {
        uint16 low;
        uint16 high;
    } halves;
    uint8  octets[4];
    struct {
        uint16 low : 4;
        uint16 high : 12;
    } nibbles;
};

struct EvidenceBits {
    uint8  present : 1;
    uint8  compressed : 1;
    uint8  encrypted : 2;
    uint8  reserved : 3;
    uint16 tag : 10;
    uint16 kind : 4;
    uint8  parity;
    uint32 window : 20;
    uint32 span : 12;
};

struct EvidenceRecord {
    uint8  magic[2];
    uint16 version : 3;
    EvidenceFlags flags : 5;
    uint8  pad;
    int8   signed_byte;
    int16  delta;
    EvidenceKind kind;
    uint32 size;
    int32  offset;
    EvidenceValue value;
    EvidenceBits bits;
    int16  trailer;
};
"""

EVIDENCE_LOG_CDEF = (
    EVIDENCE_CDEF
    + """
struct EvidenceLog {
    uint16 count;
    EvidenceRecord records[count];
};
"""
)

# Canonical records that encode the same values in both endiannesses.
LITTLE_ENDIAN_RECORD = bytes.fromhex("4556ae00a580fefffeff4433221118fcffffefcdab8909bc365a45238167fdff")
BIG_ENDIAN_RECORD = bytes.fromhex("4556d500a580fffefffe11223344fffffc1889abcdefa0af345a12345678fffd")
GOLDEN_RECORDS = {"<": LITTLE_ENDIAN_RECORD, ">": BIG_ENDIAN_RECORD}

# The same values in an aligned structure, with the padding inserted by aligned mode.
ALIGNED_RECORDS = {
    "<": bytes.fromhex("4556ae00a580fefffeff00004433221118fcffffefcdab890900bc365a00000045238167fdff0000"),
    ">": bytes.fromhex("4556d500a580fffefffe000011223344fffffc1889abcdefa000af345a00000012345678fffd0000"),
}

RECORD_SIZE = 32
ALIGNED_RECORD_SIZE = 40
BITS_SIZE = 8
UNION_SIZE = 4
UNION_OFFSET = 18
BITS_OFFSET = 22


def load(
    endian: str = "<",
    *,
    compiled: bool = False,
    align: bool = False,
    definition: str = EVIDENCE_CDEF,
) -> cstruct:
    cs = cstruct(endian=endian)
    cs.load(definition, compiled=compiled, align=align)
    return cs


def load_record(endian: str, *, compiled: bool, align: bool = False) -> StructureMetaType:
    return load(endian, compiled=compiled, align=align).EvidenceRecord


@dataclass(frozen=True)
class Slot:
    """The storage of a field, as derived from the structure definition."""

    path: str
    """Dotted path of the field, e.g. ``bits.tag``."""

    storage: int
    """Byte offset of the field, or of the storage unit for bit fields."""

    size: int
    """Size in bytes of the field, or of the storage unit for bit fields."""

    bits: int | None = None
    """Amount of bits of a bit field, ``None`` for regular fields."""

    position: int = 0
    """Bit position within the storage unit, counted from the least significant bit of the unit."""

    signed: bool = False
    """Whether the storage type is signed."""


def _unwrap(type_: type) -> type:
    """Return the base storage type of an enum or flag, or the type itself."""
    return type_.__type__ if isinstance(type_, EnumMetaType) else type_


def _is_signed(type_: type) -> bool:
    """Return whether a storage type is signed, both for ``Int`` and ``Packed`` based types."""
    if (signed := getattr(type_, "signed", None)) is not None:
        return bool(signed)

    fmt = getattr(type_, "__fmt__", None)
    # Packed integer formats are the lowercase characters, except for the float formats.
    return fmt is not None and fmt.islower() and fmt not in ("e", "f", "d", "s")


def derive_layout(type_: StructureMetaType, base: int = 0, prefix: str = "") -> dict[str, Slot]:
    """Derive the layout of a structure from its definition, independent of the read and write implementations.

    This mirrors the documented layout rules: regular fields are placed after the previous field, union members all
    start at the start of the union, and bit fields share a storage unit of their own type as long as it has room.
    """
    layout: dict[str, Slot] = {}
    is_union = isinstance(type_, UnionMetaType)

    cursor = base
    unit_type = None
    unit_bits = 0
    remaining = 0
    unit_start = 0
    position = 0

    for field in type_.__fields__:
        # Anonymous fields (e.g. a nested struct without a name) use their type name for their path.
        path = prefix + (field.name or field._name)
        field_type = _unwrap(field.type)
        explicit = base + field.offset if field.offset is not None else None

        if isinstance(field_type, StructureMetaType):
            start = explicit if explicit is not None else (base if is_union else cursor)
            layout[path] = Slot(path, start, field_type.__size__)
            layout.update(derive_layout(field_type, start, f"{path}."))

            if not is_union:
                cursor = start + field_type.__size__
            unit_type = None
            remaining = 0
            continue

        if field.bits is not None:
            # A new storage unit starts when the previous one is exhausted, when the base type changes, or for every
            # union member, which starts at the start of the union.
            if remaining <= 0 or unit_type is not field_type or is_union:
                unit_start = explicit if explicit is not None else (base if is_union else cursor)
                unit_type = field_type
                unit_bits = field_type.__size__ * 8
                remaining = unit_bits
                position = 0
                if not is_union:
                    cursor = unit_start + field_type.__size__

            layout[path] = Slot(
                path,
                unit_start,
                field_type.__size__,
                bits=field.bits,
                position=position,
                signed=_is_signed(field_type),
            )
            position += field.bits
            remaining -= field.bits
            if remaining < 0:
                raise ValueError(f"Straddled bit field in derived layout: {path}")
            continue

        start = explicit if explicit is not None else (base if is_union else cursor)
        layout[path] = Slot(path, start, field_type.__size__, signed=_is_signed(field_type))
        if not is_union:
            cursor = start + field_type.__size__
        unit_type = None
        remaining = 0

    return layout


# The exact layout the definition above promises. Changing this table is changing the on-disk format.
EXPECTED_LAYOUT = {
    "magic": Slot("magic", 0, 2),
    "version": Slot("version", 2, 2, bits=3, position=0),
    "flags": Slot("flags", 2, 2, bits=5, position=3),
    "pad": Slot("pad", 4, 1),
    "signed_byte": Slot("signed_byte", 5, 1, signed=True),
    "delta": Slot("delta", 6, 2, signed=True),
    "kind": Slot("kind", 8, 2, signed=True),
    "size": Slot("size", 10, 4),
    "offset": Slot("offset", 14, 4, signed=True),
    "value": Slot("value", UNION_OFFSET, 4),
    "value.raw": Slot("value.raw", UNION_OFFSET, 4),
    "value.signed_raw": Slot("value.signed_raw", UNION_OFFSET, 4, signed=True),
    "value.halves": Slot("value.halves", UNION_OFFSET, 4),
    "value.halves.low": Slot("value.halves.low", UNION_OFFSET, 2),
    "value.halves.high": Slot("value.halves.high", UNION_OFFSET + 2, 2),
    "value.octets": Slot("value.octets", UNION_OFFSET, 4),
    "value.nibbles": Slot("value.nibbles", UNION_OFFSET, 2),
    "value.nibbles.low": Slot("value.nibbles.low", UNION_OFFSET, 2, bits=4, position=0),
    "value.nibbles.high": Slot("value.nibbles.high", UNION_OFFSET, 2, bits=12, position=4),
    "bits": Slot("bits", BITS_OFFSET, BITS_SIZE),
    "bits.present": Slot("bits.present", BITS_OFFSET, 1, bits=1, position=0),
    "bits.compressed": Slot("bits.compressed", BITS_OFFSET, 1, bits=1, position=1),
    "bits.encrypted": Slot("bits.encrypted", BITS_OFFSET, 1, bits=2, position=2),
    "bits.reserved": Slot("bits.reserved", BITS_OFFSET, 1, bits=3, position=4),
    "bits.tag": Slot("bits.tag", BITS_OFFSET + 1, 2, bits=10, position=0),
    "bits.kind": Slot("bits.kind", BITS_OFFSET + 1, 2, bits=4, position=10),
    "bits.parity": Slot("bits.parity", BITS_OFFSET + 3, 1),
    "bits.window": Slot("bits.window", BITS_OFFSET + 4, 4, bits=20, position=0),
    "bits.span": Slot("bits.span", BITS_OFFSET + 4, 4, bits=12, position=20),
    "trailer": Slot("trailer", 30, 2, signed=True),
}

EXPECTED_TOP_LEVEL_FIELDS = (
    "magic",
    "version",
    "flags",
    "pad",
    "signed_byte",
    "delta",
    "kind",
    "size",
    "offset",
    "value",
    "bits",
    "trailer",
)

# The values both golden records encode. Values that depend on the byte order of an overlapping union view are in
# EXPECTED_UNION_VIEWS.
EXPECTED_VALUES: dict[str, int | list[int]] = {
    "magic": [0x45, 0x56],
    "version": 6,
    "flags": 0x15,  # FILE | SYMLINK | CARVED
    "pad": 0xA5,
    "signed_byte": -128,
    "delta": -2,
    "kind": -2,  # REDACTED
    "size": 0x11223344,
    "offset": -1000,
    "value.raw": 0x89ABCDEF,
    "value.signed_raw": -1985229329,
    "bits.present": 1,
    "bits.compressed": 0,
    "bits.encrypted": 2,
    "bits.reserved": 0,
    "bits.tag": 700,
    "bits.kind": 13,
    "bits.parity": 0x5A,
    "bits.window": 0x12345,
    "bits.span": 0x678,
    "trailer": -3,
}

EXPECTED_UNION_VIEWS = {
    "<": {
        "value.halves.low": 0xCDEF,
        "value.halves.high": 0x89AB,
        "value.octets": [0xEF, 0xCD, 0xAB, 0x89],
        "value.nibbles.low": 0xF,
        "value.nibbles.high": 0xCDE,
    },
    ">": {
        "value.halves.low": 0x89AB,
        "value.halves.high": 0xCDEF,
        "value.octets": [0x89, 0xAB, 0xCD, 0xEF],
        "value.nibbles.low": 0x8,
        "value.nibbles.high": 0x9AB,
    },
}

# Values that are encoded in the bytes of the field itself, so they survive an endianness change.
ENCODING_INDEPENDENT = (
    "magic",
    "version",
    "flags",
    "pad",
    "signed_byte",
    "delta",
    "kind",
    "size",
    "offset",
    "value.raw",
    "value.signed_raw",
    "bits.present",
    "bits.compressed",
    "bits.encrypted",
    "bits.reserved",
    "bits.tag",
    "bits.kind",
    "bits.parity",
    "bits.window",
    "bits.span",
    "trailer",
)


def expected_values(endian: str) -> dict[str, int | list[int]]:
    return {**EXPECTED_VALUES, **EXPECTED_UNION_VIEWS[endian]}


def snapshot(value: object, prefix: str = "") -> dict[str, int | list[int]]:
    """Return the values of all leaves of a structure, flattening unions and nested structures."""
    result: dict[str, int | list[int]] = {}
    for name in value.__class__.__lookup__:  # type: ignore[attr-defined]
        item = getattr(value, name)
        path = prefix + name
        if isinstance(item, UnionProxy):
            result.update(snapshot(item.__target__, f"{path}."))
        elif hasattr(item.__class__, "__lookup__"):
            result.update(snapshot(item, f"{path}."))
        elif isinstance(item, list):
            result[path] = [int(entry) for entry in item]
        elif hasattr(item, "value") and not hasattr(item, "dumps"):
            result[path] = int(item.value)
        else:
            result[path] = int(item)
    return result


def leaf_paths(layout: dict[str, Slot]) -> set[str]:
    """Return the paths of the fields that have no sub fields."""
    return {path for path in layout if not any(other.startswith(f"{path}.") for other in layout)}


def swap(value: int, size: int, signed: bool) -> int:
    """Return the value as if its bytes were read with the other endianness."""
    return int.from_bytes(value.to_bytes(size, "little", signed=signed), "big", signed=signed)


def leaf_owners(layout: dict[str, Slot], endian: str) -> dict[tuple[int, int], str]:
    """Map every bit field bit to the field that owns it, indexed by ``(byte offset, bit in byte)``."""
    owners = {}
    for slot in layout.values():
        if slot.bits is None:
            continue

        unit_bits = slot.size * 8
        for index in range(slot.bits):
            if endian == "<":
                bit = slot.position + index
                owners[(slot.storage + bit // 8, bit % 8)] = slot.path
            else:
                bit = unit_bits - slot.position - slot.bits + index
                owners[(slot.storage + slot.size - 1 - bit // 8, bit % 8)] = slot.path
    return owners


def regular_owners(layout: dict[str, Slot], byte: int) -> set[str]:
    """Return the non bit field leaves whose storage covers a byte."""
    result = set()
    for path in leaf_paths(layout):
        slot = layout[path]
        if slot.bits is None and slot.storage <= byte < slot.storage + slot.size:
            result.add(path)
    return result


def flip_byte(data: bytes, index: int, bit: int) -> bytes:
    result = bytearray(data)
    result[index] ^= 1 << bit
    return bytes(result)


def covered_bytes(layout: dict[str, Slot]) -> set[int]:
    """Return the byte offsets that are part of a leaf field."""
    covered = set()
    for path in leaf_paths(layout):
        slot = layout[path]
        covered.update(range(slot.storage, slot.storage + slot.size))
    return covered


def mutation_affected(data: bytes, encoded: bytes) -> set[int]:
    return {index for index, (before, after) in enumerate(zip(data, encoded, strict=True)) if before != after}


# A non-seekable stream that does support ``tell``, like a buffered network reader.
class TellOnlyStream:
    def __init__(self, data: bytes) -> None:
        self._fh = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._fh.read(size)

    def tell(self) -> int:
        return self._fh.tell()

    def seekable(self) -> bool:
        return False


# A stream that doesn't even support ``tell``.
class BareStream:
    def __init__(self, data: bytes) -> None:
        self._fh = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._fh.read(size)


# Error attribution

LAYOUT_ERROR = "layout"
ENDIANNESS_ERROR = "endianness"
BIT_WIDTH_ERROR = "bit-width"
VALUE_ERROR = "value"


def diagnose(
    record: StructureMetaType,
    data: bytes,
    *,
    endian: str,
    expected: dict[str, Slot] | None = None,
    reference: dict[str, int | list[int]] | None = None,
) -> set[str]:
    """Classify the problems of a structure definition against the reference layout and values.

    Returns a subset of:

    * ``"layout"``: a field is missing or extra, or its offset, size or kind (regular vs. bit field) differs. The
      definition describes a different on-disk format, so no value comparison can be meaningful.
    * ``"bit-width"``: the fields are at the same offsets, but a bit field covers different bits of its storage unit.
    * ``"endianness"``: the layout is correct, but every multi-byte value equals the byte swapped reference value.
    * ``"value"``: the layout is correct, but values can't be explained by either the reference or a byte swap, e.g.
      because the signedness of a field changed.
    """
    expected = expected or EXPECTED_LAYOUT
    reference = reference or expected_values(endian)

    problems: set[str] = set()
    actual = derive_layout(record)

    if set(actual) != set(expected):
        problems.add(LAYOUT_ERROR)

    for path, expected_slot in expected.items():
        slot = actual.get(path)
        if slot is None:
            problems.add(LAYOUT_ERROR)
            continue
        if (slot.storage, slot.size, slot.bits is None) != (
            expected_slot.storage,
            expected_slot.size,
            expected_slot.bits is None,
        ):
            problems.add(LAYOUT_ERROR)
        elif expected_slot.bits is not None and (slot.position, slot.bits) != (
            expected_slot.position,
            expected_slot.bits,
        ):
            problems.add(BIT_WIDTH_ERROR)

    try:
        values = snapshot(record(data, endian=endian))
    except EOFError:
        # The definition consumes a different amount of bytes, so even the framing is off.
        problems.add(LAYOUT_ERROR)
        return problems

    candidates = 0
    swapped = 0
    for path, value in reference.items():
        expected_slot = expected.get(path)
        if expected_slot is None or expected_slot.bits is not None or not isinstance(value, int):
            continue
        # Overlapping union views are re-derived from the raw value, so they can't fail independently.
        if path.startswith("value.") and path not in ("value.raw", "value.signed_raw"):
            continue

        actual_value = values.get(path)
        if actual_value == value:
            continue

        candidates += 1
        # A single byte field can't be explained by a byte swap, so a mismatch there is a plain value error.
        if expected_slot.size > 1 and actual_value == swap(value, expected_slot.size, expected_slot.signed):
            swapped += 1

    if candidates and candidates == swapped:
        problems.add(ENDIANNESS_ERROR)
    elif candidates:
        problems.add(VALUE_ERROR)

    return problems


# Layout stability


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_layout_is_stable(compiled: bool) -> None:
    """Test that the definition produces the documented offsets, sizes and bit positions."""
    cs = load(compiled=compiled)
    record = cs.EvidenceRecord

    assert verify_compiled(record, compiled)
    assert verify_compiled(cs.EvidenceBits, compiled)
    assert len(record) == RECORD_SIZE
    assert record.__alignment__ == 4
    assert len(cs.EvidenceBits) == BITS_SIZE
    assert len(cs.EvidenceValue) == UNION_SIZE

    assert tuple(field.name for field in record.__fields__) == EXPECTED_TOP_LEVEL_FIELDS
    assert derive_layout(record) == EXPECTED_LAYOUT

    # The definition constants survive and agree with the magic bytes of the sample.
    assert cs.consts["EVIDENCE_MAGIC"] == 0x4556
    assert GOLDEN_RECORDS["<"][:2] == GOLDEN_RECORDS[">"][:2] == b"EV"


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_layout_does_not_change_after_use(compiled: bool) -> None:
    """Test that committing or using the type doesn't move fields, also not after a lazy compile."""
    cs = load(compiled=compiled)
    record = cs.EvidenceRecord
    layout = derive_layout(record)
    assert layout == EXPECTED_LAYOUT

    data = GOLDEN_RECORDS["<"]
    assert record(data).dumps() == data
    record.commit()
    record(data)

    assert derive_layout(record) == layout
    assert len(record) == RECORD_SIZE
    # Regular fields keep their explicit offset, and the first field of a bit field unit marks where it starts.
    for field in record.__fields__:
        if field.offset is not None:
            assert field.offset == layout[field.name].storage


def test_layout_is_endianness_independent() -> None:
    """Test that the endianness doesn't move fields, it only changes how their bytes are interpreted."""
    little = load("<").EvidenceRecord
    big = load(">").EvidenceRecord

    assert derive_layout(little) == derive_layout(big) == EXPECTED_LAYOUT
    assert len(little) == len(big) == RECORD_SIZE


def test_layout_survives_cdef_roundtrip() -> None:
    """Test that rendering the definition and loading it again produces the same layout and behaviour."""
    reference = load()
    definition = reference.EvidenceRecord.cdef(recursive=True)

    for endian, data in GOLDEN_RECORDS.items():
        reloaded = load(endian, definition=definition).EvidenceRecord
        assert derive_layout(reloaded) == EXPECTED_LAYOUT
        assert len(reloaded) == RECORD_SIZE

        expected = snapshot(load(endian).EvidenceRecord(data))
        assert snapshot(reloaded(data)) == expected
        assert reloaded(data).dumps() == data


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_mixed_endianness_changes_values_not_positions(compiled: bool) -> None:
    """Test that the same bytes read with the other endianness keep every field at its position.

    Mixed endianness data is the main reason a forensic field can silently change meaning: the offsets are identical,
    so nothing about the layout looks wrong, while the values are byte swapped.
    """
    little = load("<", compiled=compiled).EvidenceRecord
    big = load(">", compiled=compiled).EvidenceRecord

    data = LITTLE_ENDIAN_RECORD

    # The same bytes at the same places, only interpreted with another byte order.
    assert derive_layout(little) == derive_layout(big) == EXPECTED_LAYOUT
    assert len(little(data)) == len(big(data)) == RECORD_SIZE

    little_values = snapshot(little(data))
    big_values = snapshot(big(data))

    # Every regular multi-byte field is byte swapped, everything else keeps its position and its bytes.
    for path, value in little_values.items():
        slot = EXPECTED_LAYOUT[path]
        if slot.bits is not None or isinstance(value, list) or slot.size == 1:
            continue

        if slot.size > 1:
            assert big_values[path] == swap(value, slot.size, slot.signed), f"{path} is not byte swapped"

    # The bit field positions are the same, but the bits are assigned from the other end of the storage unit, so
    # field values differ even though the layout is identical.
    assert little_values["bits.present"] == 1
    assert big_values["bits.present"] == 0
    assert little_values["bits.tag"] != big_values["bits.tag"]
    assert little_values["version"] != big_values["version"]
    assert little_values["bits.parity"] == big_values["bits.parity"] == 0x5A

    # Passing the endianness per call doesn't change the instance, and reads the other endianness' values. The unused
    # bit of the ``uint8`` unit moves to the other end of the unit, so re-encoding zeroes it, but every field value
    # roundtrips.
    record = load("<", compiled=compiled).EvidenceRecord
    per_call = record(data, endian=">")
    assert snapshot(per_call) == big_values
    assert snapshot(record(per_call.dumps(endian=">"), endian=">")) == big_values
    assert record.__cs__.endian == "<"


# Value and byte level equivalence


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_interpreted_and_compiled_agree(endian: str, compiled: bool) -> None:
    """Test that both reader implementations decode the golden record the same and write back the same bytes."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    expected = expected_values(endian)

    assert verify_compiled(record, compiled)

    obj = record(data)
    assert snapshot(obj) == expected
    assert len(obj) == RECORD_SIZE
    assert obj.dumps() == data

    # Both implementations read the same bytes into the same fields, so they can be mixed safely.
    other = load_record(endian, compiled=not compiled)
    assert snapshot(other(data)) == expected
    assert other(data).dumps() == data


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_byte_exact_roundtrip_of_generated_records(endian: str, compiled: bool) -> None:
    """Test that records built from field values re-encode byte exactly to their canonical form."""
    cs = load(endian, compiled=compiled)
    record = cs.EvidenceRecord
    union = cs.EvidenceValue
    bits = cs.EvidenceBits
    data = GOLDEN_RECORDS[endian]
    layout = derive_layout(record)

    # All values that are encoded in a bit field, so that no bit is left at its canonical value by accident.
    for version in (0, 6, 7):
        for tag in (0, 700, 1023):
            for window in (0, 0x12345, 0xFFFFF):
                obj = record(
                    magic=[0x45, 0x56],
                    version=version,
                    flags=cs.EvidenceFlags.FILE | cs.EvidenceFlags.CARVED,
                    pad=0xA5,
                    signed_byte=-128,
                    delta=-2,
                    kind=cs.EvidenceKind.REDACTED,
                    size=0x11223344,
                    offset=-1000,
                    value=union(raw=0x89ABCDEF),
                    bits=bits(
                        present=1,
                        compressed=0,
                        encrypted=2,
                        reserved=0,
                        tag=tag,
                        kind=13,
                        parity=0x5A,
                        window=window,
                        span=0x678,
                    ),
                    trailer=-3,
                )
                encoded = obj.dumps()

                # The bytes of the fields that vary are exactly the bytes of their own storage units, and nothing
                # else moves with them. This pins down that ``tag`` doesn't grow into ``kind``'s neighbours and that
                # ``window`` doesn't grow into ``span``'s neighbours.
                expected = {
                    layout["version"].storage,
                    layout["bits.tag"].storage,
                    layout["bits.tag"].storage + layout["bits.tag"].size - 1,
                }
                expected.update(
                    range(layout["bits.window"].storage, layout["bits.window"].storage + layout["bits.window"].size)
                )
                assert mutation_affected(data, encoded) <= expected

                reread = record(encoded)
                assert reread.version == version
                assert reread.bits.tag == tag
                assert reread.bits.window == window
                # The fields that share a unit with a varying field keep their values.
                assert reread.bits.kind == 13
                assert reread.bits.parity == 0x5A
                assert reread.bits.span == 0x678
                # The record is stable under re-encoding: byte for byte, and value for value.
                assert reread.dumps() == encoded
                assert snapshot(reread) == snapshot(obj)

    # The union writes its ``raw`` member, so all four bytes of the union are re-encoded.
    obj = record(data)
    obj.value = union(raw=0x11223344)
    encoded = obj.dumps()
    assert encoded[UNION_OFFSET : UNION_OFFSET + UNION_SIZE] == bytes.fromhex(
        "44332211" if endian == "<" else "11223344"
    )
    assert snapshot(record(encoded))["value.raw"] == 0x11223344
    assert layout["value"].storage == UNION_OFFSET


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_every_bit_maps_to_its_fields(endian: str, compiled: bool) -> None:
    """Test that every bit in the record belongs to exactly the fields whose storage contains it.

    This is the strongest check that a field can't move: flipping a bit may only change the field that owns it. For
    union members all overlapping views change together, since they share storage by definition. Unused bits of a
    bit field storage unit change nothing.
    """
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    layout = derive_layout(record)
    owners = leaf_owners(layout, endian)

    reference = snapshot(record(data))

    for byte in range(len(data)):
        for bit in range(8):
            changed = {
                path for path, value in snapshot(record(flip_byte(data, byte, bit))).items() if value != reference[path]
            }

            if byte in range(UNION_OFFSET, UNION_OFFSET + UNION_SIZE):
                # Every union member overlaps the same storage, so all views are re-derived from the changed bytes.
                union_leaves = {path for path in leaf_paths(layout) if path.startswith("value.")}
                assert changed, f"byte {byte} bit {bit} changed nothing"
                assert changed <= union_leaves
                continue

            if (byte, bit) in owners:
                # A bit field bit belongs to exactly one field.
                owner = owners[(byte, bit)]
                assert changed, f"byte {byte} bit {bit} changed nothing"
                assert changed == {owner}, f"byte {byte} bit {bit} changed {changed}, expected {owner}"
                continue

            # A regular field bit belongs to every regular field covering that byte (only unions overlap).
            expected = regular_owners(layout, byte)
            if expected:
                assert changed == expected, f"byte {byte} bit {bit} changed {changed}, expected {expected}"
            else:
                # Neither a field bit nor a regular byte: padding, or a bit field byte with no bits in use.
                assert changed == set(), f"byte {byte} bit {bit} changed {changed}, expected nothing"


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_union_views_overlap_only_within_the_union(endian: str, compiled: bool) -> None:
    """Test that the union views overlap exactly as their offsets say, and that nothing outside the union moves."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    layout = derive_layout(record)
    reference = snapshot(record(data))

    union_leaves = {path for path in leaf_paths(layout) if path.startswith("value.")}

    for byte in range(UNION_OFFSET, UNION_OFFSET + UNION_SIZE):
        for bit in range(8):
            changed = {
                path for path, value in snapshot(record(flip_byte(data, byte, bit))).items() if value != reference[path]
            }
            assert changed <= union_leaves
            assert changed


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_mutation_writes_only_owning_bytes(endian: str, compiled: bool) -> None:
    """Test that assigning a field changes only the bytes (or bits) that belong to that field."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    cs = record.__cs__
    layout = derive_layout(record)

    replacements: dict[str, object] = {
        "magic": [0x56, 0x45],
        "pad": 0x00,
        "signed_byte": 0x7F,
        "delta": 0x1234,
        "kind": cs.EvidenceKind.CONTENT,
        "size": 0xDEADBEEF,
        "offset": 0x01020304,
        "trailer": 0x0102,
    }

    for name, value in replacements.items():
        obj = record(data)
        setattr(obj, name, value)
        encoded = obj.dumps()
        changed = mutation_affected(data, encoded)

        slot = layout[name]
        own = set(range(slot.storage, slot.storage + slot.size))
        assert changed, f"{name} changed nothing"
        assert changed <= own, f"{name} changed {changed}, owns {own}"

        # Re-reading the result gives the new value and leaves every other field alone.
        reread = record(encoded)
        assert getattr(reread, name) == value
        assert {path: item for path, item in snapshot(reread).items() if path != name} == {
            path: item for path, item in snapshot(obj).items() if path != name
        }

    # Bit fields only change bits inside their storage unit.
    for name, value in (("tag", 0), ("tag", 1023), ("kind", 0), ("present", 0), ("parity", 0), ("window", 0)):
        obj = record(data)
        setattr(obj.bits, name, value)
        encoded = obj.dumps()
        changed = mutation_affected(data, encoded)

        slot = layout[f"bits.{name}"]
        own = set(range(slot.storage, slot.storage + slot.size))
        assert changed, f"bits.{name} changed nothing"
        assert changed <= own, f"bits.{name} changed {changed}, owns {own}"
        assert getattr(record(encoded).bits, name) == value


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_transcoding_preserves_values(endian: str, compiled: bool) -> None:
    """Test that re-encoding a record for the other endianness keeps every value that is read at its own storage."""
    other = ">" if endian == "<" else "<"
    record = load_record(endian, compiled=compiled)
    obj = record(GOLDEN_RECORDS[endian])

    transcoded = obj.dumps(endian=other)
    assert transcoded == GOLDEN_RECORDS[other]

    reread = load_record(other, compiled=compiled)(transcoded)
    actual = snapshot(reread)
    reference = snapshot(obj)

    for path in ENCODING_INDEPENDENT:
        assert actual[path] == reference[path], f"{path} changed from {reference[path]} to {actual[path]}"

    # The union itself is written by its ``raw`` member, so its storage is re-encoded too.
    assert reread.value.dumps() == transcoded[UNION_OFFSET : UNION_OFFSET + UNION_SIZE]
    assert reread.value.raw == obj.value.raw


# Truncated input and framing


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_every_truncation_raises_eof(endian: str, compiled: bool) -> None:
    """Test that a record that is one byte short is never interpreted, never partially returned."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)

    for size in range(len(data)):
        with pytest.raises(EOFError, match=r"^Read \d+ bytes, but expected \d+$"):
            record(data[:size])

    # A full record still parses after a failed attempt, and has the same flag meaning as before.
    obj = record(data)
    flags_type = record.__cs__.EvidenceFlags
    assert obj.dumps() == data
    assert int(obj.flags) == EXPECTED_VALUES["flags"]
    assert list(obj.flags) == [flags_type.FILE, flags_type.SYMLINK, flags_type.CARVED]


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_nested_types_are_truncation_safe(endian: str, compiled: bool) -> None:
    """Test that the nested types of the sample have the same truncation behaviour when read on their own."""
    data = GOLDEN_RECORDS[endian]
    cs = load(endian, compiled=compiled)

    bits_data = data[BITS_OFFSET : BITS_OFFSET + BITS_SIZE]
    union_data = data[UNION_OFFSET : UNION_OFFSET + UNION_SIZE]

    for size in range(len(bits_data)):
        with pytest.raises(EOFError):
            cs.EvidenceBits(bits_data[:size])
    for size in range(len(union_data)):
        with pytest.raises(EOFError):
            cs.EvidenceValue(union_data[:size])


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_record_framing(endian: str, compiled: bool) -> None:
    """Test that records are framed by their size, so a following record can't be mistaken for field bytes."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)

    # Two records in one stream are read at exactly their own offsets.
    stream = io.BytesIO(data + data)
    first = record(stream)
    assert stream.tell() == RECORD_SIZE
    second = record(stream)
    assert stream.tell() == 2 * RECORD_SIZE
    assert first.dumps() == second.dumps() == data

    # Reading from an offset keeps the offsets relative to the start of the record.
    stream = io.BytesIO(b"\xde\xad" + data)
    stream.read(2)
    assert snapshot(record(stream)) == expected_values(endian)
    assert stream.tell() == 2 + RECORD_SIZE

    # An array of records consumes exactly its own bytes.
    assert [obj.dumps() for obj in record[2](data + data)] == [data, data]


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_misframed_log_is_rejected(endian: str, compiled: bool) -> None:
    """Test that a log whose first record is one byte short is rejected instead of borrowing bytes from the next."""
    data = GOLDEN_RECORDS[endian]
    cs = load(endian, compiled=compiled, definition=EVIDENCE_LOG_CDEF)
    byteorder = "little" if endian == "<" else "big"

    log = (2).to_bytes(2, byteorder) + data + data
    parsed = cs.EvidenceLog(log)
    assert parsed.count == 2
    assert [obj.dumps() for obj in parsed.records] == [data, data]
    assert parsed.dumps() == log

    # The declared count promises two records, but the first one is short: the second record read runs out of bytes
    # and the whole log is rejected, rather than returning a record stitched together from both.
    misframed = (2).to_bytes(2, byteorder) + data[:-1] + data
    with pytest.raises(EOFError):
        cs.EvidenceLog(misframed)

    # A log that promises more records than are present is rejected too.
    with pytest.raises(EOFError):
        cs.EvidenceLog((3).to_bytes(2, byteorder) + data + data)

    # The valid prefix still parses on its own.
    prefix = (1).to_bytes(2, byteorder) + data
    assert len(cs.EvidenceLog(prefix).records) == 1


# Streams


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_tell_only_non_seekable_stream(endian: str, compiled: bool) -> None:
    """Test that a non-seekable stream that supports ``tell`` reads the same values as a file object."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    expected = snapshot(record(data))

    obj = record(TellOnlyStream(data))
    assert snapshot(obj) == expected
    assert obj.dumps() == data

    # Two records are read sequentially, and truncation fails where the bytes run out.
    stream = TellOnlyStream(data + data)
    assert record(stream).dumps() == data
    assert record(stream).dumps() == data
    assert stream.tell() == 2 * RECORD_SIZE

    for size in range(len(data)):
        stream = TellOnlyStream(data[:size])
        with pytest.raises(EOFError):
            record(stream)
        assert stream.tell() <= size


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_stream_without_tell_never_yields_wrong_values(endian: str, compiled: bool) -> None:
    """Test that a stream without ``tell`` either reads correctly or fails, but never returns a wrong record."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    expected = snapshot(record(data))

    for factory in (BareStream, NonSeekableStream):
        try:
            obj = record(factory(data))
        except (AttributeError, io.UnsupportedOperation):
            # The interpreted reader needs ``tell`` to honour offsets, so it refuses the stream instead of guessing.
            continue

        assert snapshot(obj) == expected
        assert obj.dumps() == data

        # If the stream is accepted, truncation must still be reported as an EOF, not as a wrong record.
        for size in range(len(data)):
            with pytest.raises(EOFError):
                record(factory(data[:size]))


def test_interpreted_reader_requires_tell() -> None:
    """Test that the interpreted reader refuses a stream without ``tell`` instead of misplacing fields.

    This documents an actual difference between the two readers: the compiled reader inlines the offsets of the
    sample, while the interpreted reader has to ask the stream for its position. Refusing the stream is the safe
    outcome; silently reading from the wrong offset would produce evidence fields from unrelated bytes.
    """
    data = GOLDEN_RECORDS["<"]

    interpreted = load_record("<", compiled=False)
    compiled = load_record("<", compiled=True)

    with pytest.raises(AttributeError):
        interpreted(BareStream(data))
    with pytest.raises(AttributeError):
        interpreted(NonSeekableStream(data))

    assert snapshot(compiled(BareStream(data))) == expected_values("<")
    assert snapshot(compiled(NonSeekableStream(data))) == expected_values("<")

    # Streams that can tell their position are accepted by both, and agree.
    assert snapshot(interpreted(TellOnlyStream(data))) == snapshot(compiled(TellOnlyStream(data)))


# Sign extension and bit width


def test_sign_extension_only_applies_to_regular_fields() -> None:
    """Test that signed regular fields sign extend, while bit fields are always returned unsigned."""
    cdef = """
    struct SignSample {
        int8   small;
        int16  medium;
        int8   bits_a : 4;
        int8   bits_b : 4;
        int16  wide_a : 4;
        int16  wide_b : 12;
    };
    """
    data = b"\x80\x00\x80\xff\xff\xff\xff"

    interpreted = load(definition=cdef).SignSample
    compiled = load(compiled=True, definition=cdef).SignSample
    assert not interpreted.__compiled__
    assert compiled.__compiled__

    for sample in (interpreted(data), compiled(data)):
        assert sample.small == -128
        assert sample.medium == -32768
        # Bit fields are not sign extended, even when their storage type is signed.
        assert sample.bits_a == 0xF
        assert sample.bits_b == 0xF
        assert sample.wide_a == 0xF
        assert sample.wide_b == 0xFFF

    assert snapshot(interpreted(data)) == snapshot(compiled(data))


def test_signed_bit_field_write_fails_loudly() -> None:
    """Test that writing unsigned bit field values through signed storage fails instead of writing wrong bytes.

    Reading a bit field is unsigned, but the storage unit is written with the signed type of the field. A unit whose
    top bit is set therefore can't be written back, and that must be an error, not a silently truncated value.
    """
    cdef = """
    struct SignedBits {
        int8  a : 4;
        int8  b : 4;
        uint8 tail;
    };
    """

    for compiled in (True, False):
        cs = cstruct()
        cs.load(cdef, compiled=compiled)
        assert verify_compiled(cs.SignedBits, compiled)

        ok = cs.SignedBits(b"\x0f\x01")
        assert (ok.a, ok.b, ok.tail) == (0xF, 0x0, 0x1)
        assert ok.dumps() == b"\x0f\x01"

        high = cs.SignedBits(b"\x8f\x01")
        assert (high.a, high.b) == (0xF, 0x8)
        with pytest.raises(struct.error):
            high.dumps()

        # The failed write leaves the object untouched, so the values are still available.
        assert (high.a, high.b) == (0xF, 0x8)
        assert cs.SignedBits(b"\x8f\x01").tail == 1

        # A negative value fits the signed storage and is written as its two's complement.
        negative = cs.SignedBits(b"\x0f\x01")
        negative.a = -1
        assert negative.dumps() == b"\xff\x01"
        # Reading is unsigned, so the same bits read back as 15.
        assert cs.SignedBits(b"\xff\x01").a == 0xF


def test_straddled_bit_fields_are_rejected() -> None:
    """Test that a definition that would straddle a storage unit is rejected, for both reader implementations."""
    cdef = """
    struct Straddled {
        uint16 a : 12;
        uint16 b : 12;
    };
    """

    for compiled in (True, False):
        with pytest.raises(ValueError, match="Straddled bit fields are unsupported"):
            cstruct().load(cdef, compiled=compiled)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_bit_field_boundary_padding_is_not_a_field(endian: str, compiled: bool) -> None:
    """Test that the unused tail bits of a bit field storage unit belong to no field.

    The ``uint8`` unit of the sample uses 7 of its 8 bits and the ``uint16`` ``tag``/``kind`` unit uses 14 of its 16,
    exactly like a compiler would pad a bit field block. Those bits must not be attributed to the next field, and
    writing the record must not preserve arbitrary values in them.
    """
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    layout = derive_layout(record)
    owners = leaf_owners(layout, endian)
    reference = snapshot(record(data))

    units = [layout["bits.present"], layout["bits.tag"], layout["bits.window"]]
    unused = []
    for unit in units:
        unused.extend(
            (unit.storage + index, bit)
            for index in range(unit.size)
            for bit in range(8)
            if (unit.storage + index, bit) not in owners
        )

    # One unused bit in the uint8 unit, two in the uint16 unit, and none in the full uint32 unit.
    unused_units = dict.fromkeys(units, 0)
    for byte, _ in unused:
        for unit in units:
            if unit.storage <= byte < unit.storage + unit.size:
                unused_units[unit] += 1
    assert list(unused_units.values()) == [1, 2, 0]
    assert len(unused) == 3

    for byte, bit in unused:
        obj = record(flip_byte(data, byte, bit))
        assert snapshot(obj) == reference
        # The neighbouring fields keep their meaning, and no bit field grows or shrinks.
        assert obj.bits.tag == reference["bits.tag"]
        assert obj.bits.kind == reference["bits.kind"]
        assert obj.bits.parity == reference["bits.parity"]
        # Padding is not preserved by the writer: the record re-encodes to the canonical bytes.
        assert obj.dumps() == data


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_bit_field_units_are_split_by_base_type(compiled: bool) -> None:
    """Test that a bit field of another base type starts a new storage unit, even if the previous unit has room.

    Sharing a unit is only safe when the base type is the same, since the unit is read and written with that type.
    """
    cdef = """
    struct SharedUnits {
        uint8 a : 4;
        uint8 b : 2;
        uint8 c : 2;
    };

    struct SplitUnits {
        uint8 a : 4;
        uint8 b : 2;
        char  c : 2;
    };

    struct SplitWidths {
        uint16 a : 4;
        uint32 b : 4;
        uint16 c : 4;
    };
    """
    cs = cstruct()
    cs.load(cdef, compiled=compiled)

    shared = derive_layout(cs.SharedUnits)
    assert {shared[name].storage for name in ("a", "b", "c")} == {0}
    assert len(cs.SharedUnits) == 1

    # A ``char`` bit field has the same size as a ``uint8`` one, but must not share its unit.
    split = derive_layout(cs.SplitUnits)
    assert split["a"].storage == split["b"].storage == 0
    assert split["c"].storage == 1
    assert len(cs.SplitUnits) == 2

    # Going back to an earlier base type starts yet another unit, it doesn't reuse a previous one.
    widths = derive_layout(cs.SplitWidths)
    assert widths["a"].storage == 0
    assert widths["b"].storage == 2
    assert widths["c"].storage == 6
    assert len(cs.SplitWidths) == 8

    # The sample itself demonstrates the ``uint8`` to ``uint16`` boundary: seven of the eight bits are used, and the
    # ``tag`` field still starts its own ``uint16`` unit on the next byte.
    sample = derive_layout(load().EvidenceRecord)
    assert sample["bits.present"].storage == BITS_OFFSET
    assert sample["bits.reserved"].storage == BITS_OFFSET
    assert sample["bits.present"].position + sample["bits.present"].bits == 1
    assert sample["bits.reserved"].bits == 3
    assert sample["bits.tag"].storage == BITS_OFFSET + 1


# Union behaviour


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_union_selection_and_views(endian: str, compiled: bool) -> None:
    """Test that the union size, its write member and every overlapping view are stable."""
    cs = load(endian, compiled=compiled)
    union = cs.EvidenceValue
    data = GOLDEN_RECORDS[endian]
    storage = data[UNION_OFFSET : UNION_OFFSET + UNION_SIZE]
    expected = expected_values(endian)

    assert len(union) == UNION_SIZE
    assert union.__alignment__ == 4
    # The write member is the largest named field, not an anonymous structure.
    assert union._write_field().name == "raw"
    assert all(field.offset is None for field in union.__fields__)

    obj = union(storage)
    assert obj.raw == expected["value.raw"]
    assert obj.signed_raw == expected["value.signed_raw"]
    assert obj.halves.low == expected["value.halves.low"]
    assert obj.halves.high == expected["value.halves.high"]
    assert list(obj.octets) == expected["value.octets"]
    assert obj.nibbles.low == expected["value.nibbles.low"]
    assert obj.nibbles.high == expected["value.nibbles.high"]
    assert obj.dumps() == storage

    # Changing a field through a nested view rebuilds every other view from the new bytes.
    expected_dumps = {"<": bytes.fromhex("0f00ab89"), ">": bytes.fromhex("8000cdef")}
    obj.nibbles.high = 0
    assert obj.nibbles.high == 0
    assert obj.nibbles.low == expected["value.nibbles.low"]
    assert obj.dumps() == expected_dumps[endian]

    obj = union(storage)
    obj.signed_raw = -1
    assert obj.raw == 0xFFFFFFFF
    assert list(obj.octets) == [0xFF] * 4
    assert obj.halves.low == obj.halves.high == 0xFFFF
    assert obj.dumps() == b"\xff\xff\xff\xff"

    # A union read from a non-seekable stream still uses the first member for writing.
    assert union(NonSeekableStream(storage)).dumps() == storage


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_union_mutation_through_record(endian: str, compiled: bool) -> None:
    """Test that changing a union view inside a record writes the new bytes at the union offset."""
    data = GOLDEN_RECORDS[endian]
    record = load_record(endian, compiled=compiled)
    slot = derive_layout(record)["value"]

    obj = record(data)
    obj.value.halves.low = 0xBEEF
    encoded = obj.dumps()

    changed = mutation_affected(data, encoded)
    own = set(range(slot.storage, slot.storage + slot.size))
    assert changed, "union mutation changed nothing"
    assert changed <= own
    assert record(encoded).value.halves.low == 0xBEEF
    assert record(encoded).value.raw == int.from_bytes(
        encoded[slot.storage : slot.storage + slot.size], "little" if endian == "<" else "big"
    )


# Flags


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_flag_bits_do_not_move(endian: str, compiled: bool) -> None:
    """Test that the version and flag bit fields keep their positions and can't change each other."""
    cs = load(endian, compiled=compiled)
    record = cs.EvidenceRecord
    flags_type = cs.EvidenceFlags
    data = GOLDEN_RECORDS[endian]

    assert [member.name for member in flags_type] == ["FILE", "DIRECTORY", "SYMLINK", "DELETED", "CARVED"]
    assert [member.value for member in flags_type] == [1, 2, 4, 8, 0x10]
    assert flags_type.FILE | flags_type.SYMLINK | flags_type.CARVED == 0x15

    obj = record(data)
    assert isinstance(obj.flags, flags_type)
    assert obj.flags == 0x15
    assert list(obj.flags) == [flags_type.FILE, flags_type.SYMLINK, flags_type.CARVED]

    # Every version value is independent from the flags.
    for version in range(8):
        obj = record(data)
        obj.version = version
        reread = record(obj.dumps())
        assert reread.version == version
        assert int(reread.flags) == 0x15
        assert list(reread.flags) == list(obj.flags)

    # Every combination of the flag bits is independent from the version, including values without a defined member.
    for raw in (0x00, 0x01, 0x10, 0x15, 0x1F):
        obj = record(data)
        obj.flags = flags_type(raw)
        encoded = obj.dumps()
        reread = record(encoded)
        assert int(reread.flags) == raw
        assert reread.version == 6
        assert [member.value for member in reread.flags] == [member.value for member in flags_type(raw)]


# Alignment and padding


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_aligned_layout_and_padding(endian: str, compiled: bool) -> None:
    """Test that aligned mode keeps the documented field positions and treats padding as padding."""
    cs = load(endian, compiled=compiled, align=True)
    record = cs.EvidenceRecord
    data = ALIGNED_RECORDS[endian]

    assert len(record) == ALIGNED_RECORD_SIZE
    assert record.__alignment__ == 4

    layout = derive_layout(record)
    assert layout["size"].storage == 12
    assert layout["value"].storage == 20
    assert layout["bits"].storage == 24
    assert layout["bits"].size == 12
    assert layout["bits.tag"].storage == 26
    assert layout["bits.parity"].storage == 28
    assert layout["bits.window"].storage == 32
    assert layout["trailer"].storage == 36
    assert len(cs.EvidenceBits) == 12

    obj = record(data)
    assert len(obj.dumps()) == ALIGNED_RECORD_SIZE
    encoded = obj.dumps()
    assert encoded == data
    assert snapshot(obj) == expected_values(endian)

    # Padding is every byte no leaf field covers, and it is written as zeroes.
    uncovered = set(range(ALIGNED_RECORD_SIZE)) - covered_bytes(layout)
    assert uncovered == {10, 11, 25, 29, 30, 31, 38, 39}
    assert all(encoded[index] == 0 for index in uncovered)

    # Arbitrary bytes in the padding don't change any value, and are dropped again when re-encoding.
    padded = bytearray(encoded)
    for index in uncovered:
        padded[index] = 0xFF
    reread = record(bytes(padded))
    assert snapshot(reread) == snapshot(obj)
    assert reread.dumps() == encoded

    # Every truncation that cuts into a field fails, including truncations inside a padding gap.
    last_field = max(slot.storage + slot.size for slot in layout.values())
    for size in range(last_field):
        with pytest.raises(EOFError):
            record(encoded[:size])

    # Only the trailing alignment padding isn't read, so a record missing nothing but that padding still parses and
    # re-encodes to the full canonical record. This is padding, not evidence, so no field is invented from it.
    for size in range(last_field, ALIGNED_RECORD_SIZE):
        obj = record(encoded[:size])
        assert obj.dumps() == encoded


# Error attribution


def _flat_flags_definition() -> str:
    """A definition whose ``flags`` field doesn't share the ``version`` storage unit."""
    return EVIDENCE_CDEF.replace("    EvidenceFlags flags : 5;", "    uint16 flags;")


def _narrow_tag_definition() -> str:
    """A definition whose ``kind`` bit field was accidentally shifted by a narrower ``tag``."""
    return EVIDENCE_CDEF.replace("    uint16 tag : 10;", "    uint16 tag : 8;")


def _unsigned_delta_definition() -> str:
    """A definition whose ``delta`` field lost its signedness without moving or changing size."""
    return EVIDENCE_CDEF.replace("    int16  delta;", "    uint16 delta;")


def test_diagnose_distinguishes_error_classes() -> None:
    """Test that the same sample and the same bytes produce a different diagnosis for each kind of error."""
    for endian in ("<", ">"):
        data = GOLDEN_RECORDS[endian]
        reference = load(endian).EvidenceRecord
        other_endian = ">" if endian == "<" else "<"

        # The correct definition has no problems.
        assert diagnose(reference, data, endian=endian) == set()

        # A field that moved changes the layout, and the record can't even be framed.
        flat = load(endian, definition=_flat_flags_definition()).EvidenceRecord
        assert diagnose(flat, data, endian=endian) == {LAYOUT_ERROR}

        # The same offsets, but a bit field covering different bits, is a bit width error.
        narrow = load(endian, definition=_narrow_tag_definition()).EvidenceRecord
        assert diagnose(narrow, data, endian=endian) == {BIT_WIDTH_ERROR}

        # The same layout read with the wrong endianness is an endianness error, not a layout or bit width error.
        assert diagnose(reference, data, endian=other_endian) == {ENDIANNESS_ERROR}

        # A field of the same position and size that lost its signedness is a value error.
        unsigned = load(endian, definition=_unsigned_delta_definition()).EvidenceRecord
        assert diagnose(unsigned, data, endian=endian) == {VALUE_ERROR}


def test_diagnose_reports_layout_for_unknown_field() -> None:
    """Test that a definition with an extra field is reported as a layout error."""
    extra = load(
        definition=EVIDENCE_CDEF.replace("    int16  trailer;", "    int16  trailer;\n    uint32 checksum;")
    ).EvidenceRecord

    problems = diagnose(extra, LITTLE_ENDIAN_RECORD, endian="<")
    assert problems == {LAYOUT_ERROR}
