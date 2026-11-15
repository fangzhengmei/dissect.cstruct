from __future__ import annotations

import io
from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

from dissect.cstruct.exception import NegativePointerAddress, PointerOutOfWindow

if TYPE_CHECKING:
    from contextvars import Token

    from dissect.cstruct.cstruct import cstruct
    from dissect.cstruct.exception import PointerDereferenceError

_UNSET = object()

# The ways a read window can be given: a ``PointerWindow``, a ``(start, end)`` pair, a ``(start, end)`` range, a
# ``start:end`` slice, or a mapping with an ``offset``/``size`` or ``start``/``end``
ReadWindow: TypeAlias = "PointerWindow | tuple[int, int] | list[int] | slice | range | Mapping[str, int]"

# The resolution that is currently running: the parse that is reading, or the dereference that is resolving a
# pointer target. Every pointer that is created in that context belongs to the same resolution.
_STATE: ContextVar[DereferenceState | None] = ContextVar("cstruct_dereference_state", default=None)
# The values that are currently being resolved, to detect a cyclic pointer chain instead of recursing forever
_CHAIN: ContextVar[frozenset[tuple[cstruct, type, int, str]]] = ContextVar(
    "cstruct_dereference_chain", default=frozenset()
)


class PointerWindow:
    """An absolute byte range of a stream that pointer dereferences are allowed to visit.

    The window is expressed in absolute stream offsets and is inclusive of ``offset`` and exclusive of
    ``offset + size``. Both the address a pointer points to and the value that is read for it must fall inside the
    window. Reading through a window behaves as if the window is the whole stream: data outside of it is not visible,
    and a value that extends past the end of the window is rejected.

    The window can be given as an offset and a size, or as a start and an end offset::

        PointerWindow(0x100, 0x20)  # offset 0x100, size 0x20
        PointerWindow(start=0x100, end=0x120)  # the same window
    """

    __slots__ = ("offset", "size")

    def __init__(
        self,
        offset: int | None = None,
        size: int | None = None,
        *,
        start: int | None = None,
        end: int | None = None,
    ):
        if start is not None:
            if offset is not None:
                raise TypeError("Cannot combine 'start' with a positional offset")
            offset = start

        if end is not None:
            if size is not None:
                raise TypeError("Cannot combine 'end' with a size")
            if offset is None:
                raise TypeError("An 'end' requires a start or offset")

            size = end - offset

        if offset is None:
            offset = 0
        if size is None:
            raise TypeError("A window requires a size or an end")

        if offset < 0:
            raise ValueError(f"Window offset must be non-negative, got {offset}")
        if size < 0:
            raise ValueError(f"Window size must be non-negative, got {size}")

        self.offset = offset
        self.size = size

    @classmethod
    def from_bounds(cls, start: int, end: int) -> PointerWindow:
        """Create a window from a start offset (inclusive) and an end offset (exclusive)."""
        return cls(start, end - start)

    @property
    def start(self) -> int:
        """The first offset that is inside the window."""
        return self.offset

    @property
    def end(self) -> int:
        """The first offset that is outside the window."""
        return self.offset + self.size

    def contains(self, offset: int, size: int = 0) -> bool:
        """Return whether the given absolute offset range is fully inside the window."""
        return offset >= self.offset and size >= 0 and offset + size <= self.end

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PointerWindow):
            return self.offset == other.offset and self.size == other.size
        return NotImplemented

    def __hash__(self) -> int:
        return hash((PointerWindow, self.offset, self.size))

    def __repr__(self) -> str:
        return f"PointerWindow(offset={self.offset:#x}, size={self.size:#x})"


class DereferenceBudget:
    """Limits the amount of pointer dereferences a single resolution is allowed to trace.

    A resolution is a parse, or the dereference of a pointer that wasn't created by a parse. Every pointer of a
    resolution shares its budget, so following a chain of pointers - including a cyclic one - stops after at most
    ``max_dereferences`` traced dereferences instead of running forever. Pointers that share the same object only
    trace it once, so a legitimately shared object doesn't exhaust the budget.

    Args:
        max_dereferences: The maximum amount of dereferences a resolution may trace.
    """

    __slots__ = ("max_dereferences",)

    def __init__(self, max_dereferences: int):
        if max_dereferences < 0:
            raise ValueError(f"Dereference budget must be non-negative, got {max_dereferences}")

        self.max_dereferences = max_dereferences

    @property
    def limit(self) -> int:
        """Alias of ``max_dereferences``."""
        return self.max_dereferences

    def allows(self, count: int) -> bool:
        """Return whether the given amount of traced dereferences is still within budget."""
        return count <= self.max_dereferences

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DereferenceBudget):
            return self.max_dereferences == other.max_dereferences
        return NotImplemented

    def __hash__(self) -> int:
        return hash((DereferenceBudget, self.max_dereferences))

    def __repr__(self) -> str:
        return f"DereferenceBudget(max_dereferences={self.max_dereferences})"


class DereferenceResult(NamedTuple):
    """The result of a non-raising pointer dereference.

    Exactly one of ``value`` and ``error`` is set. A rejected dereference carries a
    :class:`~dissect.cstruct.exception.PointerDereferenceError` subclass, so the kind of rejection can be told
    apart, and it never changed the stream position.
    """

    value: Any = None
    error: PointerDereferenceError | None = None

    @property
    def ok(self) -> bool:
        """Whether the dereference was accepted."""
        return self.error is None

    @property
    def reason(self) -> str | None:
        """The stable reason of a rejected dereference, or ``None`` if it was accepted."""
        return None if self.error is None else self.error.reason

    def __bool__(self) -> bool:
        return self.ok


def as_window(value: ReadWindow | None) -> PointerWindow | None:
    """Normalize the different ways a read window can be configured.

    A :class:`PointerWindow` is used as-is, a two element tuple or list and a ``range`` are interpreted as a start
    (inclusive) and end (exclusive) offset, a slice as ``start:end`` and a mapping by its ``offset``/``size`` or
    ``start``/``end`` keys.
    """
    if value is None or isinstance(value, PointerWindow):
        return value

    if isinstance(value, slice):
        if value.step is not None:
            raise TypeError(f"Invalid read window: {value!r}")
        start = 0 if value.start is None else value.start
        if value.stop is None:
            raise TypeError(f"Invalid read window: {value!r}")
        return PointerWindow.from_bounds(start, value.stop)

    if isinstance(value, range):
        return PointerWindow.from_bounds(value.start, value.stop)

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return PointerWindow.from_bounds(*value)

    if isinstance(value, Mapping):
        offset = value.get("offset", value.get("start"))
        size = value.get("size")
        end = value.get("end")

        if offset is None:
            raise TypeError(f"Invalid read window: {value!r}")

        return PointerWindow(offset, size, end=end)

    raise TypeError(f"Invalid read window: {value!r}")


def as_budget(value: DereferenceBudget | int | None) -> DereferenceBudget | None:
    """Normalize the different ways a dereference budget can be configured."""
    if value is None or isinstance(value, DereferenceBudget):
        return value

    if isinstance(value, int) and not isinstance(value, bool):
        return DereferenceBudget(value)

    raise TypeError(f"Invalid dereference budget: {value!r}")


class DereferenceState:
    """The shared state of a single pointer resolution.

    Holds the window and budget of the resolution, how many dereferences it traced and the values it already
    resolved. Sharing one state between all pointers of a resolution makes pointers to the same object resolve to
    the same value, and makes the budget cover a whole lazy walk instead of a single call.
    """

    __slots__ = ("budget", "count", "cs", "resolved", "window")

    def __init__(self, cs: cstruct, window: PointerWindow | None = None, budget: DereferenceBudget | None = None):
        self.cs = cs
        self.window = window
        self.budget = budget
        self.count = 0
        # Values that were resolved before, so pointers to the same object share the result
        self.resolved: dict[tuple[type, int, str], Any] = {}

    def __repr__(self) -> str:
        return f"DereferenceState(window={self.window!r}, budget={self.budget!r}, count={self.count})"


def enter_parse(
    cs: cstruct,
    window: ReadWindow | None | object = _UNSET,
    budget: DereferenceBudget | int | None | object = _UNSET,
) -> Token[DereferenceState | None] | None:
    """Start the resolution of a parse, so that all pointers it creates share one window and budget.

    Returns ``None`` when no limits are configured, in which case pointers keep their unrestricted behavior.
    """
    if window is _UNSET and budget is _UNSET and not cs._dereference_limits:
        return None

    configured_window = as_window(cs.read_window) if window is _UNSET else as_window(window)
    configured_budget = as_budget(cs.dereference_budget) if budget is _UNSET else as_budget(budget)

    if configured_window is None and configured_budget is None:
        return None

    return _STATE.set(DereferenceState(cs, configured_window, configured_budget))


def exit_parse(token: Token[DereferenceState | None] | None) -> None:
    """End the resolution of a parse."""
    if token is not None:
        _STATE.reset(token)


def enter_resolution(state: DereferenceState) -> Token[DereferenceState | None]:
    """Mark the given resolution as the one currently reading pointer targets."""
    return _STATE.set(state)


def exit_resolution(token: Token[DereferenceState | None]) -> None:
    """Mark the end of reading pointer targets for a resolution."""
    _STATE.reset(token)


def inherit_state(cs: cstruct) -> DereferenceState | None:
    """Return the resolution a newly created pointer belongs to, if any."""
    state = _STATE.get()
    return state if state is not None and state.cs is cs else None


def enter_chain(cs: cstruct, type_: type, address: int, endian: str) -> Token[frozenset]:
    """Register that a value is being resolved, so a cyclic chain is detected instead of recursing forever."""
    return _CHAIN.set(_CHAIN.get() | {(cs, type_, address, endian)})


def exit_chain(token: Token[frozenset]) -> None:
    """Unregister a value that finished resolving."""
    _CHAIN.reset(token)


def in_chain(cs: cstruct, type_: type, address: int, endian: str) -> bool:
    """Return whether a value is already being resolved, which means the chain is cyclic."""
    return (cs, type_, address, endian) in _CHAIN.get()


class WindowedStream:
    """A read-only view of a stream that is restricted to a :class:`PointerWindow`.

    Positions are reported and interpreted as absolute stream offsets, so values read through this view keep the
    same offsets as the underlying stream. Reads are capped at the end of the window, which makes data outside of
    the window invisible. ``clipped`` records that a read was cut short by the window, so a parse that then runs
    into an ``EOFError`` can tell that apart from a genuine end of the data.
    """

    __slots__ = ("_stream", "_window", "clipped")

    def __init__(self, stream: Any, window: PointerWindow):
        self._stream = stream
        self._window = window
        self.clipped = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        position = self._stream.tell()

        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = position + offset
        elif whence == io.SEEK_END:
            target = self._window.end + offset
        else:
            raise ValueError(f"Invalid whence ({whence})")

        if not self._window.contains(target):
            raise PointerOutOfWindow(
                f"Pointer dereference sought to {target:#x}, outside the window "
                f"[{self._window.offset:#x}, {self._window.end:#x})",
                address=target,
            )

        return self._stream.seek(offset, whence)

    def read(self, size: int | None = -1) -> bytes:
        position = self._stream.tell()
        if not self._window.contains(position):
            raise PointerOutOfWindow(
                f"Pointer dereference read from {position:#x}, outside the window "
                f"[{self._window.offset:#x}, {self._window.end:#x})",
                address=position,
            )

        remaining = self._window.end - position
        if size is None or size < 0:
            # A request for everything ends at the end of the window
            size = remaining
        elif size > remaining:
            # An explicit read that runs past the window is capped, which a reader then reports as a short read
            self.clipped = True
            size = remaining

        return self._stream.read(size)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._stream, attr)


def check_address(window: PointerWindow, address: int, size: int, pointer: object) -> None:
    """Reject an address range that isn't fully inside the given window."""
    if address < 0:
        raise NegativePointerAddress(
            f"Pointer address is negative ({address})",
            pointer=pointer,
            address=address,
        )

    if not window.contains(address, size):
        raise PointerOutOfWindow(
            f"Pointer value of {size} bytes at {address:#x} is outside the window "
            f"[{window.offset:#x}, {window.end:#x})",
            pointer=pointer,
            address=address,
        )
