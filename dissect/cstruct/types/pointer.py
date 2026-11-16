from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, TypeVar

from dissect.cstruct.exception import (
    NullPointerDereference,
    PointerBudgetExceeded,
    PointerCycleError,
    PointerWindowError,
)
from dissect.cstruct.types.base import BaseType, normalize_endianness
from dissect.cstruct.types.char import Char
from dissect.cstruct.types.void import Void

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from dissect.cstruct.cstruct import AllowedEndianness, Endianness

T = TypeVar("T", bound=BaseType)

# Context key under which the active :class:`_PointerResolver` is passed through the read context.
# Field names may not contain leading double underscores, so this key can't collide with one.
RESOLVER_CONTEXT_KEY = "__cstruct_pointer_resolver__"


class _BoundedStream:
    """Thin stream wrapper that hides everything past the given absolute end offset.

    Reads that would cross the end are truncated to it, so e.g. a null-terminated string can never
    read past the pointer window. All other operations are delegated to the wrapped stream.
    """

    def __init__(self, stream: BinaryIO, end: int):
        self._stream = stream
        self._end = end

    def read(self, size: int = -1) -> Any:
        stream = self._stream
        remaining = max(0, self._end - stream.tell())
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        return stream.read(size)

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def seekable(self) -> bool:
        return self._stream.seekable()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._stream, attr)


class _PointerResolver:
    """Tracks the constraints and the state of one pointer graph resolution.

    A resolver is created when parsing starts (if any limits are configured) and is passed to every
    :class:`Pointer` created during that parse through the read context. Pointers created outside of a
    parse get a fresh resolver from the cstruct instance's current limits.

    Args:
        window: Optional ``(start, end)`` byte range that pointer targets must fall within.
        budget: Optional maximum number of dereferences that may be performed.
    """

    def __init__(self, window: tuple[int, int] | None = None, budget: int | None = None):
        self.window = window
        self.budget = budget
        # Number of dereferences performed so far in this resolution
        self._dereferences = 0
        # Addresses currently being resolved by nested dereferences, for cycle detection
        self._active: set[int] = set()
        # Values already resolved, keyed by (target type, address, endianness) so pointers that share
        # a target yield the exact same object
        self._values: dict[tuple[type, int, Endianness], Any] = {}

    def check_target(self, target: int, size: int | None) -> None:
        """Validate that dereferencing ``target`` for a read of ``size`` bytes stays within the window."""
        if target < 0:
            raise PointerWindowError(f"Pointer target {target} is a negative offset")

        if self.window is not None:
            start, end = self.window
            if target < start:
                raise PointerWindowError(
                    f"Pointer target {target:#x} is before the read window [{start:#x}, {end:#x})"
                )
            if target >= end:
                raise PointerWindowError(
                    f"Pointer target {target:#x} is at or past the end of the read window [{start:#x}, {end:#x})"
                )
            if size is not None and target + size > end:
                raise PointerWindowError(
                    f"Pointer target {target:#x} ({size} bytes) extends past the read window [{start:#x}, {end:#x})"
                )

    def dereference(self, pointer: Pointer, endian: Endianness) -> Any:
        """Dereference ``pointer`` under this resolver's window, budget and cycle constraints."""
        target = int(pointer)

        if target == 0 or pointer._stream is None:
            raise NullPointerDereference

        target_type = pointer.__type__

        if issubclass(target_type, Void):
            # A void pointer has no value to read, but resolving it still counts as a dereference
            self.check_target(target, 0)
            self._consume_budget(target)
            return None

        size = target_type.__size__
        cache_key = (target_type, target, endian)
        cached = self._values.get(cache_key)
        if cached is not None:
            # Multiple pointers sharing a target always return the same object, without spending budget
            return cached

        self.check_target(target, size)

        if target in self._active:
            raise PointerCycleError(f"Pointer cycle detected at address {target:#x}")

        if self.budget is not None and self._dereferences >= self.budget:
            raise PointerBudgetExceeded(
                f"Pointer dereference budget of {self.budget} exhausted at address {target:#x}"
            )

        self._active.add(target)
        self._dereferences += 1
        try:
            value = pointer._read_target(endian)
        finally:
            self._active.discard(target)

        self._values[cache_key] = value
        return value

    def _consume_budget(self, target: int) -> None:
        if self.budget is not None and self._dereferences >= self.budget:
            raise PointerBudgetExceeded(
                f"Pointer dereference budget of {self.budget} exhausted at address {target:#x}"
            )
        self._dereferences += 1


def new_pointer_resolver(window: tuple[int, int] | None, budget: int | None) -> _PointerResolver | None:
    """Create a resolver from the given limits, or ``None`` if no limits are configured."""
    if window is None and budget is None:
        return None
    return _PointerResolver(window, budget)


class Pointer(int, BaseType, Generic[T]):
    """Pointer to some other type."""

    __type__: type[T]
    _stream: BinaryIO | None
    _context: dict[str, Any] | None
    _endian: Endianness
    _resolver: _PointerResolver | None

    def __new__(
        cls,
        value: int,
        stream: BinaryIO | None,
        *,
        context: dict[str, Any] | None = None,
        endian: Endianness,
        resolver: _PointerResolver | None = None,
    ) -> Self:
        obj = super().__new__(cls, value)
        obj._stream = stream
        obj._context = context
        obj._endian = endian
        if resolver is None and context is not None:
            resolver = context.get(RESOLVER_CONTEXT_KEY)
        obj._resolver = resolver
        obj.dereference = cache(obj.dereference)
        return obj

    def __repr__(self) -> str:
        return f"<{self.__type__.__name__}* @ {self:#x}>"

    def __str__(self) -> str:
        return str(self.dereference())

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.dereference(), attr)

    @staticmethod
    def __op(op: Callable[[int, int], int]) -> Callable[[Self, int], Self]:
        def method(self: Self, other: int) -> Self:
            return type.__call__(
                self.__class__,
                op(self, other),
                self._stream,
                context=self._context,
                endian=self._endian,
                resolver=self._resolver,
            )

        return method

    __add__ = __op(int.__add__)
    __sub__ = __op(int.__sub__)
    __mul__ = __op(int.__mul__)
    __floordiv__ = __op(int.__floordiv__)
    __mod__ = __op(int.__mod__)
    __pow__ = __op(int.__pow__)
    __lshift__ = __op(int.__lshift__)
    __rshift__ = __op(int.__rshift__)
    __and__ = __op(int.__and__)
    __xor__ = __op(int.__xor__)
    __or__ = __op(int.__or__)

    @classmethod
    def __default__(cls) -> Self:
        return cls.__new__(
            cls,
            cls.__cs__.pointer.__default__(),
            None,
            context=None,
            endian=cls.__cs__.endian,
        )

    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:
        return cls.__new__(
            cls,
            cls.__cs__.pointer._read(stream, context=context, endian=endian),
            stream,
            context=context,
            endian=endian,
        )

    @classmethod
    def _write(cls, stream: BinaryIO, data: int, *, endian: Endianness) -> int:
        return cls.__cs__.pointer._write(stream, data, endian=endian)

    def _deref_endian(self, endian: AllowedEndianness | None) -> Endianness:
        """Return the endianness to use when dereferencing."""
        return normalize_endianness(endian) if endian is not None else self._endian

    def _read_target(self, endian: Endianness) -> T:
        """Seek to the pointer target, read its value and restore the stream position."""
        stream = self._stream
        position = stream.tell()
        try:
            stream.seek(int(self))

            resolver = self._resolver
            if resolver is not None and resolver.window is not None:
                target_stream: BinaryIO = _BoundedStream(stream, resolver.window[1])
            else:
                target_stream = stream

            # Make sure nested structures and pointers reuse this resolver, so the budget, cycle and
            # shared-object state stay consistent across the whole pointer graph
            if resolver is not None and (self._context or {}).get(RESOLVER_CONTEXT_KEY) is not resolver:
                context = {**(self._context or {}), RESOLVER_CONTEXT_KEY: resolver}
            else:
                context = self._context

            if issubclass(self.__type__, Char):
                # This makes the assumption that a char pointer is a null-terminated string
                return self.__type__._read_0(target_stream, context=context, endian=endian)
            return self.__type__._read(target_stream, context=context, endian=endian)
        finally:
            # Always restore the stream position, including when a read is rejected
            stream.seek(position)

    def dereference(self, *, endian: AllowedEndianness | None = None) -> T | None:
        """Dereference the pointer and read the value it points to.

        When the cstruct instance (or parse) has pointer limits configured, the dereference is validated
        against them:

        * a target outside of the read window (including a negative offset) raises
          :class:`~dissect.cstruct.exception.PointerWindowError`,
        * exceeding the dereference budget raises
          :class:`~dissect.cstruct.exception.PointerBudgetExceeded`,
        * a nested chain that revisits an address it is already resolving raises
          :class:`~dissect.cstruct.exception.PointerCycleError`.

        In every rejection case the stream position is restored to where it was before the call. Pointers
        that share a target inside the window resolve to the same cached object.

        Args:
            endian: Optional endianness to use when reading the value.
                    If not provided, the endianness used when reading the pointer itself will be used.
        """
        if self == 0 or self._stream is None:
            raise NullPointerDereference

        endian = self._deref_endian(endian)
        resolver = self._resolver

        if resolver is None:
            # No limits configured: behave exactly as before
            return self._read_target(endian)

        return resolver.dereference(self, endian)
