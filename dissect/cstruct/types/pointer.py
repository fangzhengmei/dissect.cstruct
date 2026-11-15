from __future__ import annotations

from typing import TYPE_CHECKING, Any, BinaryIO, Generic, TypeVar

from dissect.cstruct.exception import (
    DereferenceBudgetExceeded,
    NegativePointerAddress,
    NullPointerDereference,
    PointerCycleError,
    PointerDereferenceError,
    PointerOutOfWindow,
)
from dissect.cstruct.types.base import BaseType, normalize_endianness
from dissect.cstruct.types.char import Char
from dissect.cstruct.types.state import (
    _UNSET,
    DereferenceBudget,
    DereferenceResult,
    DereferenceState,
    PointerWindow,
    WindowedStream,
    as_budget,
    as_window,
    check_address,
    enter_chain,
    enter_resolution,
    exit_chain,
    exit_resolution,
    in_chain,
    inherit_state,
)
from dissect.cstruct.types.void import Void

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from dissect.cstruct.cstruct import AllowedEndianness, Endianness, ReadWindow

T = TypeVar("T", bound=BaseType)

# Sentinel that makes ``Pointer.__new__`` inherit the resolution of the parse that created it, if any
_INHERIT = object()


class Pointer(int, BaseType, Generic[T]):
    """Pointer to some other type.

    Dereferencing a pointer can be restricted by a read window, which limits the stream offsets it may visit, and a
    dereference budget, which limits how many dereferences a single resolution may trace. Both are configured on the
    :class:`~dissect.cstruct.cstruct` instance with ``read_window`` and ``dereference_budget``, or passed to
    :meth:`dereference`, :meth:`try_dereference` and the ``read``/``reads`` entry points directly.

    Every pointer that is created while a parse or a dereference is running shares the limits and the resolved
    values of that resolution. This makes a lazily walked pointer chain and a compiled parse follow the same
    constraints, and makes multiple pointers to the same object resolve to the same value. A rejected dereference
    raises a :class:`~dissect.cstruct.exception.PointerDereferenceError` subclass and restores the stream position.
    """

    __type__: type[T]
    __stream: BinaryIO | None
    __context: dict[str, Any] | None
    __endian: Endianness
    __state: DereferenceState | None
    __values: dict[Endianness, Any] | None

    def __new__(
        cls,
        value: int,
        stream: BinaryIO | None,
        *,
        context: dict[str, Any] | None = None,
        endian: Endianness,
        state: DereferenceState | None | object = _INHERIT,
    ) -> Self:
        obj = super().__new__(cls, value)
        obj.__stream = stream
        obj.__context = context
        obj.__endian = endian

        if state is not _INHERIT:
            obj.__state = state
        else:
            obj.__state = inherit_state(cls.__cs__)

        # Pointers outside of a resolution remember the value they point to, like they always did
        obj.__values = None
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
                self.__stream,
                context=self.__context,
                endian=self.__endian,
                state=self.__state,
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
            state=None,
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

    def _resolution(
        self,
        window: ReadWindow | None | object = _UNSET,
        budget: DereferenceBudget | int | None | object = _UNSET,
    ) -> DereferenceState | None:
        """Return the resolution this dereference belongs to.

        A pointer that was created during a parse or another dereference already belongs to a resolution, and reuses
        it, so all pointers of that resolution share one window, one budget and the values that were resolved. Limits
        given to this dereference start a new resolution for the chain that is walked from here, so the budget covers
        that whole chain and the window applies to every value that is read for it.
        """
        state = self.__state
        cs = type(self).__cs__

        if window is _UNSET and budget is _UNSET:
            if state is not None:
                return state

            configured_window = as_window(cs.read_window)
            configured_budget = as_budget(cs.dereference_budget)
        else:
            configured_window = as_window(cs.read_window) if window is _UNSET else as_window(window)
            configured_budget = as_budget(cs.dereference_budget) if budget is _UNSET else as_budget(budget)

        if configured_window is None and configured_budget is None:
            return None

        return DereferenceState(cs, configured_window, configured_budget)

    def dereference(
        self,
        *,
        endian: AllowedEndianness | None = None,
        window: ReadWindow | None | object = _UNSET,
        budget: DereferenceBudget | int | None | object = _UNSET,
    ) -> T | None:
        """Dereference the pointer and read the value it points to.

        Args:
            endian: Optional endianness to use when reading the value.
                    If not provided, the endianness used when reading the pointer itself will be used.
            window: Optional read window for this dereference, instead of the cstruct instance's ``read_window``.
            budget: Optional dereference budget, instead of the cstruct instance's ``dereference_budget``.

        Returns:
            The dereferenced value, or ``None`` for a pointer to ``void``.

        Raises:
            NullPointerDereference: If the pointer is null or has no stream to read from.
            NegativePointerAddress: If the pointer has a negative address.
            PointerOutOfWindow: If the pointer or the value it reads is not fully inside the read window.
            DereferenceBudgetExceeded: If the resolution traced more dereferences than the budget allows.
            PointerCycleError: If resolving the pointer requires resolving the value it is reading again.
        """
        if self == 0 or self.__stream is None:
            raise NullPointerDereference(pointer=self, address=0)

        if issubclass(self.__type__, Void):
            return None

        endian = normalize_endianness(endian) if endian is not None else self.__endian
        address = int(self)
        state = self._resolution(window, budget)

        if address < 0:
            raise NegativePointerAddress(
                f"Pointer address is negative ({address})",
                pointer=self,
                address=address,
            )

        # The resolution already reflects the limits of this dereference, if there were any
        win = state.window if state is not None else None
        limit = state.budget if state is not None else None

        if win is not None:
            check_address(win, address, self.__type__.__size__ or 0, self)

        key = (self.__type__, address, endian)
        if state is not None:
            if (cached := state.resolved.get(key, _UNSET)) is not _UNSET:
                # Another pointer in this resolution already resolved the same object, share it
                return cached

            # Only a dereference that isn't resolved before counts against the budget, so pointers that share an
            # object don't exhaust it
            state.count += 1
            if limit is not None and not limit.allows(state.count):
                # A rejected dereference doesn't trace anything, so it doesn't use up budget
                state.count -= 1
                raise DereferenceBudgetExceeded(
                    f"Pointer dereference exceeded the budget of {limit.max_dereferences} dereferences "
                    f"for {self.__type__.__name__} at {address:#x}",
                    pointer=self,
                    address=address,
                    limit=limit.max_dereferences,
                )
        elif (values := self.__values) is not None and endian in values:
            # A pointer outside of a resolution remembers the value it points to
            return values[endian]

        cs = type(self).__cs__
        if in_chain(cs, self.__type__, address, endian):
            raise PointerCycleError(
                f"Pointer dereference of {self.__type__.__name__} at {address:#x} is cyclic",
                pointer=self,
                address=address,
            )

        position = self.__stream.tell()
        stream: BinaryIO = self.__stream if win is None else WindowedStream(self.__stream, win)
        token = enter_resolution(state) if state is not None else None
        chain_token = enter_chain(cs, self.__type__, address, endian)
        try:
            self.__stream.seek(address)

            if issubclass(self.__type__, Char):
                # This makes the assumption that a char pointer is a null-terminated string
                value = self.__type__._read_0(stream, context=self.__context, endian=endian)
            else:
                value = self.__type__._read(stream, context=self.__context, endian=endian)
        except EOFError as e:
            if win is not None and stream.clipped:
                # The value isn't fully inside the window, which the reader reports as a premature end
                raise PointerOutOfWindow(
                    f"Pointer value of {self.__type__.__name__} at {address:#x} is not fully inside the window "
                    f"[{win.offset:#x}, {win.end:#x})",
                    pointer=self,
                    address=address,
                ) from e

            raise
        finally:
            # Restore the stream position after reading the value, also when the dereference was rejected
            self.__stream.seek(position)
            exit_chain(chain_token)
            if token is not None:
                exit_resolution(token)

        if state is not None:
            state.resolved[key] = value
        else:
            if self.__values is None:
                self.__values = {}
            self.__values[endian] = value

        return value

    def try_dereference(
        self,
        *,
        endian: AllowedEndianness | None = None,
        window: ReadWindow | None | object = _UNSET,
        budget: DereferenceBudget | int | None | object = _UNSET,
    ) -> DereferenceResult:
        """Dereference the pointer like :meth:`dereference`, but return a distinguishable rejection instead of raising.

        The rejection never changes the stream position, and its ``reason`` tells a null pointer, a negative
        address, an out of window value, an exhausted budget and a cycle apart.
        """
        try:
            return DereferenceResult(self.dereference(endian=endian, window=window, budget=budget))
        except PointerDereferenceError as e:
            return DereferenceResult(None, e)


__all__ = [
    "DereferenceBudget",
    "DereferenceResult",
    "Pointer",
    "PointerWindow",
]
