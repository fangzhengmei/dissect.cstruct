from __future__ import annotations


class Error(Exception):
    pass


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class ResolveError(Error):
    pass


class PointerDereferenceError(Error):
    """Base class for rejected pointer dereferences.

    A rejected dereference never changes the position of the stream it would have read from, so a caller can tell
    a rejection apart from a regular parse error and continue parsing from a known position. The ``reason``
    attribute is a stable, machine readable identifier of the rejection.
    """

    reason: str = "rejected"

    def __init__(
        self,
        message: str | None = None,
        *,
        pointer: object | None = None,
        address: int | None = None,
        **kwargs: object,
    ):
        super().__init__(message or self.reason)
        self.pointer = pointer
        self.address = address
        for key, value in kwargs.items():
            setattr(self, key, value)


class NullPointerDereference(PointerDereferenceError):
    """A null pointer was dereferenced."""

    reason = "null_pointer"


class PointerOutOfWindow(PointerDereferenceError):
    """A pointer dereference targeted or read bytes outside of the configured evidence window."""

    reason = "out_of_window"


class NegativePointerAddress(PointerOutOfWindow):
    """A pointer dereference targeted a negative address."""

    reason = "negative_address"


class DereferenceBudgetExceeded(PointerDereferenceError):
    """A parse tried to trace more pointer dereferences than the configured budget allows."""

    reason = "budget_exceeded"


class PointerCycleError(PointerDereferenceError):
    """Resolving a pointer required resolving the same value again, so the chain is cyclic."""

    reason = "cycle"


class ArraySizeError(Error):
    pass


class ExpressionParserError(Error):
    pass


class UnknownIdentifierError(ExpressionParserError):
    pass
