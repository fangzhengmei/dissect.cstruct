from __future__ import annotations


class Error(Exception):
    pass


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class ResolveError(Error):
    pass


class PointerError(Error):
    """Base class for errors raised while resolving a pointer chain."""


class NullPointerDereference(PointerError):
    pass


class PointerWindowError(PointerError):
    """Raised when a pointer points outside of the configured read window."""


class PointerBudgetExceeded(PointerError):
    """Raised when resolving a pointer chain takes more dereferences than the configured budget."""


class PointerCycleError(PointerError):
    """Raised when a nested pointer chain dereferences an address that is already being resolved."""


class ArraySizeError(Error):
    pass


class ExpressionParserError(Error):
    pass


class UnknownIdentifierError(ExpressionParserError):
    pass
