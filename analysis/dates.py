"""Membership of published date bounds in half-open calendar periods."""

from math import isfinite
from numbers import Real


def overlaps_period(bounds, start, end):
    """Test positive interval overlap, or membership of a point-valued date.

    Missing bounds provide no dated membership. Reversed or nonnumeric bounds
    are invalid assertions and raise rather than silently changing chronology.
    This operation assigns no probability distribution or dating precision.
    """
    if not all(isinstance(v, Real) and isfinite(v) for v in (start, end)) or start >= end:
        raise ValueError("A period requires finite increasing calendar bounds")
    if bounds is None or len(bounds) == 0:
        return False
    if len(bounds) != 2:
        raise ValueError("A date requires two bounds")
    if any(v is None for v in bounds):
        return False
    if not all(isinstance(v, Real) and isfinite(v) for v in bounds):
        raise ValueError("Date bounds must be finite calendar numbers")
    low, high = bounds
    if low > high:
        raise ValueError("Date bounds are reversed")
    return start <= low < end if low == high else max(low, start) < min(high, end)
