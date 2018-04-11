
import interval
import numpy
from numpy import inf

def interval_bounds(i):
    """
    Get a single (lower_bound, upper_bound) tuple for the hull of an interval i (raise an exception if empty interval).
    """
    assert isinstance(i, interval.interval)
    return interval.interval.hull([i])[0]

def interval_finite(i):
    (a, b) = interval_bounds(i)
    return numpy.isfinite(a) and numpy.isfinite(b)

def interval_range(i):
    """
    Get range from hull of interval i.
    """
    try:
        (a, b) = interval_bounds(i)
    except ValueError:
        return 0.0
    return b - a

def interval_abs(i):
    """
    Get absolute value of interval i.
    """
    return (i | (-i)) & interval.interval[0, inf]

def interval_sign(i):
    """
    Get signum (sign function) of interval i.
    """
    ans = interval.interval()
    if len(i & interval.interval[0, 0]):
        ans |= interval.interval[0, 0]
    r1 = interval_clip_positive(i)
    r2 = interval_clip_positive(-i)
    try:
        r1 = interval_bounds(r1)
    except:
        r1 = None
    try:
        r2 = interval_bounds(r2)
    except:
        r2 = None
    #print(r1)
    #print(r2)
    if r1 is not None:
        if r1[1] > 0:
            ans |= interval.interval[1, 1]
        if r1[0] == 0:
            ans |= interval.interval[0, 0]
    if r2 is not None:
        if r2[1] > 0:
            ans |= interval.interval[-1, -1]
        if r2[0] == 0:
            ans |= interval.interval[0, 0]
    return ans

def interval_clip_positive(i):
    """
    Get positive portion of interval (clip to [0, inf]).
    """
    return i & interval.interval[0, inf]
