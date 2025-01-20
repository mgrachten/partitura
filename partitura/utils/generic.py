#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains generic class- and numerical-related utilities
"""
import warnings
from collections import defaultdict

from typing import Union, Callable, Optional, Tuple

from textwrap import dedent
import numpy as np
from scipy.interpolate import interp1d as sc_interp1d


__all__ = ["find_nearest", "iter_current_next", "partition", "iter_subclasses"]


class _OrderedSet(dict):
    def add(self, x):
        self[x] = None

    def remove(self, x):
        self.pop(x, None)


def find_nearest(array, value):
    """
    Return the index of the value in `array` that is closest to `value`.

    Parameters
    ----------
    array: ndarray
        Array of numbers
    value: float
        The query value

    Returns
    -------
    int
        Index of closest value
    """

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or np.abs(value - array[idx - 1]) <= np.abs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


# we need a globally unique value to detect whether a keyword argument was
# passed to iter_current_next
_sentinel = object()


def iter_current_next(iterable, start=_sentinel, end=_sentinel):
    """Iterate over pairs of consecutive values in an iterable.

    This creates generator that yields a (current, next) tuple per element. If the
    iterable contains less than two elements a StopIteration exception is
    raised.

    Parameters
    ----------
    iterable: iterable
        Iterable to take values from
    start: object, optional
        If specified, this value will be treated as if it were the first element
        of the iterator
    end: object, optional
        If specified, this value will be treated as if it were the last element
        of the iterator

    Yields
    ------
    (object, object)
        Pairs of items

    Examples
    --------

    >>> for pair in iter_current_next([]):
    ...     print(pair)

    >>> for pair in iter_current_next([0]):
    ...     print(pair)

    >>> for pair in iter_current_next([0, 1, 2]):
    ...     print(pair)
    (0, 1)
    (1, 2)

    >>> for pair in iter_current_next([0, 1, 2], start=None):
    ...     print(pair)
    (None, 0)
    (0, 1)
    (1, 2)

    >>> for pair in iter_current_next([0, 1, 2], end='end_value'):
    ...     print(pair)
    (0, 1)
    (1, 2)
    (2, 'end_value')

    >>> for pair in iter_current_next([], start='start', end='end'):
    ...     print(pair)
    ('start', 'end')

    """
    iterable = iter(iterable)

    cur = start
    try:
        if cur is _sentinel:
            cur = next(iterable)

        while True:
            nxt = next(iterable)
            yield (cur, nxt)
            cur = nxt

    except StopIteration:
        if cur is not _sentinel and end is not _sentinel:
            yield (cur, end)


def iter_subclasses(cls, _seen=None):
    """
    iter_subclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    Examples
    --------

    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in iter_subclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in iter_subclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """

    if not isinstance(cls, type):
        raise TypeError(
            "iter_subclasses must be called with " "new-style classes, not %.100r" % cls
        )
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in iter_subclasses(sub, _seen):
                yield sub


def first_order_derivative(func, x0, dx=0.5):
    """
    Compute the first order derivative of a function at a point `x0` using
    Parameters
    ----------
    func: callable
        Function to differentiate
    x0: np.ndarray
        Points at which to differentiate
    dx: float
        Step size

    Returns
    -------
    np.ndarray
        First order derivative of `func` at `x0`
    """
    # fixed order
    n = 1
    # fixed number of points to use, must be odd.
    num_points = 3
    weights = np.array([-1, 0, 1]) / 2.0
    val = 0.0
    ho = num_points >> 1
    for k in range(num_points):
        val += weights[k] * func(x0 + (k - ho) * dx)
    return val / np.prod((dx,) * n, axis=0)


class ReplaceRefMixin(object):
    """This class is a utility mixin class to replace references to
    objects with references to other objects. This is functionality is
    used when unfolding timelines.

    To use this functionality, a class should inherit from this class,
    and keep a list of all attributes that contain references.

    Examples
    --------
    The following class defines `prev` as a referential attribute, to
    be replaced when a class instance is copied:

    >>> class MyClass(ReplaceRefMixin):
    ...     def __init__(self, prev=None):
    ...         super().__init__()
    ...         self.prev = prev
    ...         self._ref_attrs.append('prev')

    Create two instance `a1` and `a2`, where `a1` is the `prev` of
    `a2``

    >>> a1 = MyClass()
    >>> a2 = MyClass(a1)

    Copy `a1` and `a2` to `b1` and `b2`, respectively:
    >>> from copy import copy
    >>> b1 = copy(a1)
    >>> b2 = copy(a2)

    After copying the prev of `b2` is `a1`, not `b1`:

    >>> b2.prev == b1
    False
    >>> b2.prev == a1
    True

    To fix that we define an object map:

    >>> object_map = {a1: b1, a2: b2}

    and replace the references according to the map:

    >>> b2.replace_refs(object_map)
    >>> b2.prev == b1
    True

    """

    def __init__(self):
        self._ref_attrs = []

    def replace_refs(self, o_map):
        if hasattr(self, "_ref_attrs"):
            for attr in self._ref_attrs:
                o = getattr(self, attr)
                if o is None:
                    pass
                elif isinstance(o, list):
                    o_list_new = []

                    for o_el in o:
                        if o_el in o_map:
                            o_list_new.append(o_map[o_el])
                        else:
                            warnings.warn(
                                dedent(
                                    """reference not found in
                            o_map: {} start={} end={}, substituting None
                            """.format(
                                        o_el, o_el.start, o_el.end
                                    )
                                )
                            )
                            o_list_new.append(None)

                    setattr(self, attr, o_list_new)
                else:
                    if o in o_map:
                        o_new = o_map[o]
                    else:
                        warnings.warn(
                            dedent(
                                """reference not found in o_map:
                        {} start={} end={}, substituting None
                        """.format(
                                    o, o.start, o.end
                                )
                            )
                        )
                        o_new = None
                    setattr(self, attr, o_new)


class ComparableMixin(object):
    """
    Mixin class that makes instances comparable in a rich way (i.e. in !=, <, <=
    etc), by just implementing a _cmpkey() method that returns a comparable
    value.

    source:
    http://regebro.wordpress.com/2010/12/13/python-implementing-rich-comparison-the-correct-way/

    Examples
    --------

    >>> class MyClass(ComparableMixin):
    ...     def __init__(self, x):
    ...         self.x = x
    ...     def _cmpkey(self):
    ...         return self.x
    >>>
    >>> a = MyClass(3)
    >>> b = MyClass(4)
    >>> a == b
    False
    >>> a < b
    True

    """

    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


def partition(func, iterable):
    """
    Return a dictionary containing the equivalence classes (actually bags)
    of `iterable`, partioned according to `func`. The value of a key `k` is the
    list of all elements `e` from iterable such that `k = func(e)`.

    Examples
    ========

    The following example groups the integers from 0 to 10 by their
    respective modulo 3 values:

    >>> lst = range(10)
    >>> partition(lambda x: x % 3, lst)
    {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}

    """
    result = defaultdict(list)
    for v in iterable:
        result[func(v)].append(v)
    return dict(result)


def add_field(a, descr):
    """
    Return a new array that is like `a`, but has additional fields.
    The contents of `a` are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified. Source [8]_.

    Parameters
    ----------
    a: np.ndarray
        A structured numpy array
    descr: np.dtype or list
        A numpy type description of the new fields

    Returns
    -------
    np.ndarray
         The new structured numpy array

    Examples
    --------
    >>> import numpy as np
    >>> sa = np.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == np.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == np.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> np.all(sa['id'] == sb['id'])
    True
    >>> np.all(sa['name'] == sb['name'])
    True

    Notes
    -----
    Source:

    .. [8]
       https://stackoverflow.com/questions/1201817/\
adding-a-field-to-a-structured-numpy-array

    """
    if a.dtype.fields is None:
        raise ValueError("`A` must be a structured numpy array")

    if isinstance(descr, np.dtype):
        descr = descr.descr
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


def sorted_dict_items(items, key=None):
    for item in sorted(items, key=key):
        yield item


def show_diff(a, b):
    """
    Show the difference between two strings, using the difflib package. The
    difference is printed to stdout.

    Parameters
    ----------
    a: str
        First string
    b: str
        Second string
    """

    import difflib

    differ = difflib.Differ()
    for li in differ.compare(a.split("\n"), b.split("\n")):
        print(li)


class PrettyPrintTree(object):
    def __init__(self):
        self.stack = []

    def push(self):
        self.stack.append(TreeSymbol())

    def pop(self):
        self.stack.pop()

    def next_item(self):
        assert len(self.stack) > 0
        self.stack[-1].next_item()

    def last_item(self):
        assert len(self.stack) > 0
        self.stack[-1].last_item()

    def __str__(self):
        return "".join(str(sym) for sym in self.stack)


class TreeSymbol(object):
    def __init__(self):
        self.symbols = [" │  ", " ├─ ", " └─ ", "    "]
        self.state = 0

    def next_item(self):
        self.state = 1

    def last_item(self):
        self.state = 2

    def __str__(self):
        sym = self.symbols[self.state]
        if self.state == 1:
            self.state = 0
        elif self.state == 2:
            self.state = 3
        return sym


def search(states, success, expand, combine):
    while len(states) > 0:
        state = states.pop(0)
        if success(state):
            return state
        else:
            states = combine(expand(state), states)


def interp1d(
    x: np.ndarray,
    y: np.ndarray,
    dtype: Optional[type] = None,
    axis: int = -1,
    kind: Union[str, int] = "linear",
    copy=True,
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=False,
) -> Callable[[Union[float, int, np.ndarray]], np.ndarray]:
    """
    Interpolate a 1-D function using scipy's interp1d method. This utility allows for
    handling the case where `x` and `y` are only a single value (i.e. have length one,
    which results in a ValueError if using scipy's version directly). It also allows for
    specifying the dtype of the output.

    The description of the parameters has been taken from `scipy.interpolate.interp1d`.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``. This class returns a function whose call method uses
    interpolation to find the value of new points.


    Parameters
    ----------
    x : (N,) np.ndarray
        A 1-D array of real values.
    y : (...,N,...) np.ndarray
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    dtype : type, optional
        Type of the output array (e.g.,  `float`, `int`). By default it is set to
        None (i.e., the array will have the same type as the outputs from
        scipy's interp1d method.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless ``fill_value="extrapolate"``.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``.
        - If "extrapolate", then points outside the data range will be
          extrapolated.
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Returns
    -------
    interp_fun : callable
        The interpolator instance. This method takes an input array, float
        or integer and returns an array with the specified dtype (if `dtype`
        is not None).
    """
    if len(x) > 1:
        interp_fun = sc_interp1d(
            x=x,
            y=y,
            kind=kind,
            axis=axis,
            copy=copy,
            bounds_error=bounds_error,
            fill_value=fill_value,
            assume_sorted=assume_sorted,
        )

    else:
        # If there is only one value for x and y, assume that the method
        # will always return the same value for any input.

        def interp_fun(
            input_var: Union[float, int, np.ndarray]
        ) -> Callable[[Union[float, int, np.ndarray]], np.ndarray]:
            if y.ndim > 1:
                result = np.broadcast_to(y, (len(np.atleast_1d(input_var)), y.shape[1]))
            else:
                result = np.broadcast_to(y, (len(np.atleast_1d(input_var)),))

            if not isinstance(input_var, np.ndarray):
                # the output of scipy's interp1d is always an array
                result = np.array(result[0])

            return result

    if dtype is not None:

        def typed_interp(
            input_var: Union[float, int, np.ndarray]
        ) -> Callable[[Union[float, int, np.ndarray]], np.ndarray]:
            return interp_fun(input_var).astype(dtype)

        return typed_interp
    else:
        return interp_fun


# def search_recursive(states, success, expand, combine):
#     try:
#         if not states:
#             return None
#         elif success(states[0]):
#             return states[0]
#         else:
#             new_states = combine(expand(states[0]), states[1:])
#             return search_recursive(new_states, success, expand, combine)
#     except RecursionError:
#         warnings.warn('search exhausted stack, bailing out')
#         return None


def monotonize_times(
    s: np.ndarray,
    x: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate linearly over as many points in `s` as necessary to
    obtain a monotonic sequence. The minimum and maximum of `s` are
    prepended and appended, respectively, to ensure monotonicity at
    the bounds of `s`.
    Parameters
    ----------
    s : np.ndarray
        a sequence of numbers s(x) which we want to monotonize
    x : np.ndarray or None
        The input variable of sequence s(x).
    Returns
    -------
    s_mono: np.ndarray
       a monotonic sequence that has been linearly interpolated using a subset of s
    x_mono: np.ndarray
        The input of the monotonic sequence.
    """
    eps = np.finfo(float).eps

    _s = np.r_[np.min(s) - eps, s, np.max(s) + eps]
    if x is not None:
        _x = np.r_[np.min(x) - eps, x, np.max(x) + eps]
    else:
        _x = np.r_[-eps, np.arange(len(s)), len(s) - 1 + eps]

    s_mono = np.maximum.accumulate(s)
    mask = np.r_[False, True, (np.diff(s_mono) != 0), False]
    x_mono = _x[1:-1]
    s_mono = interp1d(_x[mask], _s[mask], fill_value="extrapolate")(x_mono)

    return s_mono, x_mono


if __name__ == "__main__":
    import doctest

    doctest.testmod()
