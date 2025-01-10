#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains utilities for parsing and formatting match lines.
"""
from __future__ import annotations

from typing import Tuple, Any, Optional, Union, List, Dict, Callable
import re

import numpy as np

from collections import namedtuple

from partitura.utils.music import (
    ALTER_SIGNS,
    fifths_mode_to_key_name,
    note_name_to_pitch_spelling,
    key_name_to_fifths_mode,
    MAJOR_KEYS,
    MINOR_KEYS,
)

Version = namedtuple("Version", ["major", "minor", "patch"])

# General patterns
rational_pattern = re.compile(r"^(?P<numerator>[0-9]+)/(?P<denominator>[0-9]+)$")
double_rational_pattern = re.compile(
    r"^(?P<numerator>[0-9]+)/(?P<denominator>[0-9]+)/(?P<tuple_div>[0-9]+)$"
)
integer_pattern = re.compile(r"^(?P<integer>[0-9]+)$")
version_pattern = re.compile(
    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
)
attribute_list_pattern = re.compile(r"^\[(?P<attributes>.*)\]")

key_signature_pattern = re.compile(
    (
        r"(?P<step1>[A-G])(?P<alter1>[#b]*)\s*"
        r"(?P<mode1>[a-zA-z]+)/*"
        "(?P<step2>[A-G]*)(?P<alter2>[#b]*)"
        r"\s*(?P<mode2>[a-zA-z]*)"
    )
)

pitch_class_pattern = re.compile("(?P<step>[A-Ga-g])(?P<alter>[#bn]*)")

number_pattern = re.compile(r"\d+")
vnumber_pattern = re.compile(r"v\d+")
fingering_pattern = re.compile(r"fingering\d+")

# For matchfiles before 1.0.0.
old_version_pattern = re.compile(r"^(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)")


def interpret_version(version_string: str) -> Version:
    """
    Parse matchfile format version from a string. This method
    parses a string like "1.0.0" and returns a Version instance.

    Parameters
    ----------
    version_string : str
        The string containg the version. The version string should be
        in the form "{major}.{minor}.{patch}" or "{minor}.{patch}" for versions
        previous to 1.0.0. Incorrectly formatted strings
        will result in an error.

    Returns
    -------
    version : Version
        A named tuple specifying the version
    """
    version_info = version_pattern.search(version_string)

    if version_info is not None:
        ma, mi, pa = version_info.groups()
        version = Version(int(ma), int(mi), int(pa))
        return version

    # If using the first pattern fails, try with old version
    version_info = old_version_pattern.search(version_string)

    if version_info is not None:
        mi, pa = version_info.groups()
        version = Version(0, int(mi), int(pa))
        return version

    else:
        raise ValueError(f"The version '{version_string}' is incorrectly formatted!")


def format_version(version: Version) -> str:
    """
    Format version as a string.

    Parameters
    ----------
    version : Version
        A Version instance.

    Returns
    -------
    version_str : str
        A string representation of the version.
    """
    ma, mi, pa = version

    version_str = f"{ma}.{mi}.{pa}"
    return version_str


def interpret_as_int(value: str) -> int:
    """
    Interpret value as an integer

    Parameters
    ----------
    value : str
       The value to interpret as integer.

    Returns
    -------
    int
        The value cast as an integer.
    """
    return int(value)


def format_int(value: Optional[int]) -> str:
    """
    Format a string from an integer

    Parameters
    ----------
    value : int
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return f"{int(value)}" if value is not None else "-"


def interpret_as_float(value: str) -> float:
    """
    Interpret value as a float

    Parameters
    ----------
    value : str
       The string to interpret as float.

    Returns
    -------
    int
        The value cast as an float.
    """
    return float(value)


def format_float(value: float) -> str:
    """
    Format a float as a string (with 4 digits of precision).

    Parameters
    ----------
    value : float
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return f"{value:.4f}"


def format_float_unconstrained(value: float) -> str:
    """
    Format a float as a string.

    Parameters
    ----------
    value : float
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return str(value)


def interpret_as_string(value: Any) -> str:
    """
    Interpret value as a string. This method is for completeness.

    Parameters
    ----------
    value : Any
       The value to be interpreted as a string.

    Returns
    -------
    int
        The string representation of the value.
    """
    return str(value)


old_string_pat = re.compile(r"'(?P<value>.+)'")


def interpret_as_string_old(value: str) -> str:
    val = old_string_pat.match(value)
    if val is not None:
        return val.group("value").strip()
    else:
        return value.strip()


def format_string(value: str) -> str:
    """
    Format a string as a string (for completeness ;).

    Parameters
    ----------
    value : int
        The value to be converted to format as a string.

    Returns
    -------
    str
        The value formatted as a string.
    """
    return str(value).strip()


def format_string_old(value: str) -> str:
    return f"'{value.strip()}'"


class FractionalSymbolicDuration(object):
    """
    A class to represent symbolic duration information.

    Parameters
    ----------
    numerator : int
        The value of the numerator.
    denominator: int
        The denominator of the duration (whole notes = 1, half notes = 2, etc.)
    tuple_div : int (optional)
        Tuple divisor (for triplets, etc.). For example a single note in a quintuplet
        with a total duration of one quarter could be specified as
        `duration = FractionalSymbolicDuration(1, 4, 5)`.
    add_components : List[Tuple[int, int, Optional[int]]] (optional)
        additive components (to express durations like 1/4+1/16+1/32). The components
        are a list of tuples, each of which contains its own numerator, denominator
        and tuple_div (or None). To represent the components in the example above
        this variable would look like
        `add_components = [(1, 4, None), (1, 16, None), (1, 32, None)]`.
    """

    def __init__(
        self,
        numerator: int,
        denominator: int = 1,
        tuple_div: Optional[int] = None,
        add_components: Optional[List[Tuple[int, int, Optional[int]]]] = None,
    ) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.tuple_div = tuple_div
        self.add_components = add_components
        self.bound_integers(1024)

    def _str(
        self,
        numerator: int,
        denominator: int,
        tuple_div: Optional[int],
    ) -> str:
        """
        Helper for representing an instance as a string.
        """
        if denominator == 1 and tuple_div is None:
            return str(numerator)
        else:
            if tuple_div is None:
                return "{0}/{1}".format(numerator, denominator)
            else:
                return "{0}/{1}/{2}".format(numerator, denominator, tuple_div)

    def bound_integers(self, bound: int) -> None:
        """
        Bound numerator and denominator
        """
        denominators = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            28,
            32,
            48,
            64,
            96,
            128,
        ]
        sign = (
            np.sign(self.numerator)
            * np.sign(self.denominator)
            * (np.sign(self.tuple_div) if self.tuple_div is not None else 1)
        )
        self.numerator = np.abs(self.numerator)
        self.denominator = np.abs(self.denominator)
        self.tuple_div = np.abs(self.tuple_div) if self.tuple_div is not None else None

        if self.numerator > bound or self.denominator > bound:
            val = float(self.numerator / self.denominator)
            dif = []
            for den in denominators:
                if np.round(val * den) > 0.9:
                    dif.append(np.abs(np.round(val * den) - val * den))
                else:
                    dif.append(np.abs(1 - val * den))

            difn = np.array(dif)
            min_idx = int(np.argmin(difn))

            self.denominator = denominators[min_idx]
            if int(np.round(val * self.denominator)) < 1:
                self.numerator = sign * 1
            else:
                self.numerator = sign * int(np.round(val * self.denominator))

    def __str__(self) -> str:
        """
        Represent an instance as a string.
        """
        if self.add_components is None:
            return self._str(self.numerator, self.denominator, self.tuple_div)
        else:
            r = [self._str(*i) for i in self.add_components]
            return "+".join(r)

    def __add__(
        self, sd: Union[FractionalSymbolicDuration, int]
    ) -> FractionalSymbolicDuration:
        """
        Define addition between FractionalSymbolicDuration instances.

        Parameters
        ----------
        sd : Union[FractionalSymbolicDuration, int]
            A FractionalSymbolicDuration instance or an integer to add
            to the current instance (self).

        Returns
        -------
        FractionalSymbolicDuration
            A new instance with the value equal to the sum
            of `sd` + `self`.
        """
        if isinstance(sd, int):
            sd = FractionalSymbolicDuration(sd, 1)

        dens = np.array(
            [
                self.denominator
                * (self.tuple_div if self.tuple_div is not None else 1),
                sd.denominator * (sd.tuple_div if sd.tuple_div is not None else 1),
            ],
            dtype=int,
        )
        new_den = np.lcm(dens[0], dens[1])
        a_mult = new_den // dens
        new_num = np.dot(a_mult, [self.numerator, sd.numerator])

        if self.add_components is None and sd.add_components is None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div),
                (sd.numerator, sd.denominator, sd.tuple_div),
            ]

        elif self.add_components is not None and sd.add_components is None:
            add_components = self.add_components + [
                (sd.numerator, sd.denominator, sd.tuple_div)
            ]
        elif self.add_components is None and sd.add_components is not None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div)
            ] + sd.add_components
        else:
            add_components = self.add_components + sd.add_components

        # Remove spurious components with 0 in the numerator
        add_components = [c for c in add_components if c[0] != 0]

        return FractionalSymbolicDuration(
            numerator=new_num,
            denominator=new_den,
            add_components=add_components,
        )

    def __eq__(self, sd: FractionalSymbolicDuration) -> bool:
        """
        Equal magic method
        """
        is_equal = all(
            [
                getattr(self, attr, None) == getattr(sd, attr, None)
                for attr in ("numerator", "denominator", "tuple_div", "add_components")
            ]
        )

        return is_equal

    def __ne__(self, sd: FractionalSymbolicDuration) -> bool:
        """
        Not equal magic method
        """
        not_equal = any(
            [
                getattr(self, attr, None) != getattr(sd, attr, None)
                for attr in ("numerator", "denominator", "tuple_div", "add_components")
            ]
        )

        return not_equal

    def __radd__(
        self, sd: Union[FractionalSymbolicDuration, int]
    ) -> FractionalSymbolicDuration:
        return self.__add__(sd)

    def __float__(self) -> float:
        # Cast as float since the ability to return an instance of a strict
        # subclass of float is deprecated, and may be removed in a future
        # version of Python. (following a deprecation warning)
        return float(self.numerator / (self.denominator * (self.tuple_div or 1)))

    @classmethod
    def from_string(cls, string: str, allow_additions: bool = True):
        m = rational_pattern.match(string)
        m2 = double_rational_pattern.match(string)
        m3 = integer_pattern.match(string)
        if m:
            groups = m.groups()
            return cls(*[int(g) for g in groups])
        elif m2:
            groups = m2.groups()
            return cls(*[int(g) for g in groups])
        elif m3:
            return cls(numerator=int(m3.group("integer")))

        else:
            if allow_additions:
                parts = string.split("+")

                if len(parts) > 1:
                    iparts = [
                        cls.from_string(
                            i,
                            allow_additions=False,
                        )
                        for i in parts
                    ]

                    # to be replaced with isinstance(i,numbers.Number)
                    if all(type(i) in (int, float, cls) for i in iparts):
                        if any([isinstance(i, cls) for i in iparts]):
                            iparts = [
                                cls(i) if not isinstance(i, cls) else i for i in iparts
                            ]
                        return sum(iparts)

        raise ValueError(
            f"{string} cannot be interpreted as FractionalSymbolicDuration"
        )


def interpret_as_fractional(value: str) -> FractionalSymbolicDuration:
    """
    Interpret string as FractionalSymbolicDuration
    """

    content = attribute_list_pattern.search(value)

    if content is not None:
        # string includes square brackets
        vals_string = content.group("attributes")
        content_list = [
            FractionalSymbolicDuration.from_string(v, allow_additions=True)
            for v in vals_string.split(",")
        ]
        return content_list

    return FractionalSymbolicDuration.from_string(value, allow_additions=True)


def format_fractional(
    value: Union[List[FractionalSymbolicDuration], FractionalSymbolicDuration]
) -> str:
    """
    Format fractional symbolic duration as string
    """

    if isinstance(value, list):
        return format_list(value)

    return str(value)


def format_fractional_rational(value: FractionalSymbolicDuration) -> str:
    """
    Format fractional symbolic duration as string and ensure that the output
    is always rational ("a/b")
    """

    if value.denominator == 1 and value.tuple_div is None:
        out = f"{value.numerator}/1"

    else:
        out = str(value)

    return out


def interpret_as_list(value: str) -> List[str]:
    """
    Interpret string as list of values.

    Parameters
    ----------
    value: str

    Returns
    -------
    content_list : List[str]
    """
    content = attribute_list_pattern.search(value)

    if content is not None:
        # string includes square brackets
        vals_string = content.group("attributes")
        content_list = [v.strip() for v in vals_string.split(",")]

    else:
        # value is not inside square brackets
        content_list = [v.strip() for v in value.split(",")]

    return content_list


def interpret_as_list_int(value: str) -> List[int]:
    string_list = interpret_as_list(value)
    return [int(v) for v in string_list]


def interpret_as_list_fractional(value: str) -> List[FractionalSymbolicDuration]:
    string_list = interpret_as_list(value)
    return [FractionalSymbolicDuration.from_string(v) for v in string_list]


def format_list(value: List[Any]) -> str:
    formatted_string = f"[{','.join([str(v) for v in value])}]"
    return formatted_string


def format_accidental(value: Optional[int]) -> str:
    alter = "n" if value == 0 else ALTER_SIGNS[value]

    return alter


def format_accidental_old(value: Optional[int]) -> str:
    if value is None:
        return "-"
    else:
        return format_accidental(value)


def format_pnote_id(value: Any) -> str:
    pnote_id = f"n{str(value)}" if not str(value).startswith("n") else str(value)

    return pnote_id


## Methods for parsing special attributes


class MatchParameter(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_string(self, *args, **kwargs) -> MatchParameter:
        raise NotImplementedError


class MatchKeySignature(MatchParameter):
    def __init__(
        self,
        fifths: int,
        mode: str,
        fifths_alt: Optional[int] = None,
        mode_alt: Optional[str] = None,
        is_list: bool = False,
        other_components: Optional[List[MatchKeySignature]] = None,
        fmt: str = "v1.0.0",
    ):
        super().__init__()
        self.fifths = fifths
        self.mode = mode
        self.fifths_alt = fifths_alt
        self.mode_alt = mode_alt
        self.is_list = is_list
        self.other_components = [] if other_components is None else other_components
        self.fmt = fmt

    @property
    def fmt(self) -> str:
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: str) -> None:
        self._fmt = fmt
        for component in self.other_components:
            component.fmt = fmt

    def fifths_mode_to_key_name_v0_1_0(self, fifths: int, mode: str) -> str:
        if mode in ("major", None, "none", 1):
            keylist = MAJOR_KEYS
            suffix = "major"
        elif mode == "minor":
            keylist = MINOR_KEYS
            suffix = "minor"

        step, alter, _ = note_name_to_pitch_spelling(f"{keylist[fifths + 7]}4")
        alter_str = "n" if (alter is None or alter == 0) else ALTER_SIGNS[alter]
        name = f"{step.lower()}{alter_str}"
        ks = f"[{name},{suffix}]"

        return ks

    def fifths_mode_to_key_name_v0_3_0(self, fifths: int, mode: str) -> str:
        if mode in ("major", None, "none", 1):
            keylist = MAJOR_KEYS
            suffix = "Maj"
        elif mode == "minor":
            keylist = MINOR_KEYS
            suffix = "min"

        name = keylist[fifths + 7]

        ks = f"{name} {suffix}"

        return ks

    def __str__(self):
        if self.fmt == "v1.0.0":
            ks = fifths_mode_to_key_name(self.fifths, self.mode)

            if self.fifths_alt is not None:
                ks = f"{ks}/{fifths_mode_to_key_name(self.fifths_alt, self.mode_alt)}"

        if self.fmt == "v0.3.0":
            ks = self.fifths_mode_to_key_name_v0_3_0(self.fifths, self.mode)

            if self.fifths_alt is not None:
                ks = f"{ks}/{self.fifths_mode_to_key_name_v0_3_0(self.fifths_alt, self.mode_alt)}"

        if self.fmt == "v0.1.0":
            ks = self.fifths_mode_to_key_name_v0_1_0(self.fifths, self.mode)

        if self.is_list:
            return format_list([ks] + self.other_components)
        return ks

    def __eq__(self, ks: MatchKeySignature) -> bool:
        crit = (
            self.fifths == ks.fifths
            and self.mode == ks.mode
            and self.fifths_alt == ks.fifths_alt
            and self.mode_alt == ks.mode_alt
            and self.other_components == ks.other_components
        )

        return crit

    @classmethod
    def _parse_key_signature(cls, kstr: str) -> MatchKeySignature:
        # import pdb
        # pdb.set_trace()
        ksinfo = key_signature_pattern.search(kstr)

        if ksinfo is None:
            fmt = "v1.0.0"
            ksinfo = kstr.split("/")
            fifths1, mode1 = key_name_to_fifths_mode(ksinfo[0].upper())
            fifths2, mode2 = None, None
            if len(ksinfo) == 2:
                fifths2, mode2 = key_name_to_fifths_mode(ksinfo[1].upper())
        else:
            fmt = "v0.3.0"
            step1, alter1, mode1, step2, alter2, mode2 = ksinfo.groups()
            mode1_str = "m" if mode1.lower() in ("minor", "min") else ""
            fifths1, mode1 = key_name_to_fifths_mode(
                f"{step1.upper()}{alter1}{mode1_str}"
            )

            if step2 != "":
                mode2_str = "m" if mode2.lower() in ("minor", "min") else ""
                fifths2, mode2 = key_name_to_fifths_mode(
                    f"{step2.upper()}{alter2}{mode2_str}"
                )
            else:
                fifths2, mode2 = None, None

        return cls(
            fifths=fifths1,
            mode=mode1,
            fifths_alt=fifths2,
            mode_alt=mode2,
            is_list=False,
            fmt=fmt,
        )

    @classmethod
    def from_string(cls, string: str) -> MatchKeySignature:
        content = interpret_as_list(string)

        if len(content) == 2:
            # try parsing it as v0.1.0
            if content[1].lower() in ("minor", "major", "min", "maj"):
                note_info = pitch_class_pattern.search(content[0].lower())

                mode_str = "m" if content[1].lower() in ("min", "minor") else ""

                if note_info is not None:
                    step, alter = note_info.groups()

                alter_str = alter.replace("n", "")

                fifths, mode = key_name_to_fifths_mode(
                    f"{step.upper()}{alter_str}{mode_str}"
                )

                return cls(fifths=fifths, mode=mode, fmt="v0.1.0")

        if len(content) > 0:
            ksigs = [cls._parse_key_signature(ksig) for ksig in content]

            ks = ksigs[0]
            ks.other_components = ksigs[1:] if len(ksigs) > 1 else []

            return ks


def interpret_as_key_signature(value: str) -> MatchKeySignature:
    ks = MatchKeySignature.from_string(value)
    return ks


def format_key_signature_v1_0_0(value: MatchKeySignature) -> str:
    value.is_list = False
    value.fmt = "v1.0.0"
    return str(value)


def format_key_signature_v0_3_0(value: MatchKeySignature) -> str:
    value.is_list = False
    value.fmt = "v0.3.0"
    return str(value)


def format_key_signature_v0_3_0_list(value: MatchKeySignature) -> str:
    value.is_list = True
    value.fmt = "v0.3.0"
    return str(value)


def format_key_signature_v0_1_0(value: MatchKeySignature) -> str:
    value.is_list = False
    value.fmt = "v0.1.0"
    return str(value)


def format_time_signature_v0_1_0_list(value: MatchKeySignature) -> str:
    value.is_list = True
    value.fmt = "v0.1.0"
    return str(value)


class MatchTimeSignature(MatchParameter):
    def __init__(
        self,
        numerator: int,
        denominator: int,
        other_components: Optional[List[Any]],
        is_list: bool = False,
    ) -> None:
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator
        self.other_components = other_components
        self.is_list = is_list

    def __str__(self):
        ts = f"{self.numerator}/{self.denominator}"
        if self.is_list:
            return format_list([ts] + self.other_components)
        else:
            return ts

    def __eq__(self, ts: MatchKeySignature) -> bool:
        crit = (
            (self.numerator == ts.numerator)
            and (self.denominator == ts.denominator)
            and (self.other_components == ts.other_components)
        )
        return crit

    @classmethod
    def from_string(cls, string: str, is_list: bool = False) -> MatchTimeSignature:
        content = interpret_as_list_fractional(string.strip())
        numerator = content[0].numerator
        denominator = content[0].denominator

        other_components = [] if len(content) == 1 else content[1:]

        return cls(
            numerator=numerator,
            denominator=denominator,
            other_components=other_components,
            is_list=is_list,
        )


def interpret_as_time_signature(value: str) -> MatchTimeSignature:
    ts = MatchTimeSignature.from_string(value)
    return ts


def format_time_signature(value: MatchTimeSignature) -> str:
    value.is_list = False
    return str(value)


def format_time_signature_list(value: MatchTimeSignature) -> str:
    value.is_list = True
    return str(value)


class MatchTempoIndication(MatchParameter):
    def __init__(
        self,
        value: str,
        is_list: bool = False,
    ):
        super().__init__()
        self.value = self.from_string(value)[0]
        self.is_list = is_list

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, string: str) -> MatchTempoIndication:
        content = interpret_as_list(string)
        return content


def interpret_as_tempo_indication(value: str) -> MatchTempoIndication:
    tempo_indication = MatchTempoIndication.from_string(value)
    return tempo_indication


def format_tempo_indication(value: MatchTempoIndication) -> str:
    value.is_list = False
    return str(value)


## Miscellaneous utils


def to_snake_case(field_name: str) -> str:
    """
    Convert name in camelCase to snake_case
    """
    snake_case = "".join(
        [f"_{fn.lower()}" if fn.isupper() else fn for fn in field_name]
    )

    if snake_case.startswith("_"):
        snake_case = snake_case[1:]

    return snake_case


def to_camel_case(field_name: str) -> str:
    """
    Convert name in snake_case to camelCase
    """
    parts = field_name.split("_")

    camel_case = f"{parts[0].lower()}"

    if len(parts) > 1:
        camel_case += "".join([p.title() for p in parts[1:]])

    return camel_case


def get_kwargs_from_matchline(
    matchline: str,
    pattern: re.Pattern,
    field_names: Tuple[str],
    class_dict: Dict[str, Tuple[Callable, Callable, type]],
    pos: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Parameters
    ----------
    matchline: str
    pattern: re.Pattern
    field_names: Tuple[str]
    class_dict: Dict[str, Tuple[Callable, Callable, type]]
    pos: int

    Returns
    -------
    kwargs : dict

    """
    kwargs = None
    match_pattern = pattern.search(matchline, pos=pos)

    if match_pattern is not None:
        kwargs = dict(
            [
                (to_snake_case(fn), class_dict[fn][0](match_pattern.group(fn)))
                for fn in field_names
            ]
        )

    return kwargs
