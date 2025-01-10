#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods to compute note-level features.
"""
import sys
import warnings
import numpy as np
from scipy.interpolate import interp1d
import partitura.score as score

import types
from typing import List, Union, Tuple
from partitura.utils import ensure_notearray, ensure_rest_array, clef_sign_to_int
from partitura.score import ScoreLike
from collections import defaultdict

__all__ = [
    "list_note_feats_functions",
    "list_note_feature_functions",
    "print_note_feats_functions",
    "print_note_feature_functions",
    "make_note_feats",
    "make_note_features",
    "make_rest_feats",
    "make_rest_features",
    "compute_note_array",
    "full_note_array",
]


class InvalidNoteFeatureException(Exception):
    pass


def print_note_feats_functions():
    """Print a list of all featurefunction names defined in this module,
    with descriptions where available.

    """
    module = sys.modules[__name__]
    doc_indent = 4
    for name in list_note_feats_functions():
        print("* {}".format(name))
        member = getattr(sys.modules[__name__], name)
        if member.__doc__:
            print(
                " " * doc_indent + member.__doc__.replace("\n", " " * doc_indent + "\n")
            )


def list_note_feats_functions():
    """Return a list of all feature function names defined in this module.

    The feature function names listed here can be specified by name in
    the `make_note_features` and `make_rest_features` functions. For example:

    >>> feature, names = make_note_feats(part, ['metrical_feature', 'articulation_feature'])

    Returns
    -------
    list
        A list of strings

    """
    module = sys.modules[__name__]
    bfs = []
    exclude = {"make_feature"}
    for name in dir(module):
        if name in exclude:
            continue
        member = getattr(sys.modules[__name__], name)
        if isinstance(member, types.FunctionType) and name.endswith("_feature"):
            bfs.append(name)
    return bfs


def make_note_features(
    part: ScoreLike,
    feature_functions: Union[List, str],
    add_idx: bool = False,
    include_empty_features: bool = True,
    force_fixed_size: bool = False,
) -> Tuple[np.ndarray, List]:
    """Compute the specified feature functions for a part.

    The function returns the computed feature functions as a N x M
    array, where N equals `len(part.notes_tied)` and M equals the
    total number of descriptors of all feature functions that occur in
    part.

    Furthermore, the function returns the names of the feature functions.
    A list of strings of size M. The names have the name of the
    function prepended to the name of the descriptor. For example if a
    function named `abc_feature` returns descriptors `a`, `b`, and `c`,
    then the list of names returned by `make_feature(part,
    ['abc_feature'])` will be ['abc_feature.a', 'abc_feature.b',
    'abc_feature.c'].

    Parameters
    ----------
    part : ScoreLike
        A partitura scoreLike object, can be Score, Part, or PartGroup.
    feature_functions : list or str
        A list of feature functions. Elements of the list can be either
        the functions themselves or the names of a feature function as
        strings (or a mix), or the keywork "all". The feature functions specified by name are
        looked up in the `featuremixer.featurefunctions` module.
    add_idx : bool (default: False)
        If True, the index of the note in the part is added as a
        feature. This is useful for debugging.
    include_empty_features : bool (default: True)
        If True, features that are empty are included in the output.
        Otherwise, they are omitted.
    force_fixed_size : bool (default: False)
        If True, the output array uses only features that have a fixed size with no new entries added.

    Returns
    -------
    feature : ndarray
        The feature functions
    names : list
        The feature names
    """
    if isinstance(part, score.Score):
        part = score.merge_parts(part.parts)
    else:
        part = score.merge_parts(part)
    na = ensure_notearray(
        part,
        include_metrical_position=True,
        include_grace_notes=True,
        include_time_signature=True,
    )

    if len(set(na["id"])) != len(na):
        warnings.warn(
            "Length of note array {0} "
            "does not correspond to number of unique IDs {1}. "
            "Some feature functions may return spurious values.".format(
                len(na), len(set(na["id"]))
            )
        )

    acc = []
    if isinstance(feature_functions, str) and feature_functions == "all":
        feature_functions = list_note_feats_functions()
    elif not isinstance(feature_functions, list):
        raise TypeError(
            "feature_functions variable {} needs to be list or all".format(
                feature_functions
            )
        )

    for bf in feature_functions:
        # skip time_signature_feature if force_fixed_size is True
        if force_fixed_size and (
            bf == "time_signature_feature" or bf == time_signature_feature
        ):
            continue
        # skip metrical_feature if force_fixed_size is True
        if force_fixed_size and (bf == "metrical_feature" or bf == metrical_feature):
            continue

        if isinstance(bf, str):
            # get function by name from module
            func = getattr(sys.modules[__name__], bf)
        elif isinstance(bf, types.FunctionType):
            func = bf
        else:
            warnings.warn("Ignoring unknown feature function {}".format(bf))
        bf, bn = func(
            na,
            part,
            include_empty_features=(
                True if force_fixed_size else include_empty_features
            ),
        )
        # check if the size and number of the feature function are correct
        if bf.size != 0:
            if bf.shape[1] != len(bn):
                msg = (
                    "number of feature names {} does not equal "
                    "number of feature {}".format(len(bn), bf.shape[1])
                )
                raise InvalidNoteFeatureException(msg)
            n_notes = len(part.notes_tied)
            if len(bf) != n_notes:
                msg = (
                    "length of feature {} does not equal "
                    "number of notes {}".format(len(bf), n_notes)
                )
                raise InvalidNoteFeatureException(msg)

            if np.any(np.logical_or(np.isnan(bf), np.isinf(bf))):
                problematic = np.unique(
                    np.where(np.logical_or(np.isnan(bf), np.isinf(bf)))[1]
                )
                msg = "NaNs or Infs found in the following feature: {} ".format(
                    ", ".join(np.array(bn)[problematic])
                )
                raise InvalidNoteFeatureException(msg)

            # prefix feature names by function name
            bn = ["{}.{}".format(func.__name__, n) for n in bn]

            acc.append((bf, bn))

    if add_idx:
        _data, _names = zip(*acc)
        feature_data = np.column_stack(_data)
        feature_data_list = [list(f) + [i] for f, i in zip(feature_data, na["id"])]
        feature_names = [n for ns in _names for n in ns] + ["id"]
        feature_names_dtypes = list(
            zip(feature_names, ["f4"] * (len(feature_names) - 1) + ["U256"])
        )
        feature_data_struct = np.array(
            [tuple(f) for f in feature_data_list], dtype=feature_names_dtypes
        )
        return feature_data_struct
    else:
        _data, _names = zip(*acc)
        feature_data = np.column_stack(_data)
        feature_names = [n for ns in _names for n in ns]
        return feature_data, feature_names


def make_rest_features(
    part: Union[score.Part, score.PartGroup, List],
    feature_functions: Union[List, str],
    add_idx: bool = False,
) -> Tuple[np.ndarray, List]:
    """Compute the specified feature functions for a part.

    The function returns the computed feature functions as a N x M
    array, where N equals `len(part.rests)` and M equals the
    total number of descriptors of all feature functions that occur in
    part.

    Parameters
    ----------
    part : Part
        The score as a Part instance
    feature_functions : list or str
        A list of feature functions. Elements of the list can be either
        the functions themselves or the names of a feature function as
        strings (or a mix), or the keywork "all". The feature functions specified by name are
        looked up in the `featuremixer.featurefunctions` module.

    Returns
    -------
    feature : ndarray
        The feature functions
    names : list
        The feature names
    """
    if isinstance(part, score.Score):
        part = score.merge_parts(part.parts)
    else:
        part = score.merge_parts(part)
    na = ensure_rest_array(
        part,
        include_metrical_position=True,
        include_grace_notes=True,
        include_time_signature=True,
    )
    if na.size == 0:
        return np.array([])

    if len(set(na["id"])) != len(na):
        warnings.warn(
            "Length of rest array {0} "
            "does not correspond to number of unique IDs {1}. "
            "Some feature functions may return spurious values.".format(
                len(na), len(set(na["id"]))
            )
        )

    acc = []
    if isinstance(feature_functions, str) and feature_functions == "all":
        feature_functions = list_note_feats_functions()
    elif not isinstance(feature_functions, list):
        raise TypeError(
            "feature_functions variable {} needs to be list or all".format(
                feature_functions
            )
        )

    for bf in feature_functions:
        if isinstance(bf, str):
            # get function by name from module
            func = getattr(sys.modules[__name__], bf)
        elif isinstance(bf, types.FunctionType):
            func = bf
        else:
            warnings.warn("Ignoring unknown feature function {}".format(bf))
        bf, bn = func(na, part)
        # check if the size and number of the feature function are correct
        if bf.size != 0:
            if bf.shape[1] != len(bn):
                msg = (
                    "number of feature names {} does not equal "
                    "number of feature {}".format(len(bn), bf.shape[1])
                )
                raise InvalidNoteFeatureException(msg)
            n_notes = len(part.rests)
            if len(bf) != n_notes:
                msg = (
                    "length of feature {} does not equal "
                    "number of notes {}".format(len(bf), n_notes)
                )
                raise InvalidNoteFeatureException(msg)

            if np.any(np.logical_or(np.isnan(bf), np.isinf(bf))):
                problematic = np.unique(
                    np.where(np.logical_or(np.isnan(bf), np.isinf(bf)))[1]
                )
                msg = "NaNs or Infs found in the following feature: {} ".format(
                    ", ".join(np.array(bn)[problematic])
                )
                raise InvalidNoteFeatureException(msg)

            # prefix feature names by function name
            bn = ["{}.{}".format(func.__name__, n) for n in bn]

            acc.append((bf, bn))

    if add_idx:
        _data, _names = zip(*acc)
        feature_data = np.column_stack(_data)
        feature_data_list = [list(f) + [i] for f, i in zip(feature_data, na["id"])]
        feature_names = [n for ns in _names for n in ns] + ["id"]
        feature_names_dtypes = list(
            zip(feature_names, ["f4"] * (len(feature_names) - 1) + ["U256"])
        )
        feature_data_struct = np.array(
            [tuple(f) for f in feature_data_list], dtype=feature_names_dtypes
        )
        return feature_data_struct
    else:
        _data, _names = zip(*acc)
        feature_data = np.column_stack(_data)
        feature_names = [n for ns in _names for n in ns]
        return feature_data, feature_names


# alias
make_note_feats = make_note_features
make_rest_feats = make_rest_features
list_note_feature_functions = list_note_feats_functions
print_note_feature_functions = print_note_feats_functions


def compute_note_array(
    part: ScoreLike,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
    include_metrical_position=False,
    include_grace_notes=False,
    feature_functions=None,
    force_fixed_size=False,
):
    """
    Create an extended note array from this part.

    1) Without arguments this returns a structured array of onsets, offsets,
    pitch, and ID information: equivalent to part.note_array()

    2) With any of the flag arguments set to true, a column with the specified
    information will be added to the array: equivalent t0 part.note_array(*flags)

    3) With a list of strings or functions as feature_functions argument,
    a column (or multiple columns) with the specified information will
    be added to the array.
    See also:
    >>> make_note_features(part)
    For a list of features see:
    >>> list_note_feats_functions()

    Parameters
    ----------

    include_pitch_spelling : bool (optional)
        If `True`, includes pitch spelling information for each
        note. Default is False
    include_key_signature : bool (optional)
        If `True`, includes key signature information, i.e.,
        the key signature at the onset time of each note (all
        notes starting at the same time have the same key signature).
        Default is False
    include_time_signature : bool (optional)
        If `True`,  includes time signature information, i.e.,
        the time signature at the onset time of each note (all
        notes starting at the same time have the same time signature).
        Default is False
    include_metrical_position : bool (optional)
        If `True`,  includes metrical position information, i.e.,
        the position of the onset time of each note with respect to its
        measure (all notes starting at the same time have the same metrical
        position).
        Default is False
    include_grace_notes : bool (optional)
        If `True`,  includes grace note information, i.e. if a note is a
        grace note and the grace type "" for non grace notes).
        Default is False
    feature_functions : list or str
        A list of feature functions. Elements of the list can be either
        the functions themselves or the names of a feature function as
        strings (or a mix). The feature functions specified by name are
        looked up in the `featuremixer.featurefunctions` module.
    force_fixed_size : bool (default: False)
        If True, the output array uses only features that have a fixed
        size with no new entries added.


    Returns:

    note_array : structured array
    """
    if isinstance(part, score.Score):
        part = score.merge_parts(part.parts)
    else:
        part = score.merge_parts(part)
    na = ensure_notearray(
        part,
        include_pitch_spelling=include_pitch_spelling,
        include_key_signature=include_key_signature,
        include_time_signature=include_time_signature,
        include_metrical_position=include_metrical_position,
        include_grace_notes=include_grace_notes,
    )

    if feature_functions is not None:
        feature_data_struct = make_note_feats(
            part, feature_functions, add_idx=True, force_fixed_size=force_fixed_size
        )
        note_array_joined = np.lib.recfunctions.join_by("id", na, feature_data_struct)
        note_array = note_array_joined.data
        sort_idx = np.lexsort(
            (note_array["duration_div"], note_array["pitch"], note_array["onset_div"])
        )
        note_array = note_array[sort_idx]
    else:
        note_array = na
    return note_array


def full_note_array(part):
    """
    Create a note array with all available information.
    """
    return compute_note_array(
        part,
        include_pitch_spelling=True,
        include_key_signature=True,
        include_time_signature=True,
        include_metrical_position=True,
        include_grace_notes=True,
        feature_functions="all",
    )


def polynomial_pitch_feature(na, part, **kwargs):
    """Normalize pitch feature."""
    pitches = na["pitch"].astype(float)
    feature_names = ["pitch"]
    max_pitch = 127
    W = pitches / max_pitch
    return np.expand_dims(W, axis=1), feature_names


def duration_feature(na, part, **kwargs):
    """Duration feature.

    Parameters
    ----------
    na : structured array
        The Note array for Unified part.
    """

    feature_names = ["duration"]
    durations_beat = na["duration_beat"]
    W = durations_beat
    W.shape = (-1, 1)
    return W, feature_names


def onset_feature(na, part, **kwargs):
    """Onset feature

    Returns:
    * onset : the onset of the note in beats
    * score_position : position of the note in the score between 0 (the beginning of the piece) and 1 (the end of the piece)

    TODO:
    * rel_position_repetition
    """
    feature_names = ["onset", "score_position"]

    onsets_beat = na["onset_beat"]
    rel_position = normalize(onsets_beat, method="minmax")

    W = np.column_stack((onsets_beat, rel_position))

    return W, feature_names


def relative_score_position_feature(na, part, **kwargs):
    W, names = onset_feature(na, part, **kwargs)
    return W[:, 1:], names[1:]


def grace_feature(na, part, **kwargs):
    """Grace feature.

    Returns:
    * grace_note : 1 when the note is a grace note, 0 otherwise
    * n_grace : the length of the grace note sequence to which
                this note belongs (0 for non-grace notes)
    * grace_pos : the (1-based) position of the grace note in
                  the sequence (0 for non-grace notes)

    """

    feature_names = ["grace_note", "n_grace", "grace_pos"]

    W = np.zeros((len(na), 3))
    W[:, 0] = na["is_grace"]
    grace_notes = na[np.nonzero(na["is_grace"])]
    notes = (
        {n.id: n for n in part.notes_tied}
        if not np.all(na["pitch"] == 0)
        else {n.id: n for n in part.rests}
    )
    indices = np.nonzero(na["is_grace"])[0]
    for i, index in enumerate(indices):
        grace = grace_notes[i]
        n_grace = np.count_nonzero(grace_notes["onset_beat"] == grace["onset_beat"])
        W[index, 1] = n_grace
        W[index, 2] = (
            n_grace - sum(1 for _ in notes[grace["id"]].iter_grace_seq()) + 1
            if grace["id"] not in (None, "None", "")
            else 0
        )
    return W, feature_names


def clef_feature(na, part, **kwargs):
    """Clef feature

    This feature encodes the current clef of the staff of each note.
    Note that this feature does not return the staff number per note,
    see staff_feature for this information.
    """
    notes = {n.id: n for n in part.notes_tied}
    names = ["clef_sign", "clef_line", "clef_octave_change"]
    clef_dict = defaultdict(list)
    staff_numbers = set()
    clef_list = [clef for clef in part.iter_all(score.Clef)]
    if len(clef_list) > 0:
        for clef in clef_list:
            staff = clef.staff or 1
            staff_numbers.add(staff)
            time_key = "time_" + str(staff)
            clef_key = "clef_" + str(staff)
            clef_dict[time_key].append(clef.start.t)
            clef_dict[clef_key].append(clef)

        for staff in staff_numbers:
            time_key = "time_" + str(staff)
            interpolator_key = "interp_" + str(staff)
            start_times = np.array(clef_dict[time_key])
            clef_indices = np.arange(len(start_times))
            interpolator = interp1d(
                start_times,
                clef_indices,
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )
            clef_dict[interpolator_key].append(interpolator)

        W = np.zeros((len(notes), 3))
        for i, na_n in enumerate(na):
            n = notes[na_n["id"]]
            staff = n.staff or 1
            time = n.start.t
            clef_key = "clef_" + str(staff)
            interpolator_key = "interp_" + str(staff)
            clef_idx = clef_dict[interpolator_key][0](time)
            clef = clef_dict[clef_key][int(clef_idx)]
            sign = clef.sign or "none"
            W[i, 0] = clef_sign_to_int(sign)
            W[i, 1] = clef.line or 0
            W[i, 2] = clef.octave_change or 0

    else:
        # add dummy clef
        W = np.zeros((len(notes), 3))
        W[:, 0] = 6  # "none"
        W[:, 1] = 0
        W[:, 2] = 0

    return W, names


def loudness_direction_feature(na, part, **kwargs):
    """The loudness directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'decrescendo' and 'diminuendo' are encoded
    together in a descriptor 'loudness_decr'. The descriptor names of
    textual directions such as 'adagio' are the verbatim directions.

    Some possible descriptors:
    * p : piano
    * f : forte
    * pp : pianissimo
    * loudness_incr : crescendo direction
    * loudness_decr : decrescendo or diminuendo direction

    """

    onsets = na["onset_div"]
    N = len(onsets)
    constant = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "unknown_constant"]
    impulsive = ["fp", "sf", "sfp", "sfz", "unknown_impulsive"]
    names = constant + impulsive + ["loudness_incr", "loudness_decr"]

    directions = list(part.iter_all(score.LoudnessDirection, include_subclasses=True))
    if "include_empty_features" in kwargs.keys():
        force_size = kwargs["include_empty_features"]
    else:
        force_size = False
    if force_size:

        def to_name(d):
            if isinstance(d, score.ConstantLoudnessDirection):
                if d.text in constant:
                    return d.text
                else:
                    return "unknown_constant"
            elif isinstance(d, score.ImpulsiveLoudnessDirection):
                if d.text in impulsive:
                    return d.text
                else:
                    return "unknown_impulsive"
            elif isinstance(d, score.IncreasingLoudnessDirection):
                return "loudness_incr"
            elif isinstance(d, score.DecreasingLoudnessDirection):
                return "loudness_decr"

    else:

        def to_name(d):
            if isinstance(d, score.ConstantLoudnessDirection):
                return d.text
            elif isinstance(d, score.ImpulsiveLoudnessDirection):
                return d.text
            elif isinstance(d, score.IncreasingLoudnessDirection):
                return "loudness_incr"
            elif isinstance(d, score.DecreasingLoudnessDirection):
                return "loudness_decr"

    feature_by_name = {}
    for d in directions:
        j, bf = feature_by_name.setdefault(
            to_name(d), (len(feature_by_name), np.zeros(N))
        )
        bf += feature_function_activation(d)(onsets)

    if not force_size:
        M = len(feature_by_name) if len(feature_by_name) > 0 else 1
        names = [None] * M
    W = np.zeros((len(onsets), len(names)))
    for name, (j, bf) in feature_by_name.items():
        if force_size:
            j = names.index(name)
        else:
            names[j] = name
        W[:, j] = bf

    return W, names


def tempo_direction_feature(na, part, **kwargs):
    """The tempo directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'adagio' and 'molto adagio' are encoded
    together in a descriptor 'adagio'.

    Some possible descriptors:
    * adagio : directions like 'adagio', 'molto adagio'

    """
    onsets = na["onset_div"]
    N = len(onsets)
    constant = [
        "adagio",
        "largo",
        "lento",
        "grave",
        "larghetto",
        "adagietto",
        "andante",
        "andantino",
        "moderato",
        "allegretto",
        "allegro",
        "vivace",
        "presto",
        "prestissimo",
        "unknown_constant",
    ]
    names = constant + ["tempo_incr", "tempo_decr"]
    directions = list(part.iter_all(score.TempoDirection, include_subclasses=True))

    if "include_empty_features" in kwargs.keys():
        force_size = kwargs["include_empty_features"]
    else:
        force_size = False
    if force_size:

        def to_name(d):
            if isinstance(d, score.ResetTempoDirection):
                ref = d.reference_tempo
                if ref:
                    if ref.text in constant:
                        return ref.text
                    else:
                        return "unknown_constant"
                else:
                    if d.text in constant:
                        return d.text
                    else:
                        return "unknown_constant"
            elif isinstance(d, score.ConstantTempoDirection):
                if d.text in constant:
                    return d.text
                else:
                    return "unknown_constant"
            elif isinstance(d, score.IncreasingTempoDirection):
                return "tempo_incr"
            elif isinstance(d, score.DecreasingTempoDirection):
                return "tempo_decr"

    else:

        def to_name(d):
            if isinstance(d, score.ResetTempoDirection):
                ref = d.reference_tempo
                if ref:
                    return ref.text
                else:
                    return d.text
            elif isinstance(d, score.ConstantTempoDirection):
                return d.text
            elif isinstance(d, score.IncreasingTempoDirection):
                return "tempo_incr"
            elif isinstance(d, score.DecreasingTempoDirection):
                return "tempo_decr"

    feature_by_name = {}
    for d in directions:
        j, bf = feature_by_name.setdefault(
            to_name(d), (len(feature_by_name), np.zeros(N))
        )
        bf += feature_function_activation(d)(onsets)

    if not force_size:
        M = len(feature_by_name) if len(feature_by_name) > 0 else 1
        names = [None] * M
    W = np.zeros((len(onsets), len(names)))
    for name, (j, bf) in feature_by_name.items():
        if force_size:
            j = names.index(name)
        else:
            names[j] = name
        W[:, j] = bf

    return W, names


def articulation_direction_feature(na, part, **kwargs):
    """ """
    onsets = na["onset_div"]
    N = len(onsets)

    directions = list(
        part.iter_all(score.ArticulationDirection, include_subclasses=True)
    )
    constant_names = ["staccato", "tenuto", "accent", "marcato", "unknown_articulation"]

    if "include_empty_features" in kwargs.keys():
        force_size = kwargs["include_empty_features"]
    else:
        force_size = False
    if force_size:

        def to_name(d):
            if d.text in constant_names:
                return d.text
            else:
                return "unknown_articulation"

    else:

        def to_name(d):
            return d.text

    feature_by_name = {}

    for d in directions:
        j, bf = feature_by_name.setdefault(
            to_name(d), (len(feature_by_name), np.zeros(N))
        )
        bf += feature_function_activation(d)(onsets)

    if force_size:
        W = np.zeros((len(onsets), len(constant_names)))
        names = constant_names
    else:
        M = len(feature_by_name) if len(feature_by_name) > 0 else 1
        W = np.zeros((len(onsets), M))
        names = [None] * M

    for name, (j, bf) in feature_by_name.items():
        if force_size:
            j = names.index(name)
        else:
            names[j] = name
        W[:, j] = bf

    return W, names


def feature_function_activation(direction):
    epsilon = 1e-6

    if isinstance(
        direction, (score.DynamicLoudnessDirection, score.DynamicTempoDirection)
    ):
        # a dynamic direction will be encoded as a ramp from d.start.t to d.end.t
        # if d.end is None (e.g. just a ritardando without dashes)
        if direction.end:
            direction_end = direction.end.t
        else:
            # assume the end of d is the end of the measure:
            measure = next(direction.start.iter_prev(score.Measure, eq=True), None)
            if measure:
                direction_end = measure.start.t
            else:
                # no measure, unlikely, but not impossible.
                direction_end = direction.start.t + 1

        x = [direction.start.t, direction_end, direction_end + epsilon]
        y = [0, 1, 0]

    elif isinstance(
        direction,
        (
            score.ConstantLoudnessDirection,
            score.ConstantArticulationDirection,
            score.ConstantTempoDirection,
        ),
    ):
        x = [
            direction.start.t - epsilon,
            direction.start.t,
            direction.end.t - epsilon,
            direction.end.t,
        ]
        y = [0, 1, 1, 0]

    else:  # impulsive
        x = [
            direction.start.t - epsilon,
            direction.start.t,
            direction.start.t + epsilon,
        ]
        y = [0, 1, 0]

    return interp1d(x, y, bounds_error=False, fill_value=0)


def slur_feature(na, part, **kwargs):
    """Slur feature.

    Returns:
    * slur_incr : a ramp function that increases from 0
                  to 1 over the course of the slur
    * slur_decr : a ramp function that decreases from 1
                  to 0 over the course of the slur

    """
    names = ["slur_incr", "slur_decr"]
    onsets = na["onset_div"]
    slurs = part.iter_all(score.Slur)
    W = np.zeros((len(onsets), 2))

    for slur in slurs:
        if not slur.end:
            continue
        x = [slur.start.t, slur.end.t]
        y_inc = [0, 1]
        y_dec = [1, 0]
        W[:, 0] += interp1d(x, y_inc, bounds_error=False, fill_value=0)(onsets)
        W[:, 1] += interp1d(x, y_dec, bounds_error=False, fill_value=0)(onsets)
    # Filter out NaN values
    W[np.isnan(W)] = 0.0
    return W, names


def articulation_feature(na, part, **kwargs):
    """Articulation feature.

    This feature returns articulation-related note annotations, such as accents, legato, and tenuto.

    Possible descriptors:
    * accent : 1 when the note has an annotated accent sign
    * legato : 1 when the note has an annotated legato sign
    * staccato : 1 when the note has an annotated staccato sign
    ...

    """
    names = [
        "accent",
        "strong-accent",
        "staccato",
        "tenuto",
        "detached-legato",
        "staccatissimo",
        "spiccato",
        "scoop",
        "plop",
        "doit",
        "falloff",
        "breath-mark",
        "caesura",
        "stress",
        "unstress",
        "soft-accent",
    ]
    if "include_empty_features" in kwargs:
        force_size = kwargs["include_empty_features"]
    else:
        force_size = False

    feature_by_name = {}
    notes = {n.id: n for n in part.notes_tied}
    N = len(notes)
    for i, na_n in enumerate(na):
        n = notes[na_n["id"]]
        if n.articulations:
            for art in n.articulations:
                if art in names:
                    j, bf = feature_by_name.setdefault(
                        art, (len(feature_by_name), np.zeros(N))
                    )
                    bf[i] = 1

    if force_size:
        M = len(names)
    else:
        M = len(feature_by_name) if len(feature_by_name) > 0 else 1
        names = [None] * M
    W = np.zeros((N, M))

    for name, (j, bf) in feature_by_name.items():
        if force_size:
            j = names.index(name)
        else:
            names[j] = name
        W[:, j] = bf

    return W, names


def ornament_feature(na, part, **kwargs):
    """Ornament feature.

    This feature returns ornamentation note annotations,such as trills.

    Possible descriptors:
    * trill : 1 when the note has an annotated trill
    * mordent : 1 when the note has an annotated mordent
    ...

    """
    names = [
        "trill-mark",
        "turn",
        "delayed-turn",
        "inverted-turn",
        "delayed-inverted-turn",
        "vertical-turn",
        "inverted-vertical-turn",
        "shake",
        "wavy-line",
        "mordent",
        "inverted-mordent",
        "schleifer",
        "tremolo",
        "haydn",
        "other-ornament",
    ]
    feature_by_name = {}
    notes = {n.id: n for n in part.notes_tied}
    N = len(notes)
    for i, na_n in enumerate(na):
        n = notes[na_n["id"]]
        if n.ornaments:
            for art in n.ornaments:
                if art in names:
                    j, bf = feature_by_name.setdefault(
                        art, (len(feature_by_name), np.zeros(N))
                    )
                    bf[i] = 1
    if "include_empty_features" in kwargs.keys():
        fix_size = kwargs["include_empty_features"]
    else:
        fix_size = False
    if fix_size:
        M = len(names)
    else:
        M = len(feature_by_name) if len(feature_by_name) > 0 else 1
        names = [None] * M
    W = np.zeros((N, M))

    for name, (j, bf) in feature_by_name.items():
        if fix_size:
            j = names.index(name)
        else:
            names[j] = name
        W[:, j] = bf

    return W, names


def staff_feature(na, part, **kwargs):
    """Staff feature"""
    names = ["staff"]
    notes = {n.id: n.staff for n in part.notes_tied}
    N = len(na)
    W = np.zeros((N, 1))
    for i, n in enumerate(na):
        W[i, 0] = notes[n["id"]] if n["id"] not in (None, "None", "") else 0
    return W, names


# # for a subset of the articulations do e.g.
# def staccato_feature(part):
#     W, names = articulation_feature(part)
#     if 'staccato' in names:
#         i = names.index('staccato')
#         return W[:, i:i + 1], ['staccato']
#     else:
#         return np.empty(len(W)), []


def fermata_feature(na, part, **kwargs):
    """Fermata feature.

    Returns:
    * fermata : 1 when the note coincides with a fermata sign.

    """
    names = ["fermata"]
    onsets = na["onset_div"]
    W = np.zeros((len(onsets), 1))
    for ferm in part.iter_all(score.Fermata):
        W[onsets == ferm.start.t, 0] = 1
    return W, names


def metrical_feature(na, part, **kwargs):
    """Metrical feature

    This feature encodes the metrical position in the bar. For example
    the first beat in a 3/4 meter is encoded in a binary descriptor
    'metrical_3_4_0', the fifth beat in a 6/8 meter as
    'metrical_6_8_4', etc. Any positions that do not fall on a beat
    are encoded in a feature suffixed '_weak'. For example a note
    starting on the second 8th note in a bar of 4/4 meter will have a
    non-zero value in the 'metrical_4_4_weak' descriptor.

    """
    notes_list = part.notes_tied if not np.all(na["pitch"] == 0) else part.rests
    notes = {n.id: n for n in notes_list}
    ts_map = part.time_signature_map
    bm = part.beat_map
    feature_by_name = {}
    eps = 10**-6

    for i, na_n in enumerate(na):
        n = notes[na_n["id"]]
        beats, beat_type, mus_beats = ts_map(n.start.t).astype(int)
        measure = next(n.start.iter_prev(score.Measure, eq=True), None)

        if measure:
            measure_start = measure.start.t
        else:
            measure_start = 0

        pos = bm(n.start.t) - bm(measure_start)

        if pos % 1 < eps:
            name = "metrical_{}_{}_{}".format(beats, beat_type, int(pos))
        else:
            name = "metrical_{}_{}_weak".format(beats, beat_type)

        j, bf = feature_by_name.setdefault(
            name, (len(feature_by_name), np.zeros(len(notes)))
        )
        bf[i] = 1

    W = np.zeros((len(notes), len(feature_by_name)))
    names = [None] * len(feature_by_name)
    for name, (j, bf) in feature_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def metrical_strength_feature(na, part, **kwargs):
    """Metrical strength feature

    This feature encodes the beat phase (relative position of a note within
    the measure), as well as metrical strength of common time signatures.

    'beat_phase' encodes the position in the measure as value between 0.0 and 1.0
    'metrical_strength_downbeat' is 1.0 on downbeats, 0.0 elsewhere
    'metrical_strength_secondary' is 1.0 on measure midpoint, 0.0 elsewhere,
        not valid for triple meters
    'metrical_strength_weak' is 1.0 where both others are 0.0, 0.0 elsewhere
    """
    names = [
        "beat_phase",
        "metrical_strength_downbeat",
        "metrical_strength_secondary",
        "metrical_strength_weak",
    ]

    relod = na["rel_onset_div"].astype(float)
    totmd = na["tot_measure_div"].astype(float)
    W = np.zeros((len(na), len(names)))
    W[:, 0] = np.divide(relod, totmd)  # Onset Phase
    W[:, 1] = na["is_downbeat"].astype(float)
    W[:, 2][W[:, 0] == 0.5] = 1.00
    W[:, 3][W[:, 1] == W[:, 2]] = 1.00

    return W, names


def measure_feature(na, part, **kwargs):
    """Measure feature

    This feature encodes the measure each note is in.

    """
    notes_list = part.notes_tied if not np.all(na["pitch"] == 0) else part.rests
    notes = {n.id: n for n in notes_list}
    bm = part.beat_map

    global_start = bm(part.first_point.t)
    global_end = bm(part.last_point.t)
    global_number = 0  # default global measure number

    names = [
        "measure_number",
        "measure_start_beat",
        "measure_end_beat",
    ]
    W = np.zeros((len(notes), 3))

    for i, na_n in enumerate(na):
        n = notes[na_n["id"]]
        measure = next(n.start.iter_prev(score.Measure, eq=True), None)

        if measure:
            start = bm(measure.start.t)
            end = bm(measure.end.t)
            number = measure.number
        else:
            start = global_start
            end = global_end
            number = global_number

        W[i, 0] = number
        W[i, 1] = start
        W[i, 2] = end

    return W, names


def time_signature_feature(na, part, **kwargs):
    """TIme Signature feature
    This feature encodes the time signature of the note in two sets of one-hot vectors,
    a one hot encoding of number of beats and a one hot encoding of beat type
    """

    ts_map = part.time_signature_map
    possible_beats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "other"]
    possible_beat_types = [1, 2, 4, 8, 16, "other"]
    W_beats = np.zeros((len(na), len(possible_beats)))
    W_types = np.zeros((len(na), len(possible_beat_types)))

    names = ["time_signature_num_{0}".format(b) for b in possible_beats] + [
        "time_signature_den_{0}".format(b) for b in possible_beat_types
    ]

    for i, n in enumerate(na):
        beats, beat_type, mus_beats = ts_map(n["onset_div"]).astype(int)

        if beats in possible_beats:
            W_beats[i, beats - 1] = 1
        else:
            W_beats[i, -1] = 1

        if beat_type in possible_beat_types:
            W_types[i, possible_beat_types.index(beat_type)] = 1
        else:
            W_types[i, -1] = 1

    W = np.column_stack((W_beats, W_types))

    return W, names


def vertical_neighbor_feature(na, part, **kwargs):
    """Vertical neighbor feature.

    Describes various aspects of simultaneously starting notes.

    Returns:
    * n_total :
    * n_above :
    * n_below :
    * highest_pitch :
    * lowest_pitch :
    * pitch_range :

    """
    # the list of descriptors
    names = [
        "n_total",
        "n_above",
        "n_below",
        "highest_pitch",
        "lowest_pitch",
        "pitch_range",
    ]
    W = np.zeros((len(na), len(names)))
    for i, n in enumerate(na):
        neighbors = na[np.where(na["onset_beat"] == n["onset_beat"])]["pitch"]
        max_pitch = np.max(neighbors)
        min_pitch = np.min(neighbors)
        W[i, 0] = len(neighbors) - 1
        W[i, 1] = np.sum(neighbors > n["pitch"])
        W[i, 2] = np.sum(neighbors < n["pitch"])
        W[i, 3] = max_pitch
        W[i, 4] = min_pitch
        W[i, 5] = max_pitch - min_pitch
    return W, names


def normalize(data, method="minmax"):
    """
    Normalize data in one of several ways.

    The available normalization methods are:

    * minmax
      Rescale `data` to the range `[0, 1]` by subtracting the minimum
      and dividing by the range. If `data` is a 2d array, each column is
      rescaled to `[0, 1]`.

    * tanh
      Rescale `data` to the interval `(-1, 1)` using `tanh`. Note that
      if `data` is non-negative, the output interval will be `[0, 1)`.

    * tanh_unity
      Like "soft", but rather than rescaling strictly to the range (-1,
      1), following will hold:

      normalized = normalize(data, method="tanh_unity")
      np.where(data==1) == np.where(normalized==1)

      That is, the normalized data will equal one wherever the original data
      equals one. The target interval is `(-1/np.tanh(1), 1/np.tanh(1))`.

    Parameters
    ----------
    data: ndarray
        Data to be normalized
    method: {'minmax', 'tanh', 'tanh_unity'}, optional
        The normalization method. Defaults to 'minmax'.

    Returns
    -------
    ndarray
        Normalized copy of the data
    """

    """Normalize the data in `data`. There are several normalization

    """
    if method == "minmax":
        vmin = np.min(data, 0)
        vmax = np.max(data, 0)

        if np.isclose(vmin, vmax):
            # Return all values as 0 or as 1?
            return np.zeros_like(data)
        else:
            return (data - vmin) / (vmax - vmin)
    elif method == "tanh":
        return np.tanh(data)
    elif method == "tanh_unity":
        return np.tanh(data) / np.tanh(1)
