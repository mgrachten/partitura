#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implement a series of mid-level descriptors of the performance expressions.
Built upon the low-level basis functions from the performance codec.
"""

import sys
import types
from typing import Union, List
import warnings
import numpy as np
from scipy.signal import find_peaks
import numpy.lib.recfunctions as rfn
from partitura.score import ScoreLike
from partitura.performance import PerformanceLike, PerformedPart
from partitura.utils.generic import interp1d
from partitura.musicanalysis.performance_codec import (
    to_matched_score,
    onsetwise_to_notewise,
    encode_tempo,
)


__all__ = [
    "make_performance_features",
]

# ordinal
OLS = ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


class InvalidPerformanceFeatureException(Exception):
    pass


def make_performance_features(
    score: ScoreLike,
    performance: PerformanceLike,
    alignment: list,
    feature_functions: Union[List, str],
    add_idx: bool = True,
):
    """
    Compute the performance features. This function is defined in the same
    style of note_features.make_note_features

    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, can be a part, score
    performance : partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    feature_functions : list or str
        A list of performance feature functions. Elements of the list can be either
        the functions themselves or the names of a feature function as
        strings (or a mix).
        currently implemented:
        asynchrony_feature, articulation_feature, dynamics_feature, pedal_feature
    add_idx: bool
        add score note idx column to feature array

    Returns
    -------
    performance_features : structured array
    """
    m_score, unique_onset_idxs, snote_ids = compute_matched_score(
        score, performance, alignment
    )

    acc = []
    if isinstance(feature_functions, str) and feature_functions == "all":
        feature_functions = list_performance_feats_functions()
    elif not isinstance(feature_functions, list):
        raise TypeError(
            "feature_functions variable {} needs to be list or 'all'".format(
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
            warnings.warn("Ignoring unknown performance feature function {}".format(bf))
        features = func(m_score, unique_onset_idxs, performance)
        # check if the size and number of the feature function are correct
        if features.size != 0:
            n_notes = len(m_score)
            if len(features) != n_notes:
                msg = (
                    "length of feature {} does not equal "
                    "number of notes {}".format(len(features), n_notes)
                )
                raise InvalidPerformanceFeatureException(msg)

            features_ = rfn.structured_to_unstructured(features)
            if np.any(np.logical_or(np.isnan(features_), np.isinf(features_))):
                problematic = np.unique(
                    np.where(np.logical_or(np.isnan(features_), np.isinf(features_)))[1]
                )
                msg = "NaNs or Infs found in the following feature: {} ".format(
                    ", ".join(np.array(features.dtype.names)[problematic])
                )
                raise InvalidPerformanceFeatureException(msg)

            # prefix feature names by function name
            feature_names = [
                "{}.{}".format(func.__name__, n) for n in features.dtype.names
            ]
            features = rfn.rename_fields(
                features, dict(zip(features.dtype.names, feature_names))
            )

            acc.append(features)

    if add_idx:
        acc.append(np.array([(idx) for idx in snote_ids], dtype=[("id", "U256")]))

    performance_features = rfn.merge_arrays(acc, flatten=True, usemask=False)
    full_performance_features = rfn.join_by("id", performance_features, m_score)
    full_performance_features = full_performance_features.data

    sort_idx = np.lexsort(
        (
            full_performance_features["duration"],
            full_performance_features["pitch"],
            full_performance_features["onset"],
        )
    )
    full_performance_features = full_performance_features[sort_idx]
    return full_performance_features


def compute_matched_score(
    score: ScoreLike,
    performance: PerformanceLike,
    alignment: list,
):
    """
    Compute the matched score and add the score features

    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, can be a part, score
    performance : partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    alignment : list
        The score--performance alignment, a list of dictionaries

    Returns
    -------
    m_score : np strutured array
    unique_onset_idxs : list
    """

    m_score, snote_ids = to_matched_score(
        score, performance, alignment, include_score_markings=True
    )
    (time_params, unique_onset_idxs) = encode_tempo(
        score_onsets=m_score["onset"],
        performed_onsets=m_score["p_onset"],
        score_durations=m_score["duration"],
        performed_durations=m_score["p_duration"],
        return_u_onset_idx=True,
        tempo_smooth="average",
    )
    m_score = rfn.append_fields(
        m_score,
        ["beat_period", "timing", "articulation_log", "id"],
        [
            time_params["beat_period"],
            time_params["timing"],
            time_params["articulation_log"],
            snote_ids,
        ],
        ["f4", "f4", "f4", "U256"],
        usemask=False,
    )
    return m_score, unique_onset_idxs, snote_ids


def list_performance_feats_functions():
    """Return a list of all feature function names defined in this module.

    The feature function names listed here can be specified by name in
    the `make_performance_features` function. For example:

    >>> feature, names = make_performance_features(score,
                                                    performance,
                                                    alignment,
                                                    ['asynchrony_feature'])

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


def print_performance_feats_functions():
    """Print a list of all feature function names defined in this module,
    with descriptions where available.

    """
    module = sys.modules[__name__]
    doc_indent = 4
    for name in list_performance_feats_functions():
        print("* {}".format(name))
        member = getattr(sys.modules[__name__], name)
        if member.__doc__:
            print(
                " " * doc_indent + member.__doc__.replace("\n", " " * doc_indent + "\n")
            )


# alias
list_performance_feature_functions = list_performance_feats_functions
print_performance_feature_functions = print_performance_feats_functions

### Asynchrony


def asynchrony_feature(
    m_score: np.ndarray, unique_onset_idxs: list, performance: PerformanceLike
):
    """
    Compute the asynchrony attributes from the alignment.

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings.
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object

    Returns
    -------
    async_ : structured array
        structured array (broadcasted to the note level) with the following fields
            delta [0, 1]: the largest time difference (in seconds) between onsets in this group
            pitch_cor [-1, 1]: correlation between timing and pitch, min-scaling
            vel_cor [-1, 1]: correlation between timing and velocity, min-scaling
            voice_std [0, 1]: std of the avg timing (in seconds) of each voice in this group
    """

    async_ = np.zeros(
        len(unique_onset_idxs),
        dtype=[
            ("delta", "f4"),
            ("pitch_cor", "f4"),
            ("vel_cor", "f4"),
            ("voice_std", "f4"),
        ],
    )
    for i, onset_idxs in enumerate(unique_onset_idxs):
        note_group = m_score[onset_idxs]

        onset_times = note_group["p_onset"]
        delta = min(onset_times.max() - onset_times.min(), 1)
        async_[i]["delta"] = delta

        midi_pitch = note_group["pitch"]
        midi_pitch = midi_pitch - midi_pitch.min()  # min-scaling
        onset_times = onset_times - onset_times.min()
        cor = (
            (-1) * np.corrcoef(midi_pitch, onset_times)[0, 1]
            if (len(midi_pitch) > 1 and sum(midi_pitch) != 0 and sum(onset_times) != 0)
            else 0
        )
        # cor=nan if there is only one note in the group
        async_[i]["pitch_cor"] = cor

        assert not np.isnan(cor)

        midi_vel = note_group["velocity"].astype(float)
        midi_vel = midi_vel - midi_vel.min()
        cor = (
            (-1) * np.corrcoef(midi_vel, onset_times)[0, 1]
            if (sum(midi_vel) != 0 and sum(onset_times) != 0)
            else 0
        )
        async_[i]["vel_cor"] = cor

        assert not np.isnan(cor)

        voices = np.unique(note_group["voice"])
        voices_onsets = []
        for voice in voices:
            note_in_voice = note_group[note_group["voice"] == voice]
            voices_onsets.append(note_in_voice["p_onset"].mean())
        async_[i]["voice_std"] = min(np.std(np.array(voices_onsets)), 1)

    return onsetwise_to_notewise(async_, unique_onset_idxs)


### Dynamics


### Articulation


def articulation_feature(
    m_score: np.ndarray,
    unique_onset_idxs: list,
    performance: PerformanceLike,
    return_mask=False,
):
    """
    Compute the articulation attributes (key overlap ratio) from the alignment.
    Key overlap ratio is the ratio between key overlap time (KOT) and IOI, result in a value between (-1, inf)
        -1 is the dummy value. For normalization purposes we empirically cap the maximum to 5.

    References
    ----------
    ..  [1] B.Repp: Acoustics, Perception, and Production of Legato Articulation on a Digital Piano

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings.
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object
    return_mask : bool
        if true, return a boolean mask of legato notes, staccato notes and repeated notes

    Returns
    -------
    kor_ : structured array (1, n_notes)
        structured array on the note level with fields kor (-1, 5]
    """

    m_score = rfn.append_fields(
        m_score, "offset", m_score["onset"] + m_score["duration"], usemask=False
    )
    m_score = rfn.append_fields(
        m_score, "p_offset", m_score["p_onset"] + m_score["p_duration"], usemask=False
    )

    kor_ = np.full(len(m_score), -1, dtype=[("kor", "f4")])
    if return_mask:
        mask = np.full(
            len(m_score),
            False,
            dtype=[("legato", "?"), ("staccato", "?"), ("repeated", "?")],
        )

    # consider the note transition by each voice
    for voice in np.unique(m_score["voice"]):
        match_voiced = m_score[m_score["voice"] == voice]
        for _, note_info in enumerate(match_voiced):
            if note_info["onset"] == match_voiced["onset"].max():  # last beat
                break
            next_note_info = get_next_note(
                note_info, match_voiced
            )  # find most plausible transition

            if next_note_info:  # in some cases no meaningful transition
                j = np.where(m_score == note_info)[0].item()  # original position

                if note_info["offset"] == next_note_info["onset"]:
                    kor_[j]["kor"] = get_kor(note_info, next_note_info)

                if return_mask:  # return the
                    if (note_info["slur_feature.slur_incr"] > 0) or (
                        note_info["slur_feature.slur_decr"] > 0
                    ):
                        mask[j]["legato"] = True

                    if note_info["articulation"] == "staccato":
                        mask[j]["staccato"] = True

                    # KOR for repeated notes
                    if note_info["pitch"] == next_note_info["pitch"]:
                        mask[j]["repeated"] = True

    if return_mask:
        return kor_, mask
    else:
        return kor_


def get_kor(e1, e2):
    """
    calculate the ratio between key overlap time and IOI.
    In the case of a negative IOI (the order of notes in performance is reversed from the score),
    set at default 0.
    is bounded within the interval [-1,5]

    Parameters
    ----------
    e1 : np.ndarray
        the m_score row of first note
    e2 : np.ndarray
        the m_score of second note

    Returns
    -------
    kor : float
        Key overlap ratio

    """
    kot = e1["p_offset"] - e2["p_onset"]
    ioi = e2["p_onset"] - e1["p_onset"]

    if ioi <= 0:
        kor = 0

    kor = kot / ioi

    return min(kor, 5)


def get_next_note(note_info, match_voiced):
    """
    get the next note in the same voice that's a reasonable transition

    Parameters
    ----------
    note_info : np.ndarray
        the row of current note
    match_voiced : np.ndarray
        all notes in the same voice

    Returns
    -------
    next_position : np.ndarray
        the next note
    """

    next_position = min(o for o in match_voiced["onset"] if o > note_info["onset"])

    # if the next note is not immediate successor of the previous one...
    if next_position != note_info["onset"] + note_info["duration"]:
        return None

    next_position_notes = match_voiced[match_voiced["onset"] == next_position]

    # from the notes in the next position, find the one that's closest pitch-wise.
    closest_idx = np.abs((next_position_notes["pitch"] - note_info["pitch"])).argmin()

    return next_position_notes[closest_idx]


### Pedals


def pedal_feature(m_score: list, unique_onset_idxs: list, performance: PerformanceLike):
    """
    Compute the pedal features.

    Parameters
    ----------
    m_score : list
        correspondance between score and performance notes, with score markings.
    unique_onset_idxs : list
        a list of arrays with the note indexes that have the same onset
    performance: PerformedPart
        The original PerformedPart object

    Returns
    -------
    pedal_ : structured array (4, n_notes) with fields
        onset_value [0, 127]: The interpolated pedal value at the onset
        offset_value [0, 127]: The interpolated pedal value at the key offset
        to_prev_release [0, 10]: delta time from note onset to the previous pedal release 'peak'
        to_next_release [0, 10]: delta time from note offset to the next pedal release 'peak'
        (think about something relates to the real duration)
    """

    onset_offset_pedals, ramp_func = pedal_ramp(performance.performedparts[0], m_score)

    x = np.linspace(0, 100, 200)
    y = ramp_func(x)

    peaks, _ = find_peaks(-y, prominence=10)
    peak_timepoints = x[peaks]

    release_times = np.zeros(
        len(m_score), dtype=[("to_prev_release", "f4"), ("to_next_release", "f4")]
    )
    for i, note in enumerate(m_score):
        peaks_before = peak_timepoints[note["p_onset"] >= peak_timepoints]
        peaks_after = peak_timepoints[
            (note["p_onset"] + note["p_duration"]) <= peak_timepoints
        ]
        if len(peaks_before):
            release_times[i]["to_prev_release"] = min(
                note["p_onset"] - peaks_before.max(), 10
            )
        if len(peaks_after):
            release_times[i]["to_next_release"] = min(
                peaks_after.min() - (note["p_onset"] + note["p_duration"]), 10
            )

    return rfn.merge_arrays(
        [onset_offset_pedals, release_times], flatten=True, usemask=False
    )


def pedal_ramp(ppart: PerformedPart, m_score: np.ndarray):
    """Pedal ramp in the same shape as the m_score.

    Returns:
    * pramp : a ramp function that ranges from 0
                  to 127 with the change of sustain pedal
    """
    pedal_controls = ppart.controls
    W = np.zeros((len(m_score), 2))
    onset_timepoints = m_score["p_onset"]
    offset_timepoints = m_score["p_onset"] + m_score["p_duration"]

    timepoints = [control["time"] for control in pedal_controls]
    values = [control["value"] for control in pedal_controls]

    if len(timepoints) <= 1:  # the case there is no pedal
        timepoints, values = [0, 0], [0, 0]

    agg_ramp_func = interp1d(timepoints, values, bounds_error=False, fill_value=0)
    W[:, 0] = agg_ramp_func(onset_timepoints)
    W[:, 1] = agg_ramp_func(offset_timepoints)

    # Filter out NaN values
    W[np.isnan(W)] = 0.0

    return (
        np.array(
            [tuple(i) for i in W], dtype=[("onset_value", "f4"), ("offset_value", "f4")]
        ),
        agg_ramp_func,
    )


### Phrasing

### Tempo
