#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implements a codec to encode and decode expressive performances to a set of
expressive parameters.
"""
from typing import Union, Callable
import numpy as np
import numpy.lib.recfunctions as rfn
import warnings


from partitura.score import Part, ScoreLike
from partitura.performance import PerformedPart, PerformanceLike
from partitura.musicanalysis import note_features
from partitura.utils.misc import deprecated_alias
from partitura.utils.generic import interp1d, monotonize_times, first_order_derivative
from partitura.utils.music import ensure_notearray


__all__ = ["encode_performance", "decode_performance", "to_matched_score"]


#### Full Codecs ####


@deprecated_alias(part="score", ppart="performance")
def encode_performance(
    score: ScoreLike,
    performance: PerformanceLike,
    alignment: list,
    return_u_onset_idx=False,
    beat_normalization: str = "beat_period",  # "beat_period_log", "beat_period_ratio", "beat_period_ratio_log", "beat_period_standardized"
    tempo_smooth: Union[str, Callable] = "average",
):
    """
    Encode expressive parameters from a matched performance


    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, can be a part, score
    performance : partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    return_u_onset_idx : bool
        Return the indices of the unique score onsets
    beat_normalization : str (Optional)
        Return extra columns for normalization parameters.
    tempo_smooth : str (Optional)
        How the tempo curve is computed. average or derivative.
        Can also input a callable function for user-defined tempo curve.

    Returns
    -------
    parameters : structured array
        A performance array with 4 fields: beat_period, velocity,
        timing, and articulation_log.
        If beat_normalization is defined as any method other than beat_period,
        return the normalization value as extra columns in parameters.
    snote_ids : dict
        A dict of snote_ids corresponding to performance notes.
    unique_onset_idxs : list (optional)
        List of unique onset ids. Returned only when return_u_onset_idx
        is set to True.
    """

    m_score, snote_ids = to_matched_score(score, performance, alignment)

    # Get time-related parameters
    (time_params, unique_onset_idxs) = encode_tempo(
        score_onsets=m_score["onset"],
        performed_onsets=m_score["p_onset"],
        score_durations=m_score["duration"],
        performed_durations=m_score["p_duration"],
        return_u_onset_idx=True,
        beat_normalization=beat_normalization,
        tempo_smooth=tempo_smooth,
    )

    # Get dynamics-related parameters
    dynamics_params = np.array(m_score["velocity"] / 127.0, dtype=[("velocity", "f4")])

    # Fixing random error
    parameters = time_params
    parameters["velocity"] = dynamics_params["velocity"]

    if return_u_onset_idx:
        return parameters, snote_ids, unique_onset_idxs
    else:
        return parameters, snote_ids


@deprecated_alias(part="score")
def decode_performance(
    score: ScoreLike,
    performance_array: np.ndarray,
    snote_ids=None,
    part_id=None,
    part_name=None,
    return_alignment=False,
    beat_normalization: str = "beat_period",  # "beat_period_log", "beat_period_ratio", "beat_period_ratio_log", "beat_period_standardized"
    *args,
    **kwargs
) -> PerformedPart:
    """
    Given a Part (score) and a performance array return a PerformedPart.

    Parameters
    ----------
    score : partitura.score.ScoreLike
        Score information, could be part, Score
    performance_array : structured array
        A performed array related to the part.
    snote_ids : list
    part_id : str
    part_name : str
    return_alignment : bool
        True returns alignment list of dicts.

    Returns
    -------
    ppart : partitura.performance.PerformedPart
        A partitura PerformedPart.
    alignment: list (optional)
        A list of dicts for the alignment.
    """

    snotes = score.note_array()

    if snote_ids is None:
        snote_ids = [n["id"] for n in snotes]
        snote_info = snotes
    else:
        snote_info = snotes[np.isin(snotes["id"], snote_ids)]

    # sort
    sort_idx = np.lexsort((snote_info["pitch"], snote_info["onset_div"]))

    onsets = snote_info["onset_beat"][sort_idx]
    durations = snote_info["duration_beat"][sort_idx]
    pitches = snote_info["pitch"][sort_idx]

    pitches = np.clip(pitches, 1, 127)

    dynamics_params = performance_array["velocity"][sort_idx]
    if beat_normalization != "beat_period":
        norm_params = list(TEMPO_NORMALIZATION[beat_normalization]["param_names"])
    else:
        norm_params = []
    time_params = performance_array[
        list(("beat_period", "timing", "articulation_log")) + norm_params
    ][sort_idx]

    onsets_durations = decode_time(
        score_onsets=onsets,
        score_durations=durations,
        parameters=time_params,
        normalization=beat_normalization,
        *args,
        **kwargs
    )

    velocities = np.round(dynamics_params * 127.0)

    velocities = np.clip(velocities, 1, 127)

    notes = []
    for nid, (onset, duration), velocity, pitch in zip(
        snote_ids, onsets_durations, velocities, pitches
    ):
        notes.append(
            dict(
                id=nid,
                midi_pitch=int(pitch),
                note_on=onset,
                note_off=onset + duration,
                sound_off=onset + duration,
                velocity=int(velocity),
            )
        )
    # * rescale according to default values?
    ppart = PerformedPart(id=part_id, part_name=part_name, notes=notes)

    if return_alignment:
        alignment = []
        for snote, pnote in zip(snote_info, ppart.notes):
            alignment.append(
                dict(label="match", score_id=snote["id"], performance_id=pnote["id"])
            )

        return ppart, alignment
    else:
        return ppart


#### Time and Articulation Codecs ####


def decode_time(
    score_onsets,
    score_durations,
    parameters,
    normalization="beat_period",
    *args,
    **kwargs
):
    """
    Decode a performance into onsets and durations in seconds
    for each note in the score.
    """
    score_onsets = score_onsets.astype(float, copy=False)
    score_durations = score_durations.astype(float, copy=False)

    score_info = get_unique_seq(
        onsets=score_onsets,
        offsets=score_onsets + score_durations,
        unique_onset_idxs=None,
        return_diff=True,
    )
    unique_onset_idxs = score_info["unique_onset_idxs"]
    diff_u_onset_score = score_info["diff_u_onset"]

    # reconstruct the time by the extra parameters, for testing the inversion.
    # In practice, always reconstruct the time by beat_period.
    if normalization != "beat_period":
        tempo_param_names = list(TEMPO_NORMALIZATION[normalization]["param_names"])
        time_param = np.array(
            [
                tuple(
                    np.mean(
                        rfn.structured_to_unstructured(
                            parameters[tempo_param_names][uix]
                        ),
                        axis=0,
                    ),
                )
                for uix in unique_onset_idxs
            ],
            dtype=[(tp, "f4") for tp in tempo_param_names],
        )
        beat_period = TEMPO_NORMALIZATION[normalization]["rescale"](time_param)

    else:
        time_param = np.array(
            [
                tuple([np.mean(parameters["beat_period"][uix])])
                for uix in unique_onset_idxs
            ],
            dtype=[("beat_period", "f4")],
        )
        beat_period = time_param["beat_period"]

    ioi_perf = diff_u_onset_score * beat_period

    eq_onset = np.cumsum(np.r_[0, ioi_perf])

    performance = np.zeros((len(score_onsets), 2))

    for i, jj in enumerate(unique_onset_idxs):
        # decode onset
        performance[jj, 0] = eq_onset[i] - parameters["timing"][jj]
        # decode duration
        performance[jj, 1] = decode_articulation(
            score_durations=score_durations[jj],
            articulation_parameter=parameters["articulation_log"][jj],
            beat_period=beat_period[i],
        )

    performance[:, 0] -= np.min(performance[:, 0])

    return performance


def encode_articulation(
    score_durations, performed_durations, unique_onset_idxs, beat_period
):
    """
    Encode articulation
    """
    articulation = np.zeros_like(score_durations)
    for idx, bp in zip(unique_onset_idxs, beat_period):
        sd = score_durations[idx]
        pd = performed_durations[idx]

        # indices of notes with duration 0 (grace notes)
        grace_mask = sd <= 0

        # Grace notes have an articulation ratio of 1
        sd[grace_mask] = 1
        pd[grace_mask] = bp
        articulation[idx] = np.log2(pd / (bp * sd))

    return articulation


def decode_articulation(score_durations, articulation_parameter, beat_period):
    """
    Decode articulation
    """
    art_ratio = 2**articulation_parameter
    dur = art_ratio * score_durations * beat_period

    return dur


def encode_tempo(
    score_onsets: np.ndarray,
    performed_onsets: np.ndarray,
    score_durations,
    performed_durations,
    return_u_onset_idx: bool = False,
    beat_normalization: str = "beat_period",  # "beat_period_log", "beat_period_ratio", "beat_period_ratio_log", "beat_period_standardized"
    tempo_smooth: Union[str, Callable] = "average",  # "average" or "derivative"
) -> np.ndarray:
    """
    Compute time-related performance parameters from a performance
    """
    if score_onsets.shape != performed_onsets.shape:
        raise ValueError("The performance and the score should be of " "the same size")

    # use float64, float32 led to problems that x == x + eps evaluated to True
    # Maybe replace by np.isclose
    score_onsets = score_onsets.astype(float, copy=False)
    performed_onsets = performed_onsets.astype(float, copy=False)
    score_durations = score_durations.astype(float, copy=False)
    performed_durations = performed_durations.astype(float, copy=False)
    score = np.column_stack((score_onsets, score_durations))
    performance = np.column_stack((performed_onsets, performed_durations))

    # Compute beat period
    if isinstance(tempo_smooth, Callable):
        beat_period, s_onsets, unique_onset_idxs = tempo_smooth(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True,
        )
    elif tempo_smooth == "average":
        beat_period, s_onsets, unique_onset_idxs = tempo_by_average(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True,
        )
    elif tempo_smooth == "derivative":
        beat_period, s_onsets, unique_onset_idxs = tempo_by_derivative(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True,
        )

    # Compute equivalent onsets
    eq_onsets = (
        np.cumsum(np.r_[0, beat_period[:-1] * np.diff(s_onsets)])
        + performance[unique_onset_idxs[0], 0].mean()
    )

    # Compute tempo parameter and normalize
    if beat_normalization != "beat_period":
        tempo_params = np.array(
            TEMPO_NORMALIZATION[beat_normalization]["scale"](beat_period)
        )
        tempo_param_names = list(TEMPO_NORMALIZATION[beat_normalization]["param_names"])

    # Compute articulation parameter
    articulation_param = encode_articulation(
        score_durations=score[:, 1],
        performed_durations=performance[:, 1],
        unique_onset_idxs=unique_onset_idxs,
        beat_period=beat_period,
    )

    # Initialize array of parameters
    parameter_names = ["beat_period", "velocity", "timing", "articulation_log"]
    if beat_normalization != "beat_period":
        parameter_names += tempo_param_names
    parameters = np.zeros(len(score), dtype=[(pn, "f4") for pn in parameter_names])
    parameters["articulation_log"] = articulation_param
    for i, jj in enumerate(unique_onset_idxs):
        parameters["beat_period"][jj] = beat_period[i]
        # Defined as in Eq. (3.9) in Thesis (pp. 34)
        parameters["timing"][jj] = eq_onsets[i] - performance[jj, 0]
        if beat_normalization != "beat_period":
            parameters[tempo_param_names][jj] = tuple(tempo_params[:, i])

    if return_u_onset_idx:
        return parameters, unique_onset_idxs
    else:
        return parameters


def tempo_by_average(
    score_onsets,
    performed_onsets,
    score_durations,
    performed_durations,
    unique_onset_idxs=None,
    input_onsets=None,
    return_onset_idxs=False,
):
    """
    Computes a tempo curve using the average of the onset times of all
    notes belonging to the same score onset.
    Parameters
    ----------
    score_onsets : np.ndarray
        Onset in beats of each note in the score.
    performed_onsets : np.ndarray
        Performed onsets in seconds of each note in the score.
    score_durations : np.ndarray
        Duration in beats of each note in the score.
    performed_durations : np.ndarray
        Performed duration in seconds of each note in the score.
    unique_onset_idxs : np.ndarray or None (optional)
        Indices of the notes with the same score onset. (By default is None,
        and is therefore, inferred from `score_onsets`).
    input_onsets : np.ndarray or None
        Input onset times in beats at which the tempo curve is to be
        sampled (by default is None, which means that the tempo curve
        is returned for each unique score onset)
    return_onset_idxs : bool
        Return the indices of the unique score onsets (Default is False)
    Returns
    -------
    tempo_curve : np.ndarray
        Tempo curve in seconds per beat (spb). If `input_onsets` was provided,
        this array contains the value of the tempo in spb for each onset
        in `input_onsets`. Otherwise, this array contains the value of the
        tempo in spb for each unique score onset.
    input_onsets : np.ndarray
        The score onsets corresponding to each value of the tempo curve.
    unique_onset_idxs: list
        Each element of the list is an array of the indices of the score
        corresponding to the elements in `tempo_curve`. Only returned if
        `return_onset_idxs` is True.
    """
    # use float64, float32 led to problems that x == x + eps evaluated
    # to True
    score_onsets = np.array(score_onsets).astype(float, copy=False)
    performed_onsets = np.array(performed_onsets).astype(float, copy=False)
    score_durations = np.array(score_durations).astype(float, copy=False)
    performed_durations = np.array(performed_durations).astype(float, copy=False)

    # Get unique onsets if no provided
    if unique_onset_idxs is None:
        # Get indices of the unique onsets (quantize score onsets)
        unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(int))

    # Get score information
    score_info = get_unique_seq(
        onsets=score_onsets,
        offsets=score_onsets + score_durations,
        unique_onset_idxs=unique_onset_idxs,
    )
    # Get performance information
    perf_info = get_unique_seq(
        onsets=performed_onsets,
        offsets=performed_onsets + performed_durations,
        unique_onset_idxs=unique_onset_idxs,
    )

    # unique score onsets
    unique_s_onsets = score_info["u_onset"]
    # equivalent onsets
    eq_onsets = perf_info["u_onset"]

    # Monotonize times
    eq_onset_mt, unique_s_onsets_mt = monotonize_times(eq_onsets, x=unique_s_onsets)

    # Estimate Beat Period
    perf_iois = np.diff(eq_onset_mt)
    s_iois = np.diff(unique_s_onsets_mt)
    beat_period = perf_iois / s_iois

    tempo_fun = interp1d(
        unique_s_onsets_mt[:-1],
        beat_period,
        kind="zero",
        bounds_error=False,
        fill_value=(beat_period[0], beat_period[-1]),
    )

    if input_onsets is None:
        input_onsets = unique_s_onsets[:-1]

    tempo_curve = tempo_fun(input_onsets)
    if not (tempo_curve >= 0).all():
        warnings.warn(
            "The estimated tempo curve is not always positive. "
            "This might be due to a bad alignment."
        )
    if return_onset_idxs:
        return tempo_curve, input_onsets, unique_onset_idxs
    else:
        return tempo_curve, input_onsets


def tempo_by_derivative(
    score_onsets,
    performed_onsets,
    score_durations,
    performed_durations,
    unique_onset_idxs=None,
    input_onsets=None,
    return_onset_idxs=False,
):
    """
    Computes a tempo curve using the derivative of the average performed
    onset times of all notes belonging to the same score onset with respect
    to that score onset. This results in a curve that is smoother than the
    tempo estimated using `tempo_by_average`.

    Parameters
    ----------
    score_onsets : np.ndarray
        Onset in beats of each note in the score.
    performed_onsets : np.ndarray
        Performed onsets in seconds of each note in the score.
    score_durations : np.ndarray
        Duration in beats of each note in the score.
    performed_durations : np.ndarray
        Performed duration in seconds of each note in the score.
    unique_onset_idxs : np.ndarray or None (optional)
        Indices of the notes with the same score onset. (By default is None,
        and is therefore, inferred from `score_onsets`).
    input_onsets : np.ndarray or None
        Input onset times in beats at which the tempo curve is to be
        sampled (by default is None, which means that the tempo curve
        is returned for each unique score onset)
    return_onset_idxs : bool
        Return the indices of the unique score onsets (Default is False)

    Returns
    -------
    tempo_curve : np.ndarray
        Tempo curve in seconds per beat (spb). If `input_onsets` was provided,
        this array contains the value of the tempo in spb for each onset
        in `input_onsets`. Otherwise, this array contains the value of the
        tempo in spb for each unique score onset.
    input_onsets : np.ndarray
        The score onsets corresponding to each value of the tempo curve.
    unique_onset_idxs: list
        Each element of the list is an array of the indices of the score
        corresponding to the elements in `tempo_curve`. Only returned if
        `return_onset_idxs` is True.
    """
    # use float64, float32 led to problems that x == x + eps evaluated
    # to True
    score_onsets = np.array(score_onsets).astype(np.float64, copy=False)
    performed_onsets = np.array(performed_onsets).astype(np.float64, copy=False)
    score_durations = np.array(score_durations).astype(np.float64, copy=False)

    performed_durations = np.array(performed_durations).astype(np.float64, copy=False)

    # Get unique onsets if no provided
    if unique_onset_idxs is None:
        # Get indices of the unique onsets (quantize score onsets)
        unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(int))

    # Get score information
    score_info = get_unique_seq(
        onsets=score_onsets,
        offsets=score_onsets + score_durations,
        unique_onset_idxs=unique_onset_idxs,
        return_diff=False,
    )
    # Get performance information
    perf_info = get_unique_seq(
        onsets=performed_onsets,
        offsets=performed_onsets + performed_durations,
        unique_onset_idxs=unique_onset_idxs,
        return_diff=False,
    )

    # unique score onsets
    unique_s_onsets = score_info["u_onset"]
    # equivalent onsets
    eq_onsets = perf_info["u_onset"]

    # Monotonize times
    eq_onset_mt, unique_s_onsets_mt = monotonize_times(eq_onsets, x=unique_s_onsets)
    # Function that that interpolates the equivalent performed onsets
    # as a function of the score onset.
    onset_fun = interp1d(
        unique_s_onsets_mt, eq_onset_mt, kind="linear", fill_value="extrapolate"
    )

    if input_onsets is None:
        input_onsets = unique_s_onsets[:-1]

    tempo_curve = first_order_derivative(onset_fun, input_onsets, dx=0.5)

    if return_onset_idxs:
        return tempo_curve, input_onsets, unique_onset_idxs
    else:
        return tempo_curve, input_onsets


#### Alignment Processing ####


@deprecated_alias(part="score", ppart="performance")
def to_matched_score(
    score: Union[ScoreLike, np.ndarray],
    performance: Union[PerformanceLike, np.ndarray],
    alignment: list,
    include_score_markings=False,
):
    """
    Returns a mixed score-performance note array
    consisting of matched notes in the alignment.

    Args:
        score (score.ScoreLike): score information
        performance (performance.PerformanceLike): performance information
        alignment (List(Dict)): an alignment
        include_score_markings (bool): include dynamcis and articulation
            markings (Optional)

    Returns:
        np.ndarray: a minimal, aligned
        score-performance note array
    """

    # remove repetitions from aligment note ids
    for a in alignment:
        if a["label"] == "match":
            a["score_id"] = str(a["score_id"])

    feature_functions = None
    if include_score_markings and not isinstance(score, np.ndarray):
        feature_functions = [
            "loudness_direction_feature",
            "articulation_feature",
            "tempo_direction_feature",
            "slur_feature",
        ]

    if isinstance(score, np.ndarray):
        na = score
    else:
        na = note_features.compute_note_array(
            score,
            feature_functions=feature_functions,
        )

    if isinstance(performance, np.ndarray):
        p_na = performance
    else:
        p_na = performance.note_array()
    part_by_id = dict((n["id"], na[na["id"] == n["id"]]) for n in na)
    ppart_by_id = dict((n["id"], p_na[p_na["id"] == n["id"]]) for n in p_na)

    # pair matched score and performance notes
    note_pairs = [
        (part_by_id[a["score_id"]], ppart_by_id[a["performance_id"]])
        for a in alignment
        if (a["label"] == "match" and a["score_id"] in part_by_id)
    ]
    ms = []
    # sort according to onset (primary) and pitch (secondary)
    pitch_onset = [(sn["pitch"].item(), sn["onset_div"].item()) for sn, _ in note_pairs]
    sort_order = np.lexsort(list(zip(*pitch_onset)))
    snote_ids = []
    for i in sort_order:
        sn, n = note_pairs[int(i)]
        sn_on, sn_off = [sn["onset_beat"], sn["onset_beat"] + sn["duration_beat"]]
        sn_dur = sn_off - sn_on
        # hack for notes with negative durations
        n_dur = max(n["duration_sec"], 60 / 200 * 0.25)
        pair_info = (sn_on, sn_dur, sn["pitch"], n["onset_sec"], n_dur, n["velocity"])
        if include_score_markings:
            pair_info += (sn["voice"].item(),)
            pair_info += tuple(
                [sn[field].item() for field in sn.dtype.names if "feature" in field]
            )
        ms.append(pair_info)
        snote_ids.append(sn["id"].item())

    fields = [
        ("onset", "f4"),
        ("duration", "f4"),
        ("pitch", "i4"),
        ("p_onset", "f4"),
        ("p_duration", "f4"),
        ("velocity", "i4"),
    ]
    if include_score_markings and not isinstance(score, np.ndarray):
        fields += [("voice", "i4")]
        fields += [
            (field, sn.dtype.fields[field][0])
            for field in sn.dtype.fields
            if "feature" in field
        ]

    return np.array(ms, dtype=fields), snote_ids


def get_time_maps_from_alignment(
    ppart_or_note_array, spart_or_note_array, alignment, remove_ornaments=True
):
    """
    Get time maps to convert performance time (in seconds) to score time (in beats)
    and visceversa.

    Parameters
    ----------
    ppart_or_note_array : PerformedPart or structured array
        The performance information as either PerformedPart or the
        note_array generated from such an object.
    spart_or_note_array : Part or structured array
        Score information as either a Part object or the note array
        generated from such an object.
    alignment : list
        The score--performance alignment, a list of dictionaries.
        (see `partitura.io.importmatch.alignment_from_matchfile` for reference)
    remove_ornaments : bool (optional)
        Whether to consider or not ornaments (including grace notes)

    Returns
    -------
    ptime_to_stime_map : scipy.interpolate.interp1d
        An instance of interp1d (a callable) that maps performance time (in seconds)
        to score time (in beats).
    stime_to_ptime_map : scipy.interpolate.interp1d
        An instance of inter1d (a callable) that maps score time (in beats) to
        performance time (in seconds).

    Note
    ----
    This methods uses the average value of the score onsets of notes that are
    written in the score as part of a chord (i.e., which start at the same time).
    """
    # Ensure that we are using structured note arrays
    perf_note_array = ensure_notearray(ppart_or_note_array)
    score_note_array = ensure_notearray(spart_or_note_array)

    # Get indices of the matched notes (notes in the score
    # for which there is a performance note
    match_idx = get_matched_notes(score_note_array, perf_note_array, alignment)

    # Get onsets and durations
    score_onsets = score_note_array[match_idx[:, 0]]["onset_beat"]
    score_durations = score_note_array[match_idx[:, 0]]["duration_beat"]

    perf_onsets = perf_note_array[match_idx[:, 1]]["onset_sec"]

    # Use only unique onsets
    score_unique_onsets = np.unique(score_onsets)

    # Remove grace notes
    if remove_ornaments:
        # check that all onsets have a duration
        # ornaments (grace notes) do not have a duration
        score_unique_onset_idxs = [
            np.where(np.logical_and(score_onsets == u, score_durations > 0))[0]
            for u in score_unique_onsets
        ]

    else:
        score_unique_onset_idxs = [
            np.where(score_onsets == u)[0] for u in score_unique_onsets
        ]

    # For chords, we use the average performed onset as a proxy for
    # representing the "performeance time" of the position of the score
    # onsets
    eq_perf_onsets = np.array(
        [np.mean(perf_onsets[u.astype(int)]) for u in score_unique_onset_idxs]
    )

    # Get maps
    ptime_to_stime_map = interp1d(
        x=eq_perf_onsets,
        y=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )
    stime_to_ptime_map = interp1d(
        y=eq_perf_onsets,
        x=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return ptime_to_stime_map, stime_to_ptime_map


def get_matched_notes(spart_note_array, ppart_note_array, alignment):
    """
    Get the indices of the matched notes in an alignment

    Parameters
    ----------
    spart_note_array : structured numpy array
        note_array of the score part
    ppart_note_array : structured numpy array
        note_array of the performed part
    alignment : list
        The score--performance alignment, a list of dictionaries.
        (see `partitura.io.importmatch.alignment_from_matchfile` for reference)

    Returns
    -------
    matched_idxs : np.ndarray
        A 2D array containing the indices of the matched score and
        performed notes, where the columns are
        (index_in_score_note_array, index_in_performance_notearray)
    """
    # Get matched notes
    matched_idxs = []
    for al in alignment:
        # Get only matched notes (i.e., ignore inserted or deleted notes)
        if al["label"] == "match":
            # if ppart_note_array['id'].dtype != type(al['performance_id']):
            if not isinstance(ppart_note_array["id"], type(al["performance_id"])):
                p_id = str(al["performance_id"])
            else:
                p_id = al["performance_id"]

            p_idx = np.where(ppart_note_array["id"] == p_id)[0]

            s_idx = np.where(spart_note_array["id"] == al["score_id"])[0]

            if len(s_idx) > 0 and len(p_idx) > 0:
                s_idx = int(s_idx)
                p_idx = int(p_idx)
                matched_idxs.append((s_idx, p_idx))

    if len(matched_idxs) == 0:
        warnings.warn(
            "No matched note IDs found. "
            "Either the alignment contains no matches "
            "or the IDs in score of performance do not correspond to the alignment "
            "(maybe due to repeat unfolding)."
        )

    return np.array(matched_idxs)


#### Sequence Processing: onset-wise/note-wise/monotonicity/uniqueness ####


def get_unique_seq(onsets, offsets, unique_onset_idxs=None, return_diff=False):
    """
    Get unique onsets of a sequence of notes
    """

    first_time = np.min(onsets)

    # ensure last score time is later than last onset
    if np.max(onsets) == np.max(offsets):
        # last note without duration (grace note)
        last_time = np.max(onsets) + 1
    else:
        last_time = np.max(offsets)

    total_dur = last_time - first_time

    if unique_onset_idxs is None:
        # unique_onset_idxs = unique_onset_idx(score[:, 0])
        unique_onset_idxs = get_unique_onset_idxs(onsets)

    u_onset = np.array([np.mean(onsets[uix]) for uix in unique_onset_idxs])
    # add last offset, so we have as many IOIs as notes
    u_onset = np.r_[u_onset, last_time]

    output_dict = dict(
        u_onset=u_onset, total_dur=total_dur, unique_onset_idxs=unique_onset_idxs
    )

    if return_diff:
        output_dict["diff_u_onset"] = np.diff(u_onset)

    return output_dict


def get_unique_onset_idxs(
    onsets, eps: float = 1e-6, return_unique_onsets: bool = False
):
    """
    Get unique onsets and their indices.
    Parameters
    ----------
    onsets : np.ndarray
        Score onsets in beats.
    eps : float
        Small epsilon (for dealing with quantization in symbolic scores).
        This is particularly useful for dealing with triplets and other
        similar rhytmical structures that do not have a finite decimal
        representation.
    return_unique_onsets : bool (optional)
        If `True`, returns the unique score onsets.
    Returns
    -------
    unique_onset_idxs : np.ndarray
        Indices of the unique onsets in the score.
    unique_onsets : np.ndarray
        Unique score onsets
    """
    # Do not assume that the onsets are sorted
    # (use a stable sorting algorithm for preserving the order
    # of elements with the same onset, which is useful e.g. if the
    # notes within a same onset are ordered by pitch)
    sort_idx = np.argsort(onsets, kind="mergesort")
    split_idx = np.where(np.diff(onsets[sort_idx]) > eps)[0] + 1
    unique_onset_idxs = np.split(sort_idx, split_idx)

    if return_unique_onsets:
        # Instead of np.unique(onsets)
        unique_onsets = np.array([onsets[uix].mean() for uix in unique_onset_idxs])

        return unique_onset_idxs, unique_onsets
    else:
        return unique_onset_idxs


def notewise_to_onsetwise(notewise_inputs, unique_onset_idxs):
    """Agregate basis functions per onset"""

    if notewise_inputs.ndim == 1:
        shape = len(unique_onset_idxs)
    else:
        shape = (len(unique_onset_idxs),) + notewise_inputs.shape[1:]
    onsetwise_inputs = np.zeros(shape, dtype=notewise_inputs.dtype)

    for i, uix in enumerate(unique_onset_idxs):
        try:
            onsetwise_inputs[i] = notewise_inputs[uix].mean(0)
        except TypeError:
            for tn in notewise_inputs.dtype.names:
                onsetwise_inputs[i][tn] = notewise_inputs[uix][tn].mean()
    return onsetwise_inputs


def onsetwise_to_notewise(onsetwise_input, unique_onset_idxs):
    """Expand onsetwise predictions for each note"""
    n_notes = sum([len(uix) for uix in unique_onset_idxs])
    if onsetwise_input.ndim == 1:
        shape = n_notes
    else:
        shape = (n_notes,) + onsetwise_input.shape[1:]
    notewise_inputs = np.zeros(shape, dtype=onsetwise_input.dtype)

    for i, uix in enumerate(unique_onset_idxs):
        notewise_inputs[uix] = onsetwise_input[[i]]
    return notewise_inputs


#### Temo Parameter Normalizations ####


def bp_scale(beat_period):
    return [beat_period]


def bp_rescale(tempo_params):
    return tempo_params["beat_period"]


def beat_period_log_scale(beat_period):
    return [np.log2(beat_period)]


def beat_period_log_rescale(tempo_params):
    return 2 ** tempo_params["beat_period_log"]


def beat_period_standardized_scale(beat_period):
    beat_period_std = np.std(beat_period) * np.ones_like(beat_period)
    beat_period_mean = np.mean(beat_period) * np.ones_like(beat_period)
    beat_period_standardized = (beat_period - beat_period_mean) / beat_period_std
    return [beat_period_standardized, beat_period_mean, beat_period_std]


def beat_period_standardized_rescale(tempo_params):
    return (
        tempo_params["beat_period_standardized"] * tempo_params["beat_period_std"]
        + tempo_params["beat_period_mean"]
    )


def beat_period_ratio_scale(beat_period):
    beat_period_mean = np.mean(beat_period) * np.ones_like(beat_period)
    beat_period_ratio = beat_period / beat_period_mean
    return [beat_period_ratio, beat_period_mean]


def beat_period_ratio_rescale(tempo_params):
    return tempo_params["beat_period_ratio"] * tempo_params["beat_period_mean"]


def beat_period_ratio_log_scale(beat_period):
    beat_period_ratio, beat_period_mean = beat_period_ratio_scale(beat_period)
    return [np.log2(beat_period_ratio), beat_period_mean]


def beat_period_ratio_log_rescale(tempo_params):
    return 2 ** tempo_params["beat_period_ratio_log"] * tempo_params["beat_period_mean"]


TEMPO_NORMALIZATION = dict(
    beat_period=dict(scale=bp_scale, rescale=bp_rescale, param_names=("beat_period",)),
    beat_period_log=dict(
        scale=beat_period_log_scale,
        rescale=beat_period_log_rescale,
        param_names=("beat_period_log",),
    ),
    beat_period_ratio=dict(
        scale=beat_period_ratio_scale,
        rescale=beat_period_ratio_rescale,
        param_names=("beat_period_ratio", "beat_period_mean"),
    ),
    beat_period_ratio_log=dict(
        scale=beat_period_ratio_log_scale,
        rescale=beat_period_ratio_log_rescale,
        param_names=("beat_period_ratio_log", "beat_period_mean"),
    ),
    beat_period_standardized=dict(
        scale=beat_period_standardized_scale,
        rescale=beat_period_standardized_rescale,
        param_names=("beat_period_standardized", "beat_period_mean", "beat_period_std"),
    ),
)
