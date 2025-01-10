from partitura.score import ScoreLike, Part
from partitura.utils import (
    estimate_symbolic_duration,
    estimate_clef_properties,
    key_name_to_fifths_mode,
    fifths_mode_to_key_name,
)
import warnings
import numpy as np
from typing import Union
import numpy.lib.recfunctions as rfn
from fractions import Fraction
import partitura.musicanalysis as analysis
import partitura.score as score


def create_divs_from_beats(note_array: np.ndarray):
    """
    Append onset_div and duration_div fields to the note array.
    Assumes beats are in uniform units across the whole array
    (no time signature change that modifies beat unit, e.g., 4/4 to 6/8).

    This function may result in an error if time signature changes that affect the ratio of beat/div are present.
    Parameters
    ----------
    note_array: np.ndarray
        The note array to which the divs fields will be added.
        Normally only beat onset and duration are provided.

    Returns
    -------
    note_array: np.ndarray
        The note array with the divs fields added.
    divs: int
        the divs per beat

    """
    duration_fractions = [
        Fraction(float(ix)).limit_denominator(256) for ix in note_array["duration_beat"]
    ]
    onset_fractions = [
        Fraction(float(ix)).limit_denominator(256) for ix in note_array["onset_beat"]
    ]
    divs = np.lcm.reduce(
        [
            Fraction(float(ix)).limit_denominator(256).denominator
            for ix in np.unique(note_array["duration_beat"])
        ]
    )
    onset_divs = list(
        map(lambda r: int(divs * r.numerator / r.denominator), onset_fractions)
    )
    min_onset_div = min(onset_divs)
    if min_onset_div < 0:
        onset_divs = list(map(lambda x: x - min_onset_div, onset_divs))
    duration_divs = list(
        map(lambda r: int(divs * r.numerator / r.denominator), duration_fractions)
    )
    na_divs = np.array(
        list(zip(onset_divs, duration_divs)),
        dtype=[("onset_div", int), ("duration_div", int)],
    )
    return rfn.merge_arrays((note_array, na_divs), flatten=True, usemask=False), divs


def create_beats_from_divs(note_array: np.ndarray, divs: int):
    """
    Append onset_beats and duration_beasts fields to the note array.
    Returns beats in quarters.

    Parameters
    ----------
    note_array: np.ndarray
        The note array to which the divs fields will be added.
        Normally only beat onset and duration are provided.
    divs: int
        Divs/ticks per quarter note.

    Returns
    -------
    note_array: np.ndarray
        The note array with the divs fields added.

    """
    onset_beats = list(note_array["onset_div"] / divs)
    duration_beats = list(note_array["duration_div"] / divs)
    na_beats = np.array(
        list(zip(onset_beats, duration_beats)),
        dtype=[("onset_beat", float), ("duration_beat", float)],
    )
    return rfn.merge_arrays((note_array, na_beats), flatten=True, usemask=False)


def create_part(
    ticks: int,
    note_array: np.ndarray,
    key_sigs: list = None,
    time_sigs: list = None,
    part_id: str = None,
    part_name: str = None,
    sanitize: bool = True,
    anacrusis_divs: int = 0,
    barebones: bool = False,
):
    """
    Create a part from a note array and a list of key signatures.

    Parameters
    ----------
    ticks: int
        The number of ticks per quarter note for the part creation.
    note_array: np.ndarray
        The note array from which the part will be created.
    key_sigs: list (optional)
        A list of key signatures. Each key signature is a tuple of the form (onset, key_name, offset).
    time_sigs: list (optional)
        A list of time signatures. Each time signature is a tuple of the form (onset, ts_num, ts_den, offset).
    part_id: str (optional)
        The id of the part.
    part_name: str (optional)
        The name of the part
    sanitize: bool (optional)
        If True, then measures, tied-notes and triplets will be sanitized.
    anacrusis_divs: int (optional)
        The number of divisions in the anacrusis. If 0, then there is no anacrusis measure.
    barebones: bool (optional)
        Returns a part with only notes, no measures

    Returns
    -------
    part : partitura.score.Part
        The part created from the note array and key signatures.
    """

    warnings.warn("create_part", stacklevel=2)

    part = Part(
        part_id,
        part_name=part_name,
    )
    part.set_quarter_duration(0, ticks)

    clef = score.Clef(staff=1, **estimate_clef_properties(note_array["pitch"]))
    part.add(clef, 0)

    # key sig
    if key_sigs is not None:
        for t_start, name, t_end in key_sigs:
            fifths, mode = key_name_to_fifths_mode(name)
            t_start, t_end = int(t_start), int(t_end)
            part.add(score.KeySignature(fifths, mode), t_start, t_end)
    else:
        warnings.warn("No key signatures added")

    # time sig
    if time_sigs is not None:
        for ts_start, num, den, ts_end in time_sigs:
            time_sig = score.TimeSignature(num.item(), den.item())
            part.add(time_sig, ts_start, ts_end)
    else:
        warnings.warn("No time signatures added")
        # without time signature, no measures
        barebones = True

    warnings.warn("add notes", stacklevel=2)
    # add the notes
    for n in note_array:
        if n["duration_div"] > 0:
            note = score.Note(
                step=n["step"],
                octave=n["octave"],
                alter=n["alter"],
                voice=int(n["voice"] or 0),
                id=n["id"],
                symbolic_duration=estimate_symbolic_duration(n["duration_div"], ticks),
            )
        else:
            note = score.GraceNote(
                grace_type="appoggiatura",
                step=n["step"],
                octave=n["octave"],
                alter=n["alter"],
                voice=int(n["voice"] or 0),
                id=n["id"],
                symbolic_duration=dict(type="quarter"),
            )

        part.add(note, n["onset_div"], n["onset_div"] + n["duration_div"])

    warnings.warn("add measures", stacklevel=2)

    if not barebones and anacrusis_divs > 0:
        part.add(score.Measure(number=1, name=str(0)), 0, anacrusis_divs)

    if not barebones and sanitize:
        warnings.warn("Inferring measures", stacklevel=2)
        score.add_measures(part)

        warnings.warn("Find and tie notes", stacklevel=2)
        # tie notes where necessary (across measure boundaries, and within measures
        # notes with compound duration)
        score.tie_notes(part)

        warnings.warn("find and ensure tuplets", stacklevel=2)
        # apply simplistic tuplet finding heuristic
        score.find_tuplets(part)

        # clean up
        score.sanitize_part(part)

    warnings.warn("done create_part", stacklevel=2)
    return part


def note_array_to_score(
    note_array: Union[np.ndarray, list],
    name_id: str = "",
    divs: int = None,
    key_sigs: list = None,
    time_sigs: list = None,
    part_name: str = "",
    assign_note_ids: bool = True,
    estimate_key: bool = False,
    estimate_time: bool = False,
    sanitize: bool = True,
    return_part: bool = False,
) -> ScoreLike:
    """
    A generic function to transform an enriched note_array to part or Score.

    The function can be used for many different occasions, i.e. part_from_graph, part from note_array, part from midi score import, etc.
    This function requires a note array that contains time signatures and key signatures (optional - can also estimate it automatically).
    Note array should contain the following fields:
    - onset_div or onset_beat
    - duration_div or duration_beat
    - pitch

    For time signature and key signature the arguments are processed in the following hierarchy:

    Key sig: (["key_fifths", "key_mode"] fields) overrides (key_sigs list) overrides (estimate_key bool)
    Time sig: (["ts_beats", "ts_beat_type"] fields) overrides (time_sigs list) overrides (estimate_time bool)

    If either times in divs or beats are missing, these cases are assumed:

    Only divs: divs/ticks need to be specified, beats are computed as quarters (not relative to time signature).
    Only beats: divs/ticks as well as times in divs are computed assuming the beat times are given in quarters.

    This function thus handles the following cases:

    1) note_array fields ["onset_beat", "duration_beat", "pitch"]
        -> barebones part, divs estimated assuming uniform beats in quarters
        + estimate_time -> 4/4 time signature
        + estimate_key -> barebones + estimate key signature
        + time_sigs -> time signatures are added, times assumed in quarters (possible error against div/beat)
        + key_sigs -> key signatures are added, times assumed in quarters
        + ["ts_beats", "ts_beat_type"] -> time signatures are added (possible error against div/beat)
        + ["key_fifths", "key_mode"] -> key signatures are added

    2) note_array fields ["onset_div", "duration_div", "pitch"]
        -> barebones part, uniform beats in quarters estimated from beats
        + estimate_time -> 4/4 time signature
        + estimate_key -> barebones + estimate key signature
        + time_sigs -> time signatures are added, times assumed in divs (possible error against div/beat)
        + key_sigs -> key signatures are added, times assumed in divs
        + ["ts_beats", "ts_beat_type"] -> time signatures are added (possible error against div/beat)
        + ["key_fifths", "key_mode"] -> key signatures are added

    3) note_array fields ["onset_div", "duration_div", "onset_beat", "duration_beat", "pitch"]
        -> barebones part
        + estimate_time -> 4/4 time signature (possible error against div/beat)
        + estimate_key -> barebones + estimate key signature
        + time_sigs -> time signatures are added, times assumed in divs (possible error against div/beat)
        + key_sigs -> key signatures are added, times assumed in divs
        + ["ts_beats", "ts_beat_type"] -> time signatures are added (possible error against div/beat)
        + ["key_fifths", "key_mode"] -> key signatures are added

    Parameters
    ----------
    note_array : structure array or list of structured arrays.
        A note array with the following fields:
        - onset_div or onset_beat
        - duration_div or duration_beat
        - pitch
        - ts_beats (optional)
        - ts_beat_type (optional)
        - key_mode(optional)
        - key_fifths(optional)
        - id (optional)
    divs : int (optional)
        Divs/ticks per quarter note.
        If not given, it is estimated assuming a beats in quarters.
    key_sigs: list (optional)
        A list of key signatures. Each key signature is a tuple of the form (onset, key_name, offset).
        Overridden by note_array fields "key_mode" and "key_fifths".
        Overrides estimate_key.
    time_sigs: list (optional)
        A list of time signatures. Each time signature is a tuple of the form (onset, ts_num, ts_den, offset).
        Overridden by note_array fields "key_mode" and "key_fifths".
        Overrides estimate_time.
    estimate_key: bool (optional)
        Estimate a single key signature.
    estimate_time: bool (optional)
        Add a default time signature.
    assign_note_ids: bool (optional)
        Assign note_ids.
    sanitize: bool (optional)
        sanitize the part by adding measures, tying notes, and finding tuplets.
    return_part: bool (optional)
        Return a Partitura part object instead of a score.

    Returns
    -------
    part or score : Part or Score
        a Part object or a Score object, depending on return_part.
    """

    if isinstance(note_array, list):
        parts = [
            note_array_to_score(
                note_array=x,
                name_id=str(i),
                assign_note_ids=assign_note_ids,
                return_part=True,
                divs=divs,
                estimate_key=estimate_key,
                sanitize=sanitize,
                part_name=name_id + "_P" + str(i),
            )
            for i, x in enumerate(note_array)
        ]
        return score.Score(partlist=parts)

    # Input validation
    if not isinstance(note_array, np.ndarray):
        raise TypeError("The note array does not have the correct format.")

    if len(note_array) == 0:
        raise ValueError("The note array is empty.")

    dtypes = note_array.dtype.names

    ts_case = ["ts_beats", "ts_beat_type"]
    ks_case = ["key_fifths", "key_mode"]

    case1 = ["onset_beat", "duration_beat", "pitch"]
    case1_ex = ["onset_div", "duration_div"]
    case2 = ["onset_div", "duration_div", "pitch"]
    case2_ex = ["onset_beat", "duration_beat"]
    # case3 = ["onset_div", "duration_div", "onset_beat", "duration_beat", "pitch"]

    if not (all([x in dtypes for x in case1]) or all([x in dtypes for x in case2])):
        raise ValueError("not all necessary note array fields are available")

    # sort the array
    onset_time = "onset_div"
    duration_time = "duration_div"
    if all([x not in dtypes for x in case1_ex]):
        onset_time = "onset_beat"
        duration_time = "duration_beat"

    # Order Lexicographically
    sort_idx = np.lexsort(
        (note_array[duration_time], note_array["pitch"], note_array[onset_time])
    )
    note_array = note_array[sort_idx]

    # case 1, estimate divs
    if all([x in dtypes for x in case1] and [x not in dtypes for x in case1_ex]):
        # estimate onset_divs and duration_divs, assumes all beat times as quarters
        note_array, divs_ = create_divs_from_beats(note_array)
        if divs is not None and divs != divs_:
            raise ValueError("estimated divs don't correspond to input divs")
        else:
            divs = divs_

        # case 1: convert key sig times to divs
        if key_sigs is not None:
            key_sigs = np.array(key_sigs)
            if key_sigs.shape[1] == 2:
                key_sigs[:, 0] = (key_sigs[:, 0] / divs).astype(int)
            elif key_sigs.shape[1] == 3:
                key_sigs[:, 0] = (key_sigs[:, 0] / divs).astype(int)
                key_sigs[:, 2] = (key_sigs[:, 2] / divs).astype(int)
            else:
                raise ValueError("key_sigs is given in a wrong format")

        # case 1: convert time sig times to divs
        if time_sigs is not None:
            time_sigs = np.array(time_sigs)
            if time_sigs.shape[1] == 3:
                time_sigs[:, 0] = (time_sigs[:, 0] / divs).astype(int)
            elif time_sigs.shape[1] == 4:
                time_sigs[:, 0] = (time_sigs[:, 0] / divs).astype(int)
                time_sigs[:, 3] = (time_sigs[:, 3] / divs).astype(int)
            else:
                raise ValueError("time_sigs is given in a wrong format")

    # case 2, estimate beats
    if all([x in dtypes for x in case2] and [x not in dtypes for x in case2_ex]):
        # estimate onset_beats and duration_beats in quarters
        if divs is None:
            raise ValueError("Divs/ticks need to be specified")
        else:
            note_array = create_beats_from_divs(note_array, divs)

    if divs is None:
        # find first note with nonzero duration (in case score starts with grace_note).
        for idx, dur in enumerate(note_array["duration_beat"]):
            if dur != 0:
                break
        if all([x in dtypes for x in ts_case]):
            divs = int(
                (note_array[idx]["duration_div"] / note_array[idx]["duration_beat"])
                / (4 / note_array[idx]["ts_beat_type"])
            )
        else:
            divs = int(
                note_array[idx]["duration_div"] / note_array[idx]["duration_beat"]
            )

    # Test Note array for negative durations
    if not np.all(note_array["duration_div"] >= 0):
        raise ValueError("Note array contains negative durations.")
    if not np.all(note_array["duration_beat"] >= 0):
        raise ValueError("Note array contains negative durations.")

    # Test for negative divs
    if not np.all(note_array["onset_div"] >= 0):
        raise ValueError("Negative divs found in note_array.")

    # handle time signatures
    if all([x in dtypes for x in ts_case]):
        time_sigs = [[0, note_array[0]["ts_beats"], note_array[0]["ts_beat_type"]]]
        for n in note_array:
            if (
                n["ts_beats"] != time_sigs[-1][1]
                or n["ts_beat_type"] != time_sigs[-1][2]
            ):
                time_sigs.append([n["onset_div"], n["ts_beats"], n["ts_beat_type"]])
        global_time_sigs = np.array(time_sigs)
    elif time_sigs is not None:
        global_time_sigs = time_sigs
    elif estimate_time:
        global_time_sigs = [[0, 4, 4]]
    else:
        global_time_sigs = None

    if global_time_sigs is not None:
        global_time_sigs = np.array(global_time_sigs)
        if global_time_sigs.shape[1] == 3:
            # for convenience, we add the end times for each time signature
            ts_end_times = np.r_[
                global_time_sigs[1:, 0],
                np.max(note_array["onset_div"] + note_array["duration_div"]),
            ]
            global_time_sigs = np.column_stack((global_time_sigs, ts_end_times))
        elif global_time_sigs.shape[1] == 4:
            pass
        else:
            raise ValueError("time_sigs is given in a wrong format")

        # make sure there is a time signature from the beginning
        global_time_sigs[0, 0] = 0

    # Note id creation or re-assignment
    if "id" not in dtypes:
        note_ids = ["{}n{:4d}".format(name_id, i) for i in range(len(note_array))]
        note_array = rfn.append_fields(
            note_array, "id", np.array(note_ids, dtype="<U256")
        )
    elif assign_note_ids or np.all(note_array["id"] == note_array["id"][0]):
        note_ids = ["{}n{:4d}".format(name_id, i) for i in range(len(note_array))]
        note_array["id"] = np.array(note_ids)

    # estimate voice
    if "voice" in dtypes:
        estimate_voice_info = False
        part_voice_list = note_array["voice"]
    else:
        estimate_voice_info = True
        part_voice_list = np.full(len(note_array), np.inf)

    if estimate_voice_info:
        warnings.warn("voice estimation", stacklevel=2)
        # TODO: deal with zero duration notes in note_array.
        # Zero duration notes are currently deleted
        estimated_voices = analysis.estimate_voices(note_array)
        assert len(part_voice_list) == len(estimated_voices)
        for i, (part_voice, voice_est) in enumerate(
            zip(part_voice_list, estimated_voices)
        ):
            # Not sure if correct.
            if part_voice != np.inf:
                estimated_voices[i] = part_voice
        note_array = rfn.append_fields(
            note_array, "voice", np.array(estimated_voices, dtype=int)
        )

    # estimate pitch spelling
    if not all(x in dtypes for x in ["step", "alter", "octave"]):
        warnings.warn("pitch spelling")
        spelling_global = analysis.estimate_spelling(note_array)
        note_array = rfn.merge_arrays((note_array, spelling_global), flatten=True)

    # handle or estimate key signature
    if all([x in dtypes for x in ks_case]):
        global_key_sigs = [
            [
                0,
                fifths_mode_to_key_name(
                    note_array[0]["ks_fifths"], note_array[0]["ks_mode"]
                ),
            ]
        ]
        for n in note_array:
            global_key_sigs.append(
                [n["onset_div"], fifths_mode_to_key_name(n["ks_fifths"], n["ks_mode"])]
            )
        else:
            global_key_sigs = key_sigs
    elif key_sigs is not None:
        global_key_sigs = key_sigs
    elif estimate_key:
        k_name = analysis.estimate_key(note_array)
        global_key_sigs = [[0, k_name]]
    else:
        global_key_sigs = None

    if global_key_sigs is not None:
        global_key_sigs = np.array(global_key_sigs)
        if global_key_sigs.shape[1] == 2:
            # for convenience, we add the end times for each time signature
            ks_end_times = np.r_[
                global_key_sigs[1:, 0],
                np.max(note_array["onset_div"] + note_array["duration_div"]),
            ]
            global_key_sigs = np.column_stack((global_key_sigs, ks_end_times))
        elif global_key_sigs.shape[1] == 3:
            pass
        else:
            raise ValueError("key_sigs is given in a wrong format")

        # make sure there is a key signature from the beginning
        global_key_sigs[0, 0] = 0

    # Steps for dealing with anacrusis measure.
    anacrusis_mask = np.zeros(len(note_array), dtype=bool)
    anacrusis_mask[note_array["onset_beat"] < 0] = True

    if np.all(anacrusis_mask == False):
        anacrusis_divs = 0
    else:
        last_neg_beat = np.max(note_array[anacrusis_mask]["onset_beat"])
        last_neg_divs = np.max(note_array[anacrusis_mask]["onset_div"])
        if all([x in dtypes for x in ts_case]):
            beat_type = np.max(note_array[anacrusis_mask]["ts_beat_type"])
        else:
            beat_type = 4
        difference_from_zero = (0 - last_neg_beat) * divs * (4 / beat_type)
        anacrusis_divs = int(last_neg_divs + difference_from_zero)

    # Create the part
    part = create_part(
        ticks=divs,
        note_array=note_array,
        key_sigs=global_key_sigs,
        time_sigs=global_time_sigs,
        part_id=name_id,
        part_name=part_name,
        sanitize=sanitize,
        anacrusis_divs=anacrusis_divs,
    )
    # Return Part or Score
    if return_part:
        return part
    else:
        return score.Score(partlist=[part], id=name_id)
