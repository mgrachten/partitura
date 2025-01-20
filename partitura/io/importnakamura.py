#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing score-to-performance alignments
in Nakamura et al.'s [1]_ format.

References
----------
.. [1] Nakamura, E., Yoshii, K. and Katayose, H. (2017) "Performance Error
       Detection and Post-Processing for Fast and Accurate Symbolic Music
       Alignment"
"""
import warnings
import re
import numpy as np

from typing import Union, Tuple

from partitura.utils import note_name_to_midi_pitch
from partitura.utils.music import SIGN_TO_ALTER

from partitura.utils.misc import PathLike, deprecated_alias


NAME_PATT = re.compile(r"([A-G]{1})([xb\#]*)(\d+)")


@deprecated_alias(fn="filename")
def load_nakamuracorresp(filename: PathLike) -> Tuple[Union[np.ndarray, list]]:
    """Load a corresp file as returned by Nakamura et al.'s MIDI to MIDI alignment.

    Fields of the file format as specified in [8]_:
    (ID) (onset time) (spelled pitch) (integer pitch) (onset velocity)

    Parameters
    ----------
    filename : str
        The nakamura corresp.txt-file

    Returns
    -------
    align : structured array
        structured array of performed notes
    ref : structured array
        structured array of score notes
    alignment : list
        The score--performance alignment, a list of dictionaries
    """
    note_array_dtype = [("onset_sec", "f4"), ("pitch", "i4"), ("id", "U256")]
    dtype = [
        ("alignID", "U256"),
        ("alignOntime", "f"),
        ("alignSitch", "U256"),
        ("alignPitch", "i"),
        ("alignOnvel", "i"),
        ("refID", "U256"),
        ("refOntime", "f"),
        ("refSitch", "U256"),
        ("refPitch", "i"),
        ("refOnvel", "i"),
    ]
    result = np.loadtxt(filename, dtype=dtype, comments="//")

    align_valid = result["alignID"] != "*"
    n_align = sum(align_valid)
    align = np.empty((n_align,), dtype=note_array_dtype)
    align[:] = result[["alignOntime", "alignPitch", "alignID"]][align_valid]

    ref_valid = result["refID"] != "*"
    n_ref = sum(ref_valid)
    ref = np.empty((n_ref,), dtype=note_array_dtype)
    ref[:] = result[["refOntime", "refPitch", "refID"]][ref_valid]

    alignment = []
    for alignID, refID in result[["alignID", "refID"]]:
        if alignID == "*":
            alnote = dict(label="deletion", score_id=refID)
        elif refID == "*":
            alnote = dict(label="insertion", performance_id=alignID)
        else:
            alnote = dict(label="match", score_id=refID, performance_id=alignID)
        alignment.append(alnote)

    return align, ref, alignment


@deprecated_alias(fn="filename")
def load_nakamuramatch(filename: PathLike) -> Tuple[Union[np.ndarray, list]]:
    """Load a match file as returned by Nakamura et al.'s MIDI to musicxml alignment

    Fields of the file format as specified in [8]_:
    ID (onset time) (offset time) (spelled pitch) (onset velocity)(offset velocity)
    channel (match status) (score time) (note ID)(error index) (skip index)

    Parameters
    ----------
    filename : str
        The nakamura match.txt-file

    Returns
    -------
    align : structured array
        structured array of performed notes
    ref : structured array
        structured array of score notes
    alignment : list
        The score--performance alignment, a list of dictionaries

    References
    ----------
    .. [8] https://midialignment.github.io/MANUAL.pdf

    """
    perf_dtype = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("channel", "i4"),
        ("id", "U256"),
    ]
    score_dtype = [
        ("onset_div", "i4"),
        ("pitch", "i4"),
        ("step", "U256"),
        ("alter", "i4"),
        ("octave", "i4"),
        ("id", "U256"),
    ]
    dtype = [
        ("alignID", "U256"),
        ("alignOntime", "f"),
        ("alignOfftime", "f"),
        ("alignSitch", "U256"),
        ("alignOnvel", "i"),
        ("alignOffvel", "i"),
        ("alignChannel", "i"),
        ("matchstatus", "i"),
        ("refOntime", "f"),
        ("refID", "U256"),
        ("errorindex", "i"),
        ("skipindex", "U256"),
    ]
    dtype_missing = [("refOntime", "f"), ("refID", "U256")]
    pattern = r"//Missing\s(\d+)\t(.+)"
    # load alignment notes
    result = np.loadtxt(filename, dtype=dtype, comments="//")
    # load missing notes
    missing = np.fromregex(filename, pattern, dtype=dtype_missing)

    midi_pitch = np.array([note_name_to_midi_pitch(n) for n in result["alignSitch"]])

    align_valid = result["alignID"] != "*"
    n_align = sum(align_valid)
    align = np.empty((n_align,), dtype=perf_dtype)
    align["id"] = result["alignID"][align_valid]
    align["onset_sec"] = result["alignOntime"]
    align["duration_sec"] = (
        result["alignOfftime"][align_valid] - result["alignOntime"][align_valid]
    )
    align["pitch"] = midi_pitch[align_valid]
    align["velocity"] = result["alignOnvel"][align_valid]
    align["channel"] = result["alignChannel"][align_valid]

    ref_valid = result["refID"] != "*"
    n_valid = sum(ref_valid)
    n_ref = n_valid + len(missing)
    ref = np.empty((n_ref,), dtype=score_dtype)

    ref["id"][:n_valid] = result["refID"][ref_valid]
    ref["id"][n_valid:] = missing["refID"]
    ref["onset_div"][:n_valid] = result["refOntime"][ref_valid]
    ref["onset_div"][n_valid:] = missing["refOntime"]
    ref["pitch"][:n_valid] = midi_pitch[ref_valid]
    ref["pitch"][n_valid:] = -1
    pitch_spelling = [NAME_PATT.search(nn).groups() for nn in result["alignSitch"]]

    pitch_spelling = np.array(
        [
            (ps[0], SIGN_TO_ALTER[ps[1] if ps[1] != "" else "n"], int(ps[2]))
            for ps in pitch_spelling
        ]
    )
    # add pitch spelling information
    ref["step"][:n_valid] = pitch_spelling[ref_valid][:, 0]
    ref["alter"][:n_valid] = pitch_spelling[ref_valid][:, 1]
    ref["octave"][:n_valid] = pitch_spelling[ref_valid][:, 2]
    # * indicates that is a missing pitch
    ref["step"][n_valid:] = "*"

    alignment = []
    for alignID, refID in result[["alignID", "refID"]]:
        if alignID == "*":
            alnote = dict(label="deletion", score_id=refID)
        elif refID == "*":
            alnote = dict(label="insertion", performance_id=alignID)
        else:
            alnote = dict(label="match", score_id=refID, performance_id=alignID)
        alignment.append(alnote)

    for refID in missing["refID"]:
        alignment.append(dict(label="deletion", score_id=refID))

    return align, ref, alignment


@deprecated_alias(fn="filename")
def load_nakamuraspr(filename: PathLike) -> np.ndarray:
    """Load a spr file as returned by Nakamura et al.'s alignment methods.

    Fields of the file format as specified in [8]_:
    ID (onset time) (offset time) (spelled pitch) (onset velocity)
    (offset velocity) channel

    These files contain extra information not included in match or corresp files,
    particularly duration and pedal information, and can be used to complement the
    information from the `load_nakamuracorresp` or `load_nakamuramatch`.

    Parameters
    ----------
    filename : str
        The nakamura match.txt-file

    Returns
    -------
    note_array : structured array
        structured array with note information

    References
    ----------
    .. [8] https://midialignment.github.io/MANUAL.pdf


    TODO
    ----
    * Import pedal information
    """
    note_array_dtype = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("channel", "i4"),
        ("id", "U256"),
    ]
    dtype = [
        ("ID", "U256"),
        ("Ontime", "f"),
        ("Offtime", "f"),
        ("Sitch", "U256"),
        ("Onvel", "i"),
        ("Offvel", "i"),
        ("Channel", "i"),
    ]

    pattern = r"(\d+)\t(.+)\t(.+)\t(.+)\t(.+)\t(.+)\t(.+)"

    result = np.fromregex(filename, pattern, dtype=dtype)
    note_array = np.empty(len(result), dtype=note_array_dtype)

    note_array["id"] = result["ID"]
    note_array["onset_sec"] = result["Ontime"]
    note_array["duration_sec"] = result["Offtime"] - result["Ontime"]
    note_array["pitch"] = np.array(
        [note_name_to_midi_pitch(n) for n in result["Sitch"]]
    )
    note_array["velocity"] = result["Onvel"]
    note_array["channel"] = result["Channel"]

    return note_array
