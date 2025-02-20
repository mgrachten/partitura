#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing and exporting symbolic music formats.
"""
from typing import Union
import os
import urllib.request
from urllib.parse import urlparse
import tempfile
from .importmusicxml import load_musicxml
from .importmidi import load_score_midi, load_performance_midi
from .musescore import load_via_musescore
from .importmatch import load_match
from .importmei import load_mei
from .importkern import load_kern
from .exportkern import save_kern
from .importparangonada import load_parangonada_csv
from .exportparangonada import save_parangonada_csv
from .importmusic21 import load_music21
from .exportmei import save_mei
from .importdcml import load_dcml
from partitura.utils.misc import (
    deprecated_alias,
    deprecated_parameter,
    PathLike,
)

from partitura.score import Score, Part, merge_parts
from partitura.performance import Performance


class NotSupportedFormatError(Exception):
    pass


def is_url(input):
    try:
        result = urlparse(input)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@deprecated_alias(score_fn="filename")
@deprecated_parameter("ensure_list")
def load_score(filename: PathLike, force_note_ids="keep") -> Score:
    """
    Load a score format supported by partitura. Currently the accepted formats
    are MusicXML, MIDI, Kern and MEI, plus all formats for which
    MuseScore has support import-support (requires MuseScore 4 or 3).

    Parameters
    ----------
    filename : str or file-like  object
        Filename of the score to parse, or a file-like object
    force_note_ids : (None, bool or "keep")
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the input file will be discarded. If 'keep', only notes without
        a note id will be assigned one. If None or False, the notes in the
        resulting Part(s) will have an id only if the input file has ids for
        the notes.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A score instance.
    """
    if is_url(filename):
        url = filename
        # Send a GET request to the URL
        with urllib.request.urlopen(url) as response:
            data = response.read()

        # Extract the file extension from the URL
        extension = os.path.splitext(url)[-1]

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=True)

        # Write the content to the temporary file
        with open(temp_file.name, "wb") as f:
            f.write(data)

        filename = temp_file.name
    else:
        extension = os.path.splitext(filename)[-1].lower()

    if extension in (".mxl", ".xml", ".musicxml"):
        # Load MusicXML
        score = load_musicxml(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    elif extension in [".midi", ".mid"]:
        # Load MIDI
        if (force_note_ids is None) or (not force_note_ids):
            assign_note_ids = False
        else:
            assign_note_ids = True
        score = load_score_midi(
            filename=filename,
            assign_note_ids=assign_note_ids,
        )
    elif extension in [".mei"]:
        # Load MEI
        score = load_mei(filename=filename)
    elif extension in [".kern", ".krn"]:
        score = load_kern(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    elif extension in [
        ".mscz",
        ".mscx",
        ".musescore",
        ".mscore",
        ".ms",
        ".kar",
        ".md",
        ".cap",
        ".capx",
        ".bww",
        ".mgu",
        ".sgu",
        ".ove",
        ".scw",
        ".ptb",
        ".gtp",
        ".gp3",
        ".gp4",
        ".gp5",
        ".gpx",
        ".gp",
    ]:
        # Load MuseScore
        score = load_via_musescore(
            filename=filename,
            force_note_ids=force_note_ids,
        )
    elif extension in [".match"]:
        # Load the score information from a Matchfile
        _, _, score = load_match(
            filename=filename,
            create_score=True,
        )
    else:
        raise NotSupportedFormatError(
            f"{extension} file extension is not supported. If this should be supported, consider editing partitura/io/__init__.py file"
        )
    return score


def load_score_as_part(filename: PathLike) -> Part:
    """
    load part helper function:
    Load a score format supported by partitura and
    merge the result in a single part

    Parameters
    ----------
    filename : str or file-like  object
        Filename of the score to parse, or a file-like object

    Returns
    -------
    part: :class:`partitura.score.Part`
        A part instance.
    """
    scr = load_score(filename)
    part = merge_parts(scr.parts)
    return part


# alias
lp = load_score_as_part


@deprecated_alias(performance_fn="filename")
def load_performance(
    filename: PathLike,
    default_bpm: Union[float, int] = 120,
    merge_tracks: bool = False,
    first_note_at_zero: bool = False,
    pedal_threshold: int = 64,
) -> Performance:
    """
    Load a performance format supported by partitura. Currently the accepted formats
    are MIDI and matchfiles.

    Parameters
    ----------
    filename: str or file-like  object
        Filename of the score to parse, or a file-like object
    default_bpm : number, optional
        Tempo to use wherever the MIDI does not specify a tempo.
        Defaults to 120.
    merge_tracks: bool, optional
        For MIDI files, merges all tracks into a single track.
    first_note_at_zero: bool, optional
        Remove silence at the beginning, so that the first note (or
        first MIDI message, e.g., pedal) starts at time 0.
    pedal_threshold: int
        Threshold for the sustain pedal.

    Returns
    -------
    performance: :class:`partitura.performance.Performance`
        A `Performance` instance.

    TODO
    ----
    * Force loading scores as PerformedParts?
    """
    from partitura.utils.music import remove_silence_from_performed_part

    performance = None

    # Catch exceptions
    exception_dictionary = dict()
    try:
        performance = load_performance_midi(
            filename=filename,
            default_bpm=default_bpm,
            merge_tracks=merge_tracks,
        )

        # set threshold for sustain pedal
        performance[0].sustain_pedal_threshold = pedal_threshold

        if first_note_at_zero:
            remove_silence_from_performed_part(performance[0])

    except Exception as e:
        exception_dictionary["midi"] = e

    try:
        performance, _ = load_match(
            filename=filename,
            first_note_at_zero=first_note_at_zero,
            pedal_threshold=pedal_threshold,
        )
    except Exception as e:
        exception_dictionary["match"] = e

    if performance is None:
        for file_format, exception in exception_dictionary.items():
            print(f"Error loading score as {file_format}:")
            print(exception)
        raise NotSupportedFormatError

    return performance
