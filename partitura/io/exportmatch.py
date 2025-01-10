#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting matchfiles.

Notes
-----
* The methods only export matchfiles version 1.0.0.
"""
import numpy as np

from typing import List, Optional, Iterable

from collections import defaultdict

from fractions import Fraction

from partitura.score import Score, Part, ScoreLike
from partitura.performance import Performance, PerformedPart, PerformanceLike

from partitura.io.matchlines_v1 import (
    make_info,
    make_scoreprop,
    MatchSnote,
    MatchNote,
    MatchSnoteNote,
    MatchSnoteDeletion,
    MatchInsertionNote,
    MatchSustainPedal,
    MatchSoftPedal,
    MatchOrnamentNote,
    LATEST_VERSION,
)

from partitura.io.matchfile_utils import (
    FractionalSymbolicDuration,
    MatchKeySignature,
    MatchTimeSignature,
    MatchTempoIndication,
    Version,
)

from partitura import score
from partitura.io.matchfile_base import MatchFile

from partitura.utils.music import (
    seconds_to_midi_ticks,
)

from partitura.utils.misc import (
    PathLike,
    deprecated_alias,
    deprecated_parameter,
)

from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment

__all__ = ["save_match"]


@deprecated_parameter("magaloff_zeilinger_quirk")
def matchfile_from_alignment(
    alignment: List[dict],
    ppart: PerformedPart,
    spart: Part,
    mpq: int = 500000,
    ppq: int = 480,
    performer: Optional[str] = None,
    composer: Optional[str] = None,
    piece: Optional[str] = None,
    score_filename: Optional[PathLike] = None,
    performance_filename: Optional[PathLike] = None,
    assume_part_unfolded: bool = False,
    tempo_indication: Optional[str] = None,
    diff_score_version_notes: Optional[list] = None,
    version: Version = LATEST_VERSION,
    debug: bool = False,
) -> MatchFile:
    """
    Generate a MatchFile object from an Alignment, a PerformedPart and
    a Part

    Parameters
    ----------
    alignment : list
        A list of dictionaries containing alignment information.
        See `partitura.io.importmatch.alignment_from_matchfile`.
    ppart : partitura.performance.PerformedPart
        An instance of `PerformedPart` containing performance information.
    spart : partitura.score.Part
        An instance of `Part` containing score information.
    mpq : int
        Microseconds per quarter note.
    ppq: int
        Parts per quarter note.
    performer : str or None
        Name(s) of the performer(s) of the `PerformedPart`.
    composer : str or None
        Name(s) of the composer(s) of the piece represented by `Part`.
    piece : str or None:
        Name of the piece represented by `Part`.
    score_filename: PathLike
        Name of the file containing the score.
    performance_filename: PathLike
        Name of the (MIDI) file containing the performance.
    assume_part_unfolded: bool
        Whether to assume that the part has been unfolded according to the
        repetitions in the alignment. If False, the part will be automatically
        unfolded to have maximal coverage of the notes in the alignment.
        See `partitura.score.unfold_part_alignment`.
    tempo_indication : str or None
        The tempo direction indicated in the beginning of the score
    diff_score_version_notes : list or None
        A list of score notes that reflect a special score version (e.g., original edition/Erstdruck, Editors note etc.)
    version: Version
        Version of the match file. For now only 1.0.0 is supported.
    Returns
    -------
    matchfile : MatchFile
        An instance of `partitura.io.importmatch.MatchFile`.
    """
    if version < Version(1, 0, 0):
        raise ValueError("Version should >= 1.0.0")

    if not assume_part_unfolded:
        # unfold score according to alignment
        spart = score.unfold_part_alignment(spart, alignment)

    # Info Header Lines
    header_lines = dict()

    header_lines["version"] = make_info(
        version=version,
        attribute="matchFileVersion",
        value=version,
    )

    header_lines["performer"] = make_info(
        version=version,
        attribute="performer",
        value="-" if performer is None else performer,
    )

    header_lines["piece"] = make_info(
        version=version,
        attribute="piece",
        value="-" if piece is None else piece,
    )

    header_lines["composer"] = make_info(
        version=version,
        attribute="composer",
        value="-" if composer is None else composer,
    )

    header_lines["score_filename"] = make_info(
        version=version,
        attribute="scoreFileName",
        value="-" if score_filename is None else score_filename,
    )

    header_lines["performance_filename"] = make_info(
        version=version,
        attribute="midiFileName",
        value="-" if performance_filename is None else performance_filename,
    )

    header_lines["clock_units"] = make_info(
        version=version,
        attribute="midiClockUnits",
        value=int(ppq),
    )

    header_lines["clock_rate"] = make_info(
        version=version,
        attribute="midiClockRate",
        value=int(mpq),
    )

    # Measure map (which measure corresponds to which time point in divs)
    beat_map = spart.beat_map

    ptime_to_stime_map, _ = get_time_maps_from_alignment(
        ppart_or_note_array=ppart.note_array(),
        spart_or_note_array=spart.note_array(),
        alignment=alignment,
        remove_ornaments=True,
    )

    measures = np.array(list(spart.iter_all(score.Measure)))
    measure_starts_divs = np.array([m.start.t for m in measures])
    measure_starts_beats = beat_map(measure_starts_divs)
    measure_sorting_idx = measure_starts_divs.argsort()
    measure_starts_divs = measure_starts_divs[measure_sorting_idx]
    measures = measures[measure_sorting_idx]

    start_measure_num = 0 if measure_starts_beats.min() < 0 else 1
    measure_starts = np.column_stack(
        (
            np.arange(start_measure_num, start_measure_num + len(measure_starts_divs)),
            measure_starts_divs,
            measure_starts_beats,
        )
    )

    # Score prop header lines
    scoreprop_lines = defaultdict(list)
    # For score notes
    score_info = dict()
    # Info for sorting lines
    snote_sort_info = dict()
    for (mnum, msd, msb), m in zip(measure_starts, measures):
        time_signatures = spart.iter_all(score.TimeSignature, m.start, m.end)

        for tsig in time_signatures:
            time_divs = int(tsig.start.t)
            time_beats = float(beat_map(time_divs))
            dpq = int(spart.quarter_duration_map(time_divs))
            beat = int((time_beats - msb) // 1)

            ts_num, ts_den, _ = spart.time_signature_map(tsig.start.t)

            moffset_divs = Fraction(
                int(time_divs - msd - beat * dpq), (int(ts_den) * dpq)
            )

            scoreprop_lines["time_signatures"].append(
                make_scoreprop(
                    version=version,
                    attribute="timeSignature",
                    value=MatchTimeSignature(
                        numerator=int(ts_num),
                        denominator=int(ts_den),
                        other_components=None,
                        is_list=False,
                    ),
                    measure=int(mnum),
                    beat=beat + 1,
                    offset=FractionalSymbolicDuration(
                        numerator=moffset_divs.numerator,
                        denominator=moffset_divs.denominator,
                    ),
                    time_in_beats=time_beats,
                )
            )

        key_signatures = spart.iter_all(score.KeySignature, m.start, m.end)

        for ksig in key_signatures:
            time_divs = int(tsig.start.t)
            time_beats = float(beat_map(time_divs))
            dpq = int(spart.quarter_duration_map(time_divs))
            beat = int((time_beats - msb) // 1)

            ts_num, ts_den, _ = spart.time_signature_map(tsig.start.t)

            moffset_divs = Fraction(
                int(time_divs - msd - beat * dpq), (int(ts_den) * dpq)
            )

            scoreprop_lines["key_signatures"].append(
                make_scoreprop(
                    version=version,
                    attribute="keySignature",
                    value=MatchKeySignature(
                        fifths=int(ksig.fifths),
                        mode=ksig.mode,
                        is_list=False,
                        fmt="v1.0.0",
                    ),
                    measure=int(mnum),
                    beat=beat + 1,
                    offset=FractionalSymbolicDuration(
                        numerator=moffset_divs.numerator,
                        denominator=moffset_divs.denominator,
                    ),
                    time_in_beats=time_beats,
                )
            )

        # Get all notes in the measure
        snotes = spart.iter_all(score.Note, m.start, m.end, include_subclasses=True)
        # Beginning of each measure
        for snote in snotes:
            onset_divs, offset_divs = snote.start.t, snote.start.t + snote.duration_tied
            duration_divs = offset_divs - onset_divs

            onset_beats, offset_beats = beat_map([onset_divs, offset_divs])

            dpq = int(spart.quarter_duration_map(onset_divs))

            beat = int((onset_beats - msb) // 1)

            ts_num, ts_den, _ = spart.time_signature_map(snote.start.t)

            duration_symb = Fraction(duration_divs, dpq * 4)

            beat = int((onset_divs - msd) // dpq)

            moffset_divs = Fraction(int(onset_divs - msd - beat * dpq), (dpq * 4))

            if debug:
                duration_beats = offset_beats - onset_beats
                moffset_beat = (onset_beats - msb - beat) / ts_den
                assert np.isclose(float(duration_symb), duration_beats)
                assert np.isclose(moffset_beat, float(moffset_divs))

            score_attributes_list = []

            articulations = getattr(snote, "articulations", None)
            voice = getattr(snote, "voice", None)
            staff = getattr(snote, "staff", None)
            ornaments = getattr(snote, "ornaments", None)
            fermata = getattr(snote, "fermata", None)
            technical = getattr(snote, "technical", None)

            if voice is not None:
                score_attributes_list.append(f"v{voice}")

            if staff is not None:
                score_attributes_list.append(f"staff{staff}")

            if articulations is not None:
                score_attributes_list += list(articulations)

            if ornaments is not None:
                score_attributes_list += list(ornaments)

            if fermata is not None:
                score_attributes_list.append("fermata")

            if technical is not None:
                for tech_el in technical:
                    if isinstance(tech_el, score.Fingering):
                        score_attributes_list.append(f"fingering{tech_el.fingering}")

            if isinstance(snote, score.GraceNote):
                score_attributes_list.append("grace")

            if (
                diff_score_version_notes is not None
                and snote.id in diff_score_version_notes
            ):
                score_attributes_list.append("diff_score_version")

            score_info[snote.id] = MatchSnote(
                version=version,
                anchor=str(snote.id),
                note_name=str(snote.step).upper(),
                modifier=snote.alter if snote.alter is not None else 0,
                octave=int(snote.octave),
                measure=int(mnum),
                beat=beat + 1,
                offset=FractionalSymbolicDuration(
                    numerator=moffset_divs.numerator,
                    denominator=moffset_divs.denominator,
                ),
                duration=FractionalSymbolicDuration(
                    numerator=duration_symb.numerator,
                    denominator=duration_symb.denominator,
                ),
                onset_in_beats=onset_beats,
                offset_in_beats=offset_beats,
                score_attributes_list=score_attributes_list,
            )
            snote_sort_info[snote.id] = (
                onset_beats,
                snote.doc_order if snote.doc_order is not None else 0,
            )

    # # NOTE time position is hardcoded, not pretty...  Assumes there is only one tempo indication at the beginning of the score
    if tempo_indication is not None:
        score_tempo_direction_header = make_scoreprop(
            version=version,
            attribute="tempoIndication",
            value=MatchTempoIndication(
                tempo_indication,
                is_list=False,
            ),
            measure=measure_starts[0][0],
            beat=1,
            offset=0,
            time_in_beats=measure_starts[0][2],
        )
        scoreprop_lines["tempo_indication"].append(score_tempo_direction_header)

    perf_info = dict()
    pnote_sort_info = dict()
    for pnote in ppart.notes:
        onset = seconds_to_midi_ticks(pnote["note_on"], mpq=mpq, ppq=ppq)
        offset = seconds_to_midi_ticks(pnote["note_off"], mpq=mpq, ppq=ppq)
        perf_info[pnote["id"]] = MatchNote(
            version=version,
            id=(
                f"n{pnote['id']}"
                if not str(pnote["id"]).startswith("n")
                else str(pnote["id"])
            ),
            midi_pitch=int(pnote["midi_pitch"]),
            onset=onset,
            offset=offset,
            velocity=pnote["velocity"],
            channel=pnote.get("channel", 0),
            track=pnote.get("track", 0),
        )
        pnote_sort_info[pnote["id"]] = (
            float(ptime_to_stime_map(pnote["note_on"])),
            pnote["midi_pitch"],
        )

    sort_stime = []
    note_lines = []

    # Get ids of notes which voice overlap
    sna = spart.note_array()
    onset_pitch_slice = sna[["onset_div", "pitch"]]
    uniques, counts = np.unique(onset_pitch_slice, return_counts=True)
    duplicate_values = uniques[counts > 1]
    duplicates = dict()
    for v in duplicate_values:
        idx = np.where(onset_pitch_slice == v)[0]
        duplicates[tuple(v)] = idx
    voice_overlap_note_ids = []
    if len(duplicates) > 0:
        duplicate_idx = np.concatenate(np.array(list(duplicates.values()))).flatten()
        voice_overlap_note_ids = list(sna[duplicate_idx]["id"])

    for al_note in alignment:
        label = al_note["label"]

        if label == "match":
            snote = score_info[al_note["score_id"]]
            pnote = perf_info[al_note["performance_id"]]
            snote_note_line = MatchSnoteNote(version=version, snote=snote, note=pnote)
            note_lines.append(snote_note_line)
            sort_stime.append(snote_sort_info[al_note["score_id"]])

        elif label == "deletion":
            snote = score_info[al_note["score_id"]]
            if al_note["score_id"] in voice_overlap_note_ids:
                snote.ScoreAttributesList.append("voice_overlap")
            deletion_line = MatchSnoteDeletion(version=version, snote=snote)
            note_lines.append(deletion_line)
            sort_stime.append(snote_sort_info[al_note["score_id"]])

        elif label == "insertion":
            note = perf_info[al_note["performance_id"]]
            insertion_line = MatchInsertionNote(version=version, note=note)
            note_lines.append(insertion_line)
            sort_stime.append(pnote_sort_info[al_note["performance_id"]])

        elif label == "ornament":
            ornament_type = al_note["type"]
            snote = score_info[al_note["score_id"]]
            note = perf_info[al_note["performance_id"]]
            ornament_line = MatchOrnamentNote(
                version=version,
                anchor=snote.Anchor,
                note=note,
                ornament_type=[ornament_type],
            )

            note_lines.append(ornament_line)
            sort_stime.append(pnote_sort_info[al_note["performance_id"]])

    # sort notes by score onset (performed insertions are sorted
    # according to the interpolation map
    sort_stime = np.array(sort_stime)
    sort_stime_idx = np.lexsort((sort_stime[:, 1], sort_stime[:, 0]))
    note_lines = np.array(note_lines)[sort_stime_idx]

    # Create match lines for pedal information
    pedal_lines = []
    for c in ppart.controls:
        t = seconds_to_midi_ticks(c["time"], mpq=mpq, ppq=ppq)
        value = int(c["value"])
        if c["number"] == 64:  # c['type'] == 'sustain_pedal':
            sustain_pedal = MatchSustainPedal(version=version, time=t, value=value)
            pedal_lines.append(sustain_pedal)

        if c["number"] == 67:  # c['type'] == 'soft_pedal':
            soft_pedal = MatchSoftPedal(version=version, time=t, value=value)
            pedal_lines.append(soft_pedal)

    pedal_lines.sort(key=lambda x: x.Time)

    # Construct header of match file
    header_order = [
        "version",
        "piece",
        "score_filename",
        "performance_filename",
        "composer",
        "performer",
        "clock_units",
        "clock_rate",
        "key_signatures",
        "time_signatures",
        "tempo_indication",
    ]
    all_match_lines = []
    for h in header_order:
        if h in header_lines:
            all_match_lines.append(header_lines[h])

        if h in scoreprop_lines:
            all_match_lines += scoreprop_lines[h]

    # Concatenate all lines
    all_match_lines += list(note_lines) + pedal_lines

    matchfile = MatchFile(lines=all_match_lines)

    return matchfile


@deprecated_alias(spart="score_data", ppart="performance_data")
def save_match(
    alignment: List[dict],
    performance_data: PerformanceLike,
    score_data: ScoreLike,
    out: PathLike = None,
    mpq: int = 500000,
    ppq: int = 480,
    performer: Optional[str] = None,
    composer: Optional[str] = None,
    piece: Optional[str] = None,
    score_filename: Optional[PathLike] = None,
    performance_filename: Optional[PathLike] = None,
    assume_unfolded: bool = False,
) -> Optional[MatchFile]:
    """
    Save an Alignment of a PerformedPart to a Part in a match file.

    Parameters
    ----------
    alignment : list
        A list of dictionaries containing alignment information.
        See `partitura.io.importmatch.alignment_from_matchfile`.
    performance_data : `PerformanceLike
        The performance information as a `Performance`
    score_data : `ScoreLike`
        The musical score. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    out : str
        Out to export the matchfile.
    mpq : int
        Milliseconds per quarter note.
    ppq: int
        Parts per quarter note.
    performer : str or None
        Name(s) of the performer(s) of the `PerformedPart`.
    composer : str or None
        Name(s) of the composer(s) of the piece represented by `Part`.
    piece : str or None:
        Name of the piece represented by `Part`.
    score_filename: PathLike
        Name of the file containing the score.
    performance_filename: PathLike
        Name of the (MIDI) file containing the performance.
    assume_part_unfolded: bool
        Whether to assume that the part has been unfolded according to the
        repetitions in the alignment. If False, the part will be automatically
        unfolded to have maximal coverage of the notes in the alignment.
        See `partitura.score.unfold_part_alignment`.

    Returns
    -------
    matchfile: MatchFile
        If no output is specified using `out`, the function returns
        a `MatchFile` object. Otherwise, the function returns None.
    """

    # For now, we assume that we align only one Part and a PerformedPart

    if isinstance(score_data, (Score, Iterable)):
        spart = score_data[0]
    elif isinstance(score_data, Part):
        spart = score_data
    elif isinstance(score_data, score.PartGroup):
        spart = score_data.children[0]
    else:
        raise ValueError(
            "`score_data` should be a `Score`, a `Part`, a `PartGroup` or a "
            f"list of `Part` objects, but is {type(score_data)}"
        )

    if isinstance(performance_data, (Performance, Iterable)):
        ppart = performance_data[0]
    elif isinstance(performance_data, PerformedPart):
        ppart = performance_data
    else:
        raise ValueError(
            "`performance_data` should be a `Performance`, a `PerformedPart`, or a "
            f"list of `PerformedPart` objects, but is {type(performance_data)}"
        )

    # Get matchfile
    matchfile = matchfile_from_alignment(
        alignment=alignment,
        ppart=ppart,
        spart=spart,
        mpq=mpq,
        ppq=ppq,
        performer=performer,
        composer=composer,
        piece=piece,
        score_filename=score_filename,
        performance_filename=performance_filename,
        assume_part_unfolded=assume_unfolded,
    )

    if out is not None:
        # write matchfile
        matchfile.write(out)
    else:
        return matchfile
