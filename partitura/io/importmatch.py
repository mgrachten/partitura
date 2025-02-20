#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""
import os
from typing import Union, Tuple, Optional, Callable, List
import warnings
from functools import partial
import numpy as np

from partitura import score
from partitura.score import Part, Score
from partitura.performance import PerformedPart, Performance

from partitura.io.matchlines_v0 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV0,
    MatchInfo as MatchInfoV0,
    MatchTrillNote as MatchTrillNoteV0,
)

from partitura.io.matchlines_v1 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV1,
    MatchInfo as MatchInfoV1,
    MatchOrnamentNote as MatchOrnamentNoteV1,
)

from partitura.io.matchfile_base import (
    MatchError,
    MatchFile,
    MatchLine,
    BaseSnoteLine,
    BaseSnoteNoteLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseOrnamentLine,
)

from partitura.io.matchfile_utils import (
    Version,
    number_pattern,
    vnumber_pattern,
    fingering_pattern,
    MatchTimeSignature,
    format_pnote_id,
)

from partitura.utils.music import midi_ticks_to_seconds


from partitura.utils.misc import deprecated_alias, PathLike, get_document_name

from partitura.utils.generic import interp1d, iter_current_next

__all__ = ["load_match"]


def get_version(line: str) -> Version:
    """
    Get version from the first line. Since the
    first version of the format did not include this line,
    we assume that the version is 0.1.0 if no version is
    found.

    Parameters
    ----------
    line: str
        The first line of the match file.

    Returns
    -------
    version : Version
        The version of the match file
    """
    version = Version(0, 1, 0)

    for parser in (MatchInfoV1, MatchInfoV0):
        try:
            ml = parser.from_matchline(line)
            if isinstance(getattr(ml, "Value", None), Version):
                version = ml.Value
                return version

        except MatchError:
            pass

    return version


def parse_matchline(
    line: str,
    from_matchline_methods: List[Callable[[str], MatchLine]],
    version: Version,
    debug: bool = False,
) -> Optional[MatchLine]:
    """
    Return objects representing the line as one of:

    * hammer_bounce-PlayedNote.
    * info(Attribute, Value).
    * insertion-PlayedNote.
    * ornament(Anchor)-PlayedNote.
    * ScoreNote-deletion.
    * ScoreNote-PlayedNote.
    * ScoreNote-trailing_score_note.
    * trailing_played_note-PlayedNote.
    * trill(Anchor)-PlayedNote.
    * meta(Attribute, Value, Bar, Beat).
    * sustain(Time, Value)
    * soft(Time, Value)

    or None if none can be matched

    Parameters
    ----------
    line : str
        Line of the match file
    from_matchline_methods : List[Callable[[str], MatchLine]]

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    matchline = None
    for from_matchline in from_matchline_methods:
        try:
            matchline = from_matchline(line, version=version)
            break
        except Exception as e:
            if not isinstance(e, MatchError):
                print(line, e, version)  # pragma: no cover
            continue

    return matchline


@deprecated_alias(fn="filename", create_part="create_score")
def load_matchfile(
    filename: PathLike,
) -> MatchFile:
    """
    Load a Matchfile as a `MatchFile` instance
    """

    if not os.path.exists(filename):
        raise ValueError("Filename does not exist")  # pragma: no cover

    with open(filename) as f:
        raw_lines = f.read().splitlines()

    version = get_version(raw_lines[0])

    from_matchline_methods = FROM_MATCHLINE_METHODSV1
    if version < Version(1, 0, 0):
        from_matchline_methods = FROM_MATCHLINE_METHODSV0

    parsed_lines = list()
    # Functionality to remove duplicate lines
    len_raw_lines = len(raw_lines)
    np_lines = np.array(raw_lines, dtype=str)
    # Remove empty lines
    np_lines = np_lines[np_lines != ""]
    # Remove duplicate lines
    _, idx = np.unique(np_lines, return_index=True)
    np_lines = np_lines[np.sort(idx)]
    # Parse lines
    f = partial(
        parse_matchline, version=version, from_matchline_methods=from_matchline_methods
    )
    f_vec = np.vectorize(f)
    parsed_lines_raw = f_vec(np_lines)
    # do not return unparseable lines
    parsed_lines = parsed_lines_raw[parsed_lines_raw != None].tolist()
    # Create MatchFile instance
    mf = MatchFile(lines=parsed_lines)
    # Validate match for duplicate snote_ids or pnote_ids
    validate_match_ids(mf)
    return mf


@deprecated_alias(fn="filename", create_part="create_score")
def load_match(
    filename: PathLike,
    create_score: bool = False,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
    offset_duration_whole: bool = True,
) -> Tuple[Union[Performance, list, Score]]:
    """
    Load a matchfile.

    Parameters
    ----------
    filename : str
        The matchfile
    create_score : bool, optional
        When True create a Part object from the snote information in
        the match file. Defaults to False.
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.
        Defaults to False.
    offset_duration_whole: Boolean, optional
        A flag for the type of offset and duration given in the matchfile.
        When true, the function expects the values to be given in whole
        notes (e.g. 1/4 for a quarter note) independet of time signature.
        Defaults to True.

    Returns
    -------
    performance : :class:partitura.performance.Performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    scr : :class:partitura.score.Score
        The score. This item is only returned when `create_score` = True.
    """
    # Parse Matchfile
    mf = load_matchfile(filename)

    # Generate PerformedPart
    ppart = performed_part_from_match(mf, pedal_threshold, first_note_at_zero)

    performance = Performance(
        id=get_document_name(filename),
        performedparts=ppart,
    )
    # Generate Part
    if create_score:
        spart = part_from_matchfile(
            mf,
            match_offset_duration_in_whole=offset_duration_whole,
        )

        scr = score.Score(id=get_document_name(filename), partlist=[spart])
    # Alignment
    alignment = alignment_from_matchfile(mf)

    if create_score:
        return performance, alignment, scr
    else:
        return performance, alignment


def note_alignment_from_matchfile(mf: MatchFile) -> List[dict]:
    """
    Get a note-level alignment from a MatchFile instance

    Parameters
    ----------
    mf : MatchFile
        A score-to-performance alignment

    Returns
    -------
    results : List[dict]
        An alignmnet as a list of dictionaries for each note.
    """
    result = []

    for line in mf.lines:
        if isinstance(line, BaseSnoteNoteLine):
            result.append(
                dict(
                    label="match",
                    score_id=str(line.snote.Anchor),
                    performance_id=format_pnote_id(line.note.Id),
                )
            )

        elif isinstance(
            line,
            BaseDeletionLine,
        ):
            if "leftOutTied" in line.snote.ScoreAttributesList:
                continue
            else:
                result.append(dict(label="deletion", score_id=str(line.snote.Anchor)))
        elif isinstance(
            line,
            BaseInsertionLine,
        ):
            result.append(
                dict(label="insertion", performance_id=format_pnote_id(line.note.Id))
            )
        elif isinstance(line, BaseOrnamentLine):
            if isinstance(line, MatchTrillNoteV0):
                ornament_type = "trill"
            elif isinstance(line, MatchOrnamentNoteV1):
                ornament_type = line.OrnamentType
            else:
                ornament_type = "generic_ornament"
            result.append(
                dict(
                    label="ornament",
                    score_id=str(line.Anchor),
                    performance_id=format_pnote_id(line.note.Id),
                    type=ornament_type,
                )
            )

    return result


# alias
alignment_from_matchfile = note_alignment_from_matchfile


def performed_part_from_match(
    mf: MatchFile,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
) -> PerformedPart:
    """
    Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    # Get midi time units
    mpq = mf.info("midiClockRate")  # 500000 -> microseconds per quarter
    ppq = mf.info("midiClockUnits")  # 500 -> parts per quarter

    # PerformedNote instances for all MatchNotes
    notes = []

    notes = list()
    note_onsets_in_secs = np.array(np.zeros(len(mf.notes)), dtype=float)
    note_onsets_in_tick = np.array(np.zeros(len(mf.notes)), dtype=int)
    for i, note in enumerate(mf.notes):
        n_onset_sec = midi_ticks_to_seconds(note.Onset, mpq, ppq)
        note_onsets_in_secs[i] = n_onset_sec
        note_onsets_in_tick[i] = note.Onset
        notes.append(
            dict(
                id=format_pnote_id(note.Id),
                midi_pitch=note.MidiPitch,
                note_on=n_onset_sec,
                note_off=midi_ticks_to_seconds(note.Offset, mpq, ppq),
                note_on_tick=note.Onset,
                note_off_tick=note.Offset,
                sound_off=midi_ticks_to_seconds(note.Offset, mpq, ppq),
                velocity=note.Velocity,
                track=getattr(note, "Track", 0),
                channel=getattr(note, "Channel", 0),
            )
        )
    # Set first note_on to zero in ticks and seconds if first_note_at_zero
    if first_note_at_zero and len(note_onsets_in_secs) > 0:
        offset = note_onsets_in_secs.min()
        offset_tick = note_onsets_in_tick.min()
        if offset > 0 and offset_tick > 0:
            for note in notes:
                note["note_on"] -= offset
                note["note_off"] -= offset
                note["sound_off"] -= offset
                note["note_on_tick"] -= offset_tick
                note["note_off_tick"] -= offset_tick

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = [
        dict(
            number=64,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
        )
        for ped in mf.sustain_pedal
    ]

    # SoftPedal instances for soft pedal lines
    soft_pedal = [
        dict(
            number=67,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
        )
        for ped in mf.soft_pedal
    ]

    # Make performed part
    ppart = PerformedPart(
        id="P1",
        part_name=mf.info("piece"),
        notes=notes,
        controls=sustain_pedal + soft_pedal,
        sustain_pedal_threshold=pedal_threshold,
    )
    return ppart


def sort_snotes(snotes: List[BaseSnoteLine]) -> List[BaseSnoteLine]:
    """
    Sort s(core)notes.

    Parameters
    ----------
    snotes : list
        The score notes

    Returns
    -------
    snotes_sorted : list
        The sorted score notes
    """
    sidx = np.lexsort(
        list(zip(*[(float(n.Offset), float(n.Beat), float(n.Measure)) for n in snotes]))
    )
    return [snotes[i] for i in sidx if snotes[i].NoteName.lower() != "r"]


def part_from_matchfile(
    mf: MatchFile,
    match_offset_duration_in_whole: bool = True,
) -> Part:
    """
    Create a score part from a matchfile.

    Parameters
    ----------
    mf : MatchFile
        An instance of `MatchFile`

    match_offset_duration_in_whole: Boolean
        A flag for the type of offset and duration given in the matchfile.
        When true, the function expects the values to be given in whole
        notes (e.g. 1/4 for a quarter note) independet of time signature.


    Returns
    -------
    part : partitura.score.Part
        An instance of `Part` containing score information.

    """
    part = score.Part("P1", mf.info("piece"))
    snotes = sort_snotes(mf.snotes)

    ts = mf.time_signatures
    min_time = snotes[0].OnsetInBeats  # sorted by OnsetInBeats
    max_time = max(n.OffsetInBeats for n in snotes)
    (
        beats_map_from_beats,
        beats_map,
        beat_type_map_from_beats,
        beat_type_map,
        min_time_q,
        max_time_q,
    ) = make_timesig_maps(ts, max_time)

    # compute necessary divs based on the types of notes in the
    # match snotes (only integers)
    divs_arg = [
        max(int((beat_type_map(note.OnsetInBeats) / 4)), 1)
        * note.Offset.denominator
        * (note.Offset.tuple_div or 1)
        for note in snotes
    ]
    divs_arg += [
        max(int((beat_type_map(note.OnsetInBeats) / 4)), 1)
        * note.Duration.denominator
        * (note.Duration.tuple_div or 1)
        for note in snotes
    ]

    onset_in_beats = np.array([note.OnsetInBeats for note in snotes])
    unique_onsets, inv_idxs = np.unique(onset_in_beats, return_inverse=True)

    iois_in_beats = np.diff(unique_onsets)
    beat_to_quarter = 4 / beat_type_map(onset_in_beats)

    iois_in_quarters_offset = np.r_[
        beat_to_quarter[0] * onset_in_beats[0],
        (4 / beat_type_map(unique_onsets[:-1])) * iois_in_beats,
    ]
    onset_in_quarters = np.cumsum(iois_in_quarters_offset)
    iois_in_quarters = np.diff(onset_in_quarters)

    # ___ these divs are relative to quarters;
    divs = np.lcm.reduce(np.unique(divs_arg))
    onset_in_divs = np.r_[0, np.cumsum(divs * iois_in_quarters)][inv_idxs]
    onset_in_quarters = onset_in_quarters[inv_idxs]

    part.set_quarter_duration(0, divs)
    bars = np.unique([n.Measure for n in snotes])
    t = min_time
    t = t * 4 / beat_type_map(min_time)
    offset = t
    bar_times = {}

    if t > 0:
        # if we have an incomplete first measure that isn't an anacrusis
        # measure, add a rest (dummy)
        # if starting beat is above zero, add padding
        rest = score.Rest()
        part.add(rest, start=0, end=t * divs)
        onset_in_divs += t * divs
        offset = 0
        t = t - t % beats_map(min_time)

    for b_name in bars:
        notes_in_this_bar = [
            (ni, n) for ni, n in enumerate(snotes) if n.Measure == b_name
        ]
        a_note_in_this_bar = notes_in_this_bar[0][1]
        a_note_id_in_this_bar = notes_in_this_bar[0][0]
        bar_offset = (
            (a_note_in_this_bar.Beat - 1)
            * 4
            / beat_type_map_from_beats(a_note_in_this_bar.OnsetInBeats)
        )
        on_off_scale = 1
        if not match_offset_duration_in_whole:
            on_off_scale = beat_type_map_from_beats(a_note_in_this_bar.OnsetInBeats)
        beat_offset = (
            4
            / on_off_scale
            * a_note_in_this_bar.Offset.numerator
            / (
                a_note_in_this_bar.Offset.denominator
                * (a_note_in_this_bar.Offset.tuple_div or 1)
            )
        )

        barline_in_quarters = (
            onset_in_quarters[a_note_id_in_this_bar] - bar_offset - beat_offset
        )
        bar_times[b_name] = barline_in_quarters

    for ni, note in enumerate(snotes):
        # start of bar in quarter units
        bar_start = bar_times[note.Measure]

        on_off_scale = 1
        # on_off_scale = 1 means duration and beat offset are given in
        # whole notes, else they're given in beats (as in the KAIST data)
        if not match_offset_duration_in_whole:
            on_off_scale = beat_type_map(bar_start)

        # offset within bar in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map(bar_start)
        bar_offset = (note.Beat - 1) * 4 / beat_type_map(bar_start)

        # offset within beat in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map(bar_start)
        beat_offset = (
            4
            / on_off_scale
            * note.Offset.numerator
            / (note.Offset.denominator * (note.Offset.tuple_div or 1))
        )

        # convert the onset time in quarters (0 at first barline) to onset
        # time in divs (0 at first note)
        onset_divs = int(round(divs * (bar_start + bar_offset + beat_offset - offset)))

        if not np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01):
            warnings.warn(
                "Calculated `onset_divs` does not match `OnsetInBeats` " "information!."
            )
            onset_divs = onset_in_divs[ni]
        assert onset_divs >= 0
        assert np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01)
        is_tied = False
        articulations = set()
        if "staccato" in note.ScoreAttributesList or "stac" in note.ScoreAttributesList:
            articulations.add("staccato")
        if "accent" in note.ScoreAttributesList:
            articulations.add("accent")
        if "leftOutTied" in note.ScoreAttributesList:
            is_tied = True

        # dictionary with keyword args with which the Note
        # (or GraceNote) will be instantiated
        note_attributes = dict(
            step=note.NoteName,
            octave=note.Octave,
            alter=note.Modifier,
            id=note.Anchor,
            articulations=articulations,
            technical=[],
        )

        staff_nr = next(
            (a[-1] for a in note.ScoreAttributesList if a.startswith("staff")), None
        )
        try:
            note_attributes["staff"] = int(staff_nr)
        except (TypeError, ValueError):
            # no staff attribute, or staff attribute does not end with a number
            note_attributes["staff"] = None

        if "s" in note.ScoreAttributesList:
            note_attributes["voice"] = 1
        elif any(a.startswith("v") for a in note.ScoreAttributesList):
            note_attributes["voice"] = next(
                (
                    int(a[1:])
                    for a in note.ScoreAttributesList
                    if vnumber_pattern.match(a)
                ),
                None,
            )
        else:
            note_attributes["voice"] = next(
                (int(a) for a in note.ScoreAttributesList if number_pattern.match(a)),
                None,
            )

        if any(a.startswith("fingering") for a in note.ScoreAttributesList):
            note_attributes["technical"].append(
                next(
                    (
                        score.Fingering(int(a[9:]))
                        for a in note.ScoreAttributesList
                        if fingering_pattern.match(a)
                    ),
                    None,
                )
            )

        # get rid of this if as soon as we have a way to iterate over the
        # duration components. For now we have to treat the cases simple
        # and compound durations separately.
        if note.Duration.add_components:
            prev_part_note = None

            for i, (num, den, tuple_div) in enumerate(note.Duration.add_components):
                # when we add multiple notes that are tied, the first note will
                # get the original note id, and subsequent notes will get a
                # derived note id (by appending, 'a', 'b', 'c',...)
                if i > 0:
                    # tnote_id = 'n{}_{}'.format(note.Anchor, i)
                    note_attributes["id"] = score._make_tied_note_id(
                        note_attributes["id"]
                    )

                part_note = score.Note(**note_attributes)

                # duration_divs from local beats --> 4/beat_type_map(bar_start)

                duration_divs = int(
                    (4 / on_off_scale) * divs * num / (den * (tuple_div or 1))
                )

                assert duration_divs > 0
                offset_divs = onset_divs + duration_divs
                part.add(part_note, onset_divs, offset_divs)

                if prev_part_note:
                    prev_part_note.tie_next = part_note
                    part_note.tie_prev = prev_part_note
                prev_part_note = part_note
                onset_divs = offset_divs

        else:
            num = note.Duration.numerator
            den = note.Duration.denominator
            tuple_div = note.Duration.tuple_div

            # duration_divs from local beats --> 4/beat_type_map(bar_start)
            duration_divs = int(
                divs * 4 / on_off_scale * num / (den * (tuple_div or 1))
            )
            offset_divs = onset_divs + duration_divs

            # notes with duration 0, are also treated as grace notes, even if
            # they do not have a 'grace' score attribute
            if "grace" in note.ScoreAttributesList or note.Duration.numerator == 0:
                part_note = score.GraceNote(
                    grace_type="appoggiatura", **note_attributes
                )

            else:
                part_note = score.Note(**note_attributes)
            part.add(part_note, onset_divs, offset_divs)
        # Check if the note is tied and if so, add the tie information
        if is_tied:
            found = False
            # iterate over all notes in the Timeline that end at the starting point.
            for el in part_note.start.iter_ending(score.Note):
                if isinstance(el, score.Note):
                    condition = (
                        el.step == note_attributes["step"]
                        and el.octave == note_attributes["octave"]
                        and el.alter == note_attributes["alter"]
                    )
                    if condition:
                        el.tie_next = part_note
                        part_note.tie_prev = el
                        found = True
                        break
            if not found:
                warnings.warn(
                    "Tie information found, but no previous note found to tie to for note {}.".format(
                        part_note.id
                    )
                )
    # add time signatures
    for ts_beat_time, ts_bar, tsg in ts:
        ts_beats = tsg.numerator
        ts_beat_type = tsg.denominator
        # check if time signature is in a known measure (from notes)
        if ts_bar in bar_times.keys():
            bar_start_divs = int(divs * (bar_times[ts_bar] - offset))  # in quarters
            bar_start_divs = max(0, bar_start_divs)
        else:
            bar_start_divs = 0
        part.add(score.TimeSignature(ts_beats, ts_beat_type), bar_start_divs)
    # add key signatures
    for ks_beat_time, ks_bar, keys in mf.key_signatures:
        if ks_bar in bar_times.keys():
            bar_start_divs = int(divs * (bar_times[ks_bar] - offset))  # in quarters
            bar_start_divs = max(0, bar_start_divs)
        else:
            bar_start_divs = 0

        # TODO
        # * use key estimation if there are multiple defined keys
        # fifths, mode = key_name_to_fifths_mode(key_name)
        part.add(score.KeySignature(keys.fifths, keys.mode), ks_bar)

    add_staffs(part)
    # add_clefs(part)

    prev_measure = None
    for measure_counter, measure_name in enumerate(bar_times.keys()):
        barline_in_quarters = bar_times[measure_name]
        barline_in_divs = int(round(divs * (barline_in_quarters - offset)))
        if barline_in_divs < 0:
            barline_in_divs = 0
        if prev_measure is not None:
            part.add(prev_measure, None, barline_in_divs)
        prev_measure = score.Measure(number=measure_counter + 1, name=str(measure_name))
        part.add(prev_measure, barline_in_divs)
    last_closing_barline = barline_in_divs + int(
        round(
            divs
            * beats_map(barline_in_quarters)
            * 4
            / beat_type_map(barline_in_quarters)
        )
    )
    part.add(prev_measure, None, last_closing_barline)

    # add the rest of the measures automatically
    score.add_measures(part)
    score.tie_notes(part)
    score.find_tuplets(part)

    n_voices = set([n.voice for n in part.notes])
    if len(n_voices) == 1 and None in n_voices:
        for note in part.notes_tied:
            if note.voice is None:
                note.voice = 1
    elif len(n_voices) > 1 and None in n_voices:
        n_voices.remove(None)
        for note in part.notes_tied:
            if note.voice is None:
                note.voice = max(n_voices) + 1

    return part


def make_timesig_maps(
    ts_orig: List[Tuple[float, int, MatchTimeSignature]],
    max_time: float,
) -> Tuple[Callable, Callable, Callable, Callable, float, float]:
    """
    Create time signature (interpolation) maps

    Parameters
    ----------
    ts_orig : List[Tuple[float, int, MatchTimeSignature]]
        A list of tuples containing position in beats, measure and
        MatchTimeSignature instances
    max_time : float
        Maximal time of the time signatures

    Returns
    -------
    beats_map: callable
    qbeats_map: callable
    beat_type_map: callable
    qbeat_type_map: callable
    start_q: float
    end_q: float
    """
    # TODO: make sure that ts_orig covers range from min_time
    # return two functions that map score times (in quarter units) to time sig
    # beats, and time sig beat_type respectively
    ts = list(ts_orig)
    assert len(ts) > 0
    ts.append((max_time, None, ts[-1][2]))

    x = np.array([t for t, _, _ in ts])
    y = np.array([(x.numerator, x.denominator) for _, _, x in ts])

    start_q = x[0] * 4 / y[0, 1]
    x_q = np.cumsum(np.r_[start_q, 4 * np.diff(x) / y[:-1, 1]])
    end_q = x_q[-1]

    # TODO: fix error with bounds
    qbeats_map = interp1d(
        x_q,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    qbeat_type_map = interp1d(
        x_q,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )
    beats_map = interp1d(
        x,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    beat_type_map = interp1d(
        x,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )

    return beats_map, qbeats_map, beat_type_map, qbeat_type_map, start_q, end_q


def add_staffs(part: Part, split: int = 55, only_missing: bool = True) -> None:
    """
    Method to add staff information to a part

    Parameters
    ----------
    part: Part
        Part to add staff information to.
    split: int
        MIDI pitch to split staff into upper and lower. Default is 55
    only_missing: bool
        If True, only add staff to those notes that do not have staff info already.
    """
    # assign staffs using a hard limit
    notes = part.notes_tied
    for n in notes:
        if only_missing and n.staff:
            continue

        if n.midi_pitch > split:
            staff = 1
        else:
            staff = 2

        n.staff = staff

        n_tied = n.tie_next
        while n_tied:
            n_tied.staff = staff
            n_tied = n_tied.tie_next

    part.add(score.Clef(staff=1, sign="G", line=2, octave_change=0), 0)
    part.add(score.Clef(staff=2, sign="F", line=4, octave_change=0), 0)


def validate_match_ids(mf):
    """
    Check a matchfile for duplicate snote and note IDs.

    This function will:
    - remove all deletions with a score ID that occurs in multiple lines.
    - remove all insertions with a performance ID that occurs in multiple lines.

    Handles cases with conflicting match/insertion(s) and match/deletion(s) tuples
    with any number of insertions or deletions and a single match by keeping
    only the match.

    Unhandled cases:
    - multiple conflicting matches: all are kept.
    - multiple insertions with the same ID: all are deleted.
    - multiple deletions with the same ID: all are deleted.

    Parameters
    ----------
    mf: MatchFile
        MatchFile to validate

    Returns
    -------
    Updates the representation of the matchfile by removing match lines.
    """

    # Check if the matchfile is valid (i.e. check for snote duplicates)
    sids = np.array([n.Anchor for n in mf.snotes])
    # First check if score ids are unique
    sids_unique, counts = np.unique(sids, return_counts=True)
    sids_to_check = sids_unique[np.where(counts > 1)[0]]
    if len(sids_to_check) > 0:
        indices_to_del = []
        for i, line in enumerate(mf.lines):
            if isinstance(line, BaseDeletionLine):
                if line.Anchor in sids_to_check:
                    indices_to_del.append(i)
        warnings.warn(
            "Matchfile contains duplicate score notes. "
            "Removing {} deletions.".format(len(indices_to_del))
        )
        mf.lines = np.delete(mf.lines, indices_to_del)

    # Check if the matchfile is valid (i.e. check for performance note duplicates)
    pids = np.array([n.Id for n in mf.notes])
    pids_unique, counts = np.unique(pids, return_counts=True)
    pids_to_check = pids_unique[np.where(counts > 1)[0]]
    if len(pids_to_check) > 0:
        indices_to_del = []
        for i, line in enumerate(mf.lines):
            if isinstance(line, BaseInsertionLine):
                if line.Id in pids_to_check:
                    indices_to_del.append(i)
        warnings.warn(
            "Matchfile contains duplicate performance notes. "
            "Removing {} insertions.".format(len(indices_to_del))
        )
        mf.lines = np.delete(mf.lines, indices_to_del)


if __name__ == "__main__":
    pass
