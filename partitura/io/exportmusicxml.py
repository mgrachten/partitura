#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting MusicXML files.
"""
import math
from collections import defaultdict
from lxml import etree
import partitura.score as score
from operator import itemgetter
from typing import Optional

from .importmusicxml import DYN_DIRECTIONS, PEDAL_DIRECTIONS
from partitura.utils import partition, iter_current_next, to_quarter_tempo

from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_musicxml"]

DOCTYPE = """<!DOCTYPE score-partwise PUBLIC\n  "-//Recordare//DTD MusicXML 3.1 Partwise//EN"\n  "http://www.musicxml.org/dtds/partwise.dtd">"""  # noqa: E501
MEASURE_SEP_COMMENT = "======================================================="
ARTICULATIONS = [
    "accent",
    "breath-mark",
    "caesura",
    "detached-legato",
    "doit",
    "falloff",
    "plop",
    "scoop",
    "spiccato",
    "staccatissimo",
    "staccato",
    "stress",
    "strong-accent",
    "tenuto",
    "unstress",
]


def range_number_from_counter(e, label, counter):
    key = (label, e)
    number = counter.get(key, None)

    if number is None:
        number = 1 + sum(1 for o in counter.keys() if o[0] == label)
        assert number is not None
        counter[key] = number

    else:
        del counter[key]

    return number


def filter_string(s):
    """
    Make (unicode) string fit for passing it to lxml, which means (at least)
    removing null characters.
    """
    return s.replace("\x00", "")


def make_note_el(note, dur, voice, counter, n_of_staves):
    # child order
    # <grace> | <chord> | <cue>
    # <pitch>
    # <duration>
    # <tie type="stop"/>
    # <voice>
    # <type>
    # <notations>

    note_e = etree.Element("note")

    if note.id is not None:
        note_id = note.id
        # make sure note_id is unique by appending _x to the note_id for the
        # x-th repetition of the id
        counter[note_id] = counter.get(note_id, 0) + 1

        if counter[note_id] > 1:
            note_id += "_{}".format(counter[note_id])

        note_e.attrib["id"] = filter_string(note_id)

    if isinstance(note, score.Note):
        if isinstance(note, score.GraceNote):
            if note.grace_type == "acciaccatura":
                etree.SubElement(note_e, "grace", slash="yes")

            else:
                etree.SubElement(note_e, "grace")

        pitch_e = etree.SubElement(note_e, "pitch")

        etree.SubElement(pitch_e, "step").text = "{}".format(note.step)

        if note.alter not in (None, 0):
            etree.SubElement(pitch_e, "alter").text = "{}".format(note.alter)

        etree.SubElement(pitch_e, "octave").text = "{}".format(note.octave)

    elif isinstance(note, score.UnpitchedNote):
        unpitch_e = etree.SubElement(note_e, "unpitched")

        etree.SubElement(unpitch_e, "display-step").text = "{}".format(note.step)

        etree.SubElement(unpitch_e, "display-octave").text = "{}".format(note.octave)

        if note.notehead is not None:
            nh_e = etree.SubElement(note_e, "notehead")
            nh_e.text = "{}".format(note.notehead)
            if note.noteheadstyle:
                nh_e.attrib["filled"] = "yes"
            else:
                nh_e.attrib["filled"] = "no"

    elif isinstance(note, score.Rest):
        if not note.hidden:
            etree.SubElement(note_e, "rest")

    if not isinstance(note, score.GraceNote):
        duration_e = etree.SubElement(note_e, "duration")
        duration_e.text = "{:d}".format(int(dur))

    notations = []

    if note.tie_prev is not None:
        etree.SubElement(note_e, "tie", type="stop")
        notations.append(etree.Element("tied", type="stop"))

    if note.tie_next is not None:
        etree.SubElement(note_e, "tie", type="start")
        notations.append(etree.Element("tied", type="start"))

    if voice not in (None, 0):
        etree.SubElement(note_e, "voice").text = "{}".format(voice)

    if note.stem_direction is not None:
        stem_e = etree.SubElement(note_e, "stem")
        stem_e.text = note.stem_direction

    if note.fermata is not None:
        notations.append(etree.Element("fermata"))

    if note.articulations:
        articulations = []
        for articulation in note.articulations:
            if articulation in ARTICULATIONS:
                articulations.append(etree.Element(articulation))
        if articulations:
            articulations_e = etree.Element("articulations")
            articulations_e.extend(articulations)
            notations.append(articulations_e)

    if note.technical:
        technical = []
        for technical_notation in note.technical:
            if isinstance(technical_notation, score.Fingering):
                tech_el = etree.Element("fingering")
                tech_el.text = str(technical_notation.fingering)
                technical.append(tech_el)

        if technical:
            technical_e = etree.Element("technical")
            technical_e.extend(technical)
            notations.append(technical_e)

    sym_dur = note.symbolic_duration or {}

    if sym_dur.get("type") is not None:
        etree.SubElement(note_e, "type").text = sym_dur["type"]

    for i in range(sym_dur.get("dots", 0)):
        etree.SubElement(note_e, "dot")

    if (
        sym_dur.get("actual_notes") is not None
        and sym_dur.get("normal_notes") is not None
    ):
        time_mod_e = etree.SubElement(note_e, "time-modification")
        actual_e = etree.SubElement(time_mod_e, "actual-notes")
        actual_e.text = str(sym_dur["actual_notes"])
        normal_e = etree.SubElement(time_mod_e, "normal-notes")
        normal_e.text = str(sym_dur["normal_notes"])

    if note.staff is not None:
        if note.staff != 1 or n_of_staves > 1:
            etree.SubElement(note_e, "staff").text = "{}".format(note.staff)

    for slur in note.slur_stops:
        number = range_number_from_counter(slur, "slur", counter)

        notations.append(etree.Element("slur", number="{}".format(number), type="stop"))

    for slur in note.slur_starts:
        number = range_number_from_counter(slur, "slur", counter)

        notations.append(
            etree.Element("slur", number="{}".format(number), type="start")
        )

    for tuplet in note.tuplet_stops:
        tuplet_key = ("tuplet", tuplet)
        number = counter.get(tuplet_key, None)

        if number is None:
            number = 1
            counter[tuplet_key] = number

        else:
            del counter[tuplet_key]

        notations.append(
            etree.Element("tuplet", number="{}".format(number), type="stop")
        )

    for tuplet in note.tuplet_starts:
        tuplet_key = ("tuplet", tuplet)
        number = counter.get(tuplet_key, None)

        if number is None:
            number = 1 + sum(1 for o in counter.keys() if o[0] == "tuplet")
            counter[tuplet_key] = number

        else:
            del counter[tuplet_key]

        tuplet_e = etree.Element("tuplet", number="{}".format(number), type="start")
        if tuplet.actual_notes is not None and tuplet.normal_notes is not None and tuplet.type is not None:
            # tuplet-actual tag
            tuplet_actual_e = etree.SubElement(tuplet_e, "tuplet-actual")
            tuplet_actual_notes_e = etree.SubElement(tuplet_actual_e, "tuplet-number")
            tuplet_actual_notes_e.text = str(tuplet.actual_notes)
            tuplet_actual_type_e = etree.SubElement(tuplet_actual_e, "tuplet-type")
            tuplet_actual_type_e.text = str(tuplet.type)
            # tuplet-normal tag
            tuplet_normal_e = etree.SubElement(tuplet_e, "tuplet-normal")
            tuplet_normal_notes_e = etree.SubElement(tuplet_normal_e, "tuplet-number")
            tuplet_normal_notes_e.text = str(tuplet.normal_notes)
            tuplet_normal_type_e = etree.SubElement(tuplet_normal_e, "tuplet-type")
            tuplet_normal_type_e.text = str(tuplet.type)
        notations.append(tuplet_e)

    if notations:
        notations_e = etree.SubElement(note_e, "notations")
        notations_e.extend(notations)

    return note_e


def do_note(note, measure_end, part, voice, counter, n_of_staves):
    if isinstance(note, score.GraceNote):
        dur_divs = 0

    else:
        dur_divs = note.end.t - note.start.t

    note_e = make_note_el(note, dur_divs, voice, counter, n_of_staves)

    return (note.start.t, dur_divs, note_e)


def linearize_measure_contents(part, start, end, state):
    """
    Determine the document order of events starting between `start` (inclusive)
    and `end` (exlusive).  (notes, directions, divisions, time signatures). This
    function finds any mid-measure attribute/divisions and splits up the measure
    into segments by divisions, to be linearized separately and
    concatenated. The actual linearization is done by
    the `linearize_segment_contents` function.

    Parameters
    ----------
    start: score.TimePoint
        start
    end: score.TimePoint
        end
    part: score.Part

    Returns
    -------
    list
        The contents of measure in document order
    """
    splits = [start]
    q_times = part.quarter_durations(start.t, end.t)
    if len(q_times) > 0:
        quarter = start.quarter
        tp = start.next
        while tp and tp != end:
            if tp.quarter != quarter:
                splits.append(tp)
                quarter = tp.quarter
            tp = tp.next

    splits.append(end)
    contents = []

    for i in range(1, len(splits)):
        contents.extend(
            linearize_segment_contents(part, splits[i - 1], splits[i], state)
        )

    return contents


def remove_voice_polyphony_single(notes, voice_spans):
    """
    Test wether a list of notes satisfies the MusicXML constraints on voices that:
    - all notes starting at the same time have the same duration
    - no <backup> is required to specify the voice in document order
    whenever a note violates the constraints change its voice
    (choosing a new voice that is not currently in use)

    Parameters
    ----------
    notes: list
        List of notes in a voice

    Returns
    -------
    type
        Description of return value
    """

    extraneous = defaultdict(list)

    by_onset = defaultdict(list)
    for note in notes:
        if not isinstance(note, score.GraceNote):
            by_onset[note.start.t].append(note)
    onsets = sorted(by_onset.keys())

    for o in onsets:
        chord_dur = min(n.duration for n in by_onset[o])

        for n in by_onset[o]:
            if n.duration > chord_dur:
                voice = find_free_voice(voice_spans, n.start.t, n.end.t)
                voice_spans.append((n.start.t, n.end.t, voice))
                extraneous[voice].append(n)
                notes.remove(n)

    # now remove any notes that exceed next onset
    by_onset = defaultdict(list)
    for note in notes:
        by_onset[note.start.t].append(note)
    onsets = sorted(by_onset.keys())

    for o1, o2 in iter_current_next(onsets):
        for n in by_onset[o1]:
            if o1 + n.duration > o2:
                voice = find_free_voice(voice_spans, n.start.t, n.end.t)
                voice_spans.append((n.start.t, n.end.t, voice))
                extraneous[voice].append(n)
                notes.remove(n)

    return extraneous


def find_free_voice(voice_spans, start, end):
    free_voice = min(voice for _, _, voice in voice_spans) + 1

    for vstart, vend, voice in voice_spans:
        if (end > vstart) and (start < vend):
            free_voice = max(free_voice, voice + 1)

    return free_voice


def remove_voice_polyphony(notes_by_voice):
    voice_spans = [(-math.inf, math.inf, max(notes_by_voice.keys()))]
    extraneous = defaultdict(list)
    # n_orig = sum(len(nn) for nn in notes_by_voice.values())

    for voice, vnotes in notes_by_voice.items():
        v_extr = remove_voice_polyphony_single(vnotes, voice_spans)

        for new_voice, new_vnotes in v_extr.items():
            extraneous[new_voice].extend(new_vnotes)

    # n_1 = sum(len(nn) for nn in notes_by_voice.values())
    # n_2 = sum(len(nn) for nn in extraneous.values())
    # n_new = n_1 + n_2
    # assert n_orig == n_new
    # assert len(set(notes_by_voice.keys()).intersection(set(extraneous.keys()))) == 0
    for v, vnotes in extraneous.items():
        notes_by_voice[v] = vnotes


# def fill_gaps_with_rests(notes_by_voice, start, end, part):
#     for voice, notes in notes_by_voice.items():
#         if len(notes) == 0:
#             rest = score.Rest(voice=voice or None)
#             part.add(rest, start.t, end.t)
#         else:
#             t = start.t
#             for note in notes:
#                 if note.start.t > t:
#                     rest = score.Rest(voice=voice or None)
#                     part.add(rest, t, note.start.t)
#                 t = note.end.t
#             if note.end.t < end.t:
#                 rest = score.Rest(voice=voice or None)
#                 part.add(rest, note.end.t, end.t)


def linearize_segment_contents(part, start, end, state):
    """
    Determine the document order of events starting between `start` (inclusive)
    and `end` (exlusive).
    (notes, directions, divisions, time signatures).
    """

    notes = part.iter_all(
        score.GenericNote, start=start, end=end, include_subclasses=True
    )

    notes_by_voice = partition(lambda n: n.voice or 0, notes)
    if len(notes_by_voice) == 0:
        # if there are no notes in this segment, we add a rest
        # NOTE: altering the part instance while exporting is bad!
        # rest = score.Rest()
        # part.add(start.t, rest, end.t)
        # notes_by_voice = {0: [rest]}
        notes_by_voice[None] = []

    # make sure there is no polyphony within voices by assigning any violating
    # notes to a new (free) voice.
    remove_voice_polyphony(notes_by_voice)

    # fill_gaps_with_rests(notes_by_voice, start, end, part)
    # # redo
    # notes = part.iter_all(score.GenericNote,
    #                               start=start, end=end,
    #                               include_subclasses=True)
    # notes_by_voice = partition(lambda n: n.voice or 0, notes)

    voices_e = defaultdict(list)

    for voice in sorted(notes_by_voice.keys()):
        voice_notes = notes_by_voice[voice]
        # sort by pitch (then step in case of enharmonic notes)
        voice_notes.sort(
            key=lambda n: (
                (n.midi_pitch, n.step) if hasattr(n, "midi_pitch") else (-1, "")
            ),
            reverse=True,
        )
        # grace notes should precede other notes at the same onset
        voice_notes.sort(key=lambda n: not isinstance(n, score.GraceNote))
        # voice_notes.sort(key=lambda n: -n.duration)
        voice_notes.sort(key=lambda n: n.start.t)

        n_of_staves = part.number_of_staves

        for n in voice_notes:
            if isinstance(n, score.GraceNote):
                # check if it is the first in its sequence
                if not n.grace_prev:
                    # if so we add the whole grace sequence at once to ensure
                    # the correct order
                    for m in n.iter_grace_seq():
                        note_e = do_note(
                            m, end.t, part, voice, state["note_id_counter"], n_of_staves
                        )
                        voices_e[voice].append(note_e)
            else:
                note_e = do_note(
                    n, end.t, part, voice, state["note_id_counter"], n_of_staves
                )
                voices_e[voice].append(note_e)

        add_chord_tags(voices_e[voice])

    harmony_e = do_harmony(part, start, end)
    attributes_e = do_attributes(part, start, end)
    directions_e = do_directions(part, start, end, state["range_counter"])
    prints_e = do_prints(part, start, end)
    barline_e = do_barlines(part, start, end)

    other_e = harmony_e + attributes_e + directions_e + barline_e + prints_e

    contents = merge_measure_contents(voices_e, other_e, start.t)

    return contents


def do_prints(part, start, end):
    pages = part.iter_all(score.Page, start, end)
    systems = part.iter_all(score.System, start, end)
    by_onset = defaultdict(dict)
    for page in pages:
        by_onset[page.start.t]["new-page"] = "yes"
    for system in systems:
        by_onset[system.start.t]["new-system"] = "yes"
    result = []
    for onset, attrs in by_onset.items():
        result.append((onset, None, etree.Element("print", **attrs)))
    return result


def do_barlines(part, start, end):
    # all fermata that are not linked to a note (fermata at time end may be part
    # of the current or the next measure, depending on the location attribute
    # (which is stored in fermata.ref)).
    fermata = [
        ferm
        for ferm in part.iter_all(score.Fermata, start, end)
        if ferm.ref in (None, "left", "middle", "right")
    ] + [
        ferm
        for ferm in part.iter_all(score.Fermata, end, end.next)
        if ferm.ref in (None, "right")
    ]
    repeat_start = part.iter_all(score.Repeat, start, end)
    repeat_end = part.iter_all(score.Repeat, start.next, end.next, mode="ending")
    ending_start = part.iter_all(score.Ending, start, end)
    ending_end = part.iter_all(score.Ending, start.next, end.next, mode="ending")
    by_onset = defaultdict(list)

    for obj in fermata:
        by_onset[obj.start.t].append(etree.Element("fermata"))

    for obj in repeat_start:
        if obj.start is not None:
            by_onset[obj.start.t].append(etree.Element("repeat", direction="forward"))

    for obj in ending_start:
        if obj.start is not None:
            by_onset[obj.start.t].append(
                etree.Element("ending", type="start", number=str(obj.number))
            )

    for obj in repeat_end:
        if obj.end is not None:
            by_onset[obj.end.t].append(etree.Element("repeat", direction="backward"))

    for obj in ending_end:
        if obj.end is not None:
            by_onset[obj.end.t].append(
                etree.Element("ending", type="stop", number=str(obj.number))
            )

    result = []

    for onset in sorted(by_onset.keys()):
        attrib = {}

        if onset == start.t:
            attrib["location"] = "left"

        elif onset == end.t:
            attrib["location"] = "right"

        else:
            attrib["location"] = "middle"

        barline_e = etree.Element("barline", **attrib)

        barline_e.extend(by_onset[onset])
        result.append((onset, None, barline_e))

    return result


def add_chord_tags(notes):
    prev_dur = None
    prev = None
    for onset, dur, note in notes:
        if onset == prev:
            if dur == prev_dur:
                note.insert(0, etree.Element("chord"))

        if any(e.tag == "grace" for e in note):
            # if note is a grace note we don't want to trigger a chord for the
            # next note
            prev = None
        else:
            prev = onset
            prev_dur = dur


def forward_backup_if_needed(t, t_prev):
    result = []
    gap = 0

    if t > t_prev:
        gap = t - t_prev
        e = etree.Element("forward")
        ee = etree.SubElement(e, "duration")
        ee.text = "{:d}".format(int(gap))
        result.append((t_prev, gap, e))

    elif t < t_prev:
        gap = t_prev - t
        e = etree.Element("backup")
        ee = etree.SubElement(e, "duration")
        ee.text = "{:d}".format(int(gap))
        result.append((t_prev, -gap, e))

    return result, gap


def merge_with_voice(notes, other, measure_start):
    by_onset = defaultdict(list)

    for onset, dur, el in notes:
        by_onset[onset].append((dur, el))

    for onset, dur, el in other:
        by_onset[onset].append((dur, el))

    result = []
    last_t = measure_start
    fb_cost = 0
    # order to insert simultaneously starting elements; it is important to put
    # notes last, since they update the position, and thus would lead to
    # needless backup/forward insertions
    order = {
        "barline": 0,
        "attributes": 1,
        "direction": 2,
        "print": 3,
        "sound": 4,
        "harmony": 5,
        "note": 6,
    }
    last_note_onset = measure_start

    for onset in sorted(by_onset.keys()):
        elems = by_onset[onset]
        elems.sort(key=lambda x: order.get(x[1].tag, len(order)))

        for dur, el in elems:
            if el.tag == "note":
                if el.find("chord") is not None:
                    last_t = last_note_onset

                last_note_onset = onset

            els, cost = forward_backup_if_needed(onset, last_t)
            fb_cost += cost
            result.extend(els)
            result.append((onset, dur, el))
            last_t = onset + (dur or 0)

    return result, fb_cost


def merge_measure_contents(notes, other, measure_start):
    merged = {}
    # cost (measured as the total forward/backup jumps needed to merge) all
    # elements in `other` into each voice
    cost = {}

    for voice in sorted(notes.keys()):
        # merge `other` with each voice, and keep track of the cost
        merged[voice], cost[voice] = merge_with_voice(
            notes[voice], other, measure_start
        )

    if not merged:
        merged[0] = []
        cost[0] = 0

    # CHANGE: disabled cost-based merging of non-note elements into stream
    # because this led to attributes not being in the beginning of the measure,
    # which in turn led to problems with musescore
    # fix: add atributes first, then the notes.
    # problem: unclear whether this cost-based merging will break anything or
    # was just cosmetic to avoid too many forwards and backwards.
    # related issue: https://github.com/CPJKU/partitura/issues/390

    # get the voice for which merging notes and other has lowest cost
    # merge_voice = sorted(cost.items(), key=itemgetter(1))[0][0]
    result = []
    pos = measure_start
    for i, voice in enumerate(sorted(notes.keys())):
        if i == 0:  # voice == merge_voice:
            elements = merged[voice]

        else:
            elements = notes[voice]

        # backup/forward when switching voices if necessary
        if elements:
            gap = elements[0][0] - pos

            if gap < 0:
                e = etree.Element("backup")
                ee = etree.SubElement(e, "duration")
                ee.text = "{:d}".format(-int(gap))
                result.append(e)

            elif gap > 0:
                e = etree.Element("forward")
                ee = etree.SubElement(e, "duration")
                ee.text = "{:d}".format(int(gap))
                result.append(e)

        result.extend([e for _, _, e in elements])

        # update current position
        if elements:
            pos = elements[-1][0] + (elements[-1][1] or 0)

    return result


def do_directions(part, start, end, counter):
    result = []

    # ending directions
    directions = part.iter_all(
        score.DynamicDirection,
        start.next,
        end.next,
        include_subclasses=True,
        mode="ending",
    )

    for direction in directions:
        text = direction.raw_text or direction.text
        e0 = etree.Element("direction")
        e1 = etree.SubElement(e0, "direction-type")

        if getattr(direction, "wedge", False):
            number = range_number_from_counter(direction, "wedge", counter)
            e2 = etree.SubElement(e1, "wedge", number="{}".format(number), type="stop")

        else:
            number = range_number_from_counter(direction, "wedge", counter)
            etree.SubElement(e1, "dashes", number="{}".format(number), type="stop")

        elem = (direction.end.t, None, e0)
        result.append(elem)

    tempos = part.iter_all(score.Tempo, start, end)
    directions = part.iter_all(score.Direction, start, end, include_subclasses=True)

    for tempo in tempos:
        # e0 = etree.Element('direction')
        # e1 = etree.SubElement(e0, 'direction-type')
        # e2 = etree.SubElement(e1, 'words')
        unit = "q" if tempo.unit is None else tempo.unit
        # e2.text = '{}={}'.format(unit, tempo.bpm)
        # result.append((tempo.start.t, None, e0))
        e3 = etree.Element(
            "sound", tempo="{}".format(int(to_quarter_tempo(unit, tempo.bpm)))
        )
        result.append((tempo.start.t, None, e3))

    for direction in directions:
        text = direction.raw_text or direction.text

        if text in PEDAL_DIRECTIONS:
            # Pedal directions create an element for start
            # and an element for ending

            # Use end of the segment as ending of the pedal sign
            ped_end = end if direction.end is None else direction.end

            # Create a pedal start element
            if direction.start.t >= start.t:
                e0s = etree.Element("direction", placement="below")
                e1s = etree.SubElement(e0s, "direction-type")
                # For sustain pedals
                if isinstance(direction, score.SustainPedalDirection):
                    pedal_kwargs = {}
                    if direction.line:
                        pedal_kwargs["line"] = "yes"
                    # For Flake8 (ignore unused variable), since
                    # etree.SubElement adds e2s to e1s
                    e2s = etree.SubElement(  # noqa: F841
                        e1s, "pedal", type="start", **pedal_kwargs
                    )
                if direction.staff is not None and direction.staff != 1:
                    e3s = etree.SubElement(e0s, "staff")
                    e3s.text = str(direction.staff)
                elem = (direction.start.t, None, e0s)
                result.append(elem)
            if ped_end.t <= end.t:
                e0e = etree.Element("direction", placement="below")
                e1e = etree.SubElement(e0e, "direction-type")
                if isinstance(direction, score.SustainPedalDirection):
                    pedal_kwargs = {}
                    if direction.line:
                        pedal_kwargs["line"] = "yes"
                    else:
                        pedal_kwargs["sign"] = "yes"
                    # For Flake8 (ignore unused variable), since
                    # etree.SubElement adds e2e to e1e
                    e2e = etree.SubElement(  # noqa: F841
                        e1e, "pedal", type="end", **pedal_kwargs
                    )
                if direction.staff is not None and direction.staff != 1:
                    e3e = etree.SubElement(e0e, "staff")
                    e3e.text = str(direction.staff)
                elem = (ped_end.t, None, e0e)
                result.append(elem)
        else:
            e0 = etree.Element("direction")
            e1 = etree.SubElement(e0, "direction-type")

            if text in DYN_DIRECTIONS:
                e2 = etree.SubElement(e1, "dynamics")
                etree.SubElement(e2, text)

            elif getattr(direction, "wedge", False):
                if isinstance(direction, score.IncreasingLoudnessDirection):
                    wtype = "crescendo"
                else:
                    wtype = "diminuendo"

                number = range_number_from_counter(direction, "wedge", counter)
                e2 = etree.SubElement(
                    e1, "wedge", number="{}".format(number), type=wtype
                )

            else:
                e2 = etree.SubElement(e1, "words")
                e2.text = filter_string(text)

                if (
                    isinstance(direction, score.DynamicDirection)
                    and direction.end is not None
                ):
                    e3 = etree.SubElement(e0, "direction-type")
                    number = range_number_from_counter(direction, "dashes", counter)
                    etree.SubElement(
                        e3, "dashes", number="{}".format(number), type="start"
                    )

            if direction.staff is not None and direction.staff != 1:
                e5 = etree.SubElement(e0, "staff")
                e5.text = str(direction.staff)

            elem = (direction.start.t, None, e0)
            result.append(elem)

    return result


def do_harmony(part, start, end):
    """
    Produce xml objects for harmony (Roman Numeral Text)
    """
    harmony = part.iter_all(score.RomanNumeral, start, end)
    result = []
    for h in harmony:
        harmony_e = etree.Element("harmony", print_frame="no")
        function = etree.SubElement(harmony_e, "function")
        function.text = h.text
        kind_e = etree.SubElement(harmony_e, "kind", text="")
        kind_e.text = "none"
        result.append((h.start.t, None, harmony_e))
    harmony = part.iter_all(score.ChordSymbol, start, end)
    for h in harmony:
        harmony_e = etree.Element("harmony", print_frame="no")
        kind_e = (
            etree.SubElement(harmony_e, "kind", text=h.kind)
            if h.kind is not None
            else etree.SubElement(harmony_e, "kind", text="")
        )
        kind_e.text = "none"
        root_e = etree.SubElement(harmony_e, "root")
        root_step_e = etree.SubElement(root_e, "root-step")
        root_step_e.text = h.root
        if h.bass is not None:
            bass_e = etree.SubElement(harmony_e, "bass")
            bass_step_e = etree.SubElement(bass_e, "bass-step")
            bass_step_e.text = h.bass
        result.append((h.start.t, None, harmony_e))

    # Does harmony annotation for cadences
    # TODO: Merge with existing Roman Numeral and ChordSymbol annotations if they exist.
    harmony = part.iter_all(score.Cadence, start, end)
    for h in harmony:
        harmony_e = etree.Element("harmony", print_frame="no")
        function = etree.SubElement(harmony_e, "function")
        function.text = "|" + h.text
        kind_e = etree.SubElement(harmony_e, "kind", text="")
        kind_e.text = "none"
        result.append((h.start.t, None, harmony_e))
    return result


def do_attributes(part, start, end):
    """
    Produce xml objects for non-note measure content

    Parameters
    ----------
    others: type
        Description of `others`

    Returns
    -------
    type
        Description of return value
    """

    by_start = defaultdict(list)

    # for o in part.iter_all(score.Divisions, start, end):
    #     by_start[o.start.t].append(o)
    for t, quarter in part.quarter_durations(start.t, end.t):
        by_start[t].append(int(quarter))
    for o in part.iter_all(score.KeySignature, start, end):
        by_start[o.start.t].append(o)
    for o in part.iter_all(score.TimeSignature, start, end):
        by_start[o.start.t].append(o)
    for o in part.iter_all(score.Staff, start, end):
        by_start[o.start.t].append(o)

    # sort clefs by number before adding them to by_start
    clefs_by_start = defaultdict(list)

    for o in part.iter_all(score.Clef, start, end):
        clefs_by_start[o.start.t].append(o)

    for t, clefs in clefs_by_start.items():
        clefs.sort(key=lambda clef: getattr(clef, "number", 0))
        by_start[t].extend(clefs)

    result = []

    # hacky: flag to include staves element before the first clef
    staves_included = False

    for t in sorted(by_start.keys()):
        attr_e = etree.Element("attributes")

        for o in by_start[t]:
            if isinstance(o, int):
                etree.SubElement(attr_e, "divisions").text = "{}".format(o)

            elif isinstance(o, score.KeySignature):
                ks_e = etree.SubElement(attr_e, "key")
                etree.SubElement(ks_e, "fifths").text = "{}".format(o.fifths)

                if o.mode:
                    etree.SubElement(ks_e, "mode").text = "{}".format(o.mode)

            elif isinstance(o, score.TimeSignature):
                ts_e = etree.SubElement(attr_e, "time")
                etree.SubElement(ts_e, "beats").text = "{}".format(o.beats)
                etree.SubElement(ts_e, "beat-type").text = "{}".format(o.beat_type)

            elif isinstance(o, score.Clef):
                if not staves_included:
                    staves_e = etree.SubElement(attr_e, "staves")
                    staves_e.text = "{}".format(len(clefs))
                    staves_included = True

                clef_e = etree.SubElement(attr_e, "clef")

                if o.staff and o.staff != 1:
                    clef_e.set("number", "{}".format(o.staff))

                etree.SubElement(clef_e, "sign").text = "{}".format(o.sign)
                etree.SubElement(clef_e, "line").text = "{}".format(o.line)

                if o.octave_change:
                    etree.SubElement(clef_e, "clef-octave-change").text = "{}".format(
                        o.octave_change
                    )
            elif isinstance(o, score.Staff):
                staff_e = etree.SubElement(attr_e, "staff-details")
                if o.lines:
                    etree.SubElement(staff_e, "staff-lines").text = "{}".format(o.lines)

        result.append((t, None, attr_e))

    return result


@deprecated_alias(parts="score_data")
def save_musicxml(
    score_data: score.ScoreLike,
    out: Optional[PathLike] = None,
) -> Optional[str]:
    """
    Save a one or more Part or PartGroup instances in MusicXML format.

    Parameters
    ----------
    score_data : Score, list, Part, or PartGroup
        The musical score to be saved. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    out: str, file-like object, or None, optional
        Output file

    Returns
    -------
    None or str
        If no output file is specified using `out` the function returns the
        MusicXML data as a string. Otherwise the function returns None.
    """

    if not isinstance(score_data, score.Score):
        score_data = score.Score(
            id=None,
            partlist=score_data,
        )

    root = etree.Element("score-partwise")

    partlist_e = etree.SubElement(root, "part-list")
    state = {
        "note_id_counter": {},
        "range_counter": {},
    }

    group_stack = []

    def close_group_stack():
        while group_stack:
            # close group
            etree.SubElement(
                partlist_e,
                "part-group",
                number="{}".format(group_stack[-1].number),
                type="stop",
            )
            # remove from stack
            group_stack.pop()

    def handle_parents(part):
        # 1. get deepest parent that is in group_stack (keep track of parents to
        # add)
        pg = part.parent
        to_add = []
        while pg:
            if pg in group_stack:
                break
            to_add.append(pg)
            pg = pg.parent

        # close groups while not equal to pg
        while group_stack:
            if pg == group_stack[-1]:
                break
            else:
                # close group
                etree.SubElement(
                    partlist_e,
                    "part-group",
                    number="{}".format(group_stack[-1].number),
                    type="stop",
                )
                # remove from stack
                group_stack.pop()

        # start all parents in to_add
        for pg in reversed(to_add):
            # start group
            pg_e = etree.SubElement(
                partlist_e, "part-group", number="{}".format(pg.number), type="start"
            )
            if pg.group_symbol is not None:
                symb_e = etree.SubElement(pg_e, "group-symbol")
                symb_e.text = pg.group_symbol
            if pg.group_name is not None:
                name_e = etree.SubElement(pg_e, "group-name")
                name_e.text = pg.group_name

            group_stack.append(pg)

    for part in score_data:
        handle_parents(part)

        # handle part list entry
        scorepart_e = etree.SubElement(partlist_e, "score-part", id=part.id)

        partname_e = etree.SubElement(scorepart_e, "part-name")
        if part.part_name:
            partname_e.text = filter_string(part.part_name)

        if part.part_abbreviation:
            partabbrev_e = etree.SubElement(scorepart_e, "part-abbreviation")
            partabbrev_e.text = filter_string(part.part_abbreviation)

        # write the part itself

        part_e = etree.SubElement(root, "part", id=part.id)

        for measure in part.iter_all(score.Measure):
            part_e.append(etree.Comment(MEASURE_SEP_COMMENT))
            attrib = {}

            if measure.name is not None:
                attrib["number"] = str(measure.name)

            measure_e = etree.SubElement(part_e, "measure", **attrib)
            contents = linearize_measure_contents(
                part, measure.start, measure.end, state
            )
            measure_e.extend(contents)

    close_group_stack()

    if out:
        if hasattr(out, "write"):
            out.write(
                etree.tostring(
                    root.getroottree(),
                    encoding="UTF-8",
                    xml_declaration=True,
                    pretty_print=True,
                    doctype=DOCTYPE,
                )
            )

        else:
            with open(out, "wb") as f:
                f.write(
                    etree.tostring(
                        root.getroottree(),
                        encoding="UTF-8",
                        xml_declaration=True,
                        pretty_print=True,
                        doctype=DOCTYPE,
                    )
                )

    else:
        return etree.tostring(
            root.getroottree(),
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=DOCTYPE,
        )
