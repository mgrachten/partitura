#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing MIDI files.
"""
import warnings

from collections import defaultdict
from typing import Union, Optional, List, Tuple, Dict
import numpy as np


import mido

import partitura.score as score
import partitura.performance as performance
from partitura.utils import (
    estimate_symbolic_duration,
    key_name_to_fifths_mode,
    fifths_mode_to_key_name,
    estimate_clef_properties,
    deprecated_alias,
    deprecated_parameter,
    PathLike,
    get_document_name,
    ensure_notearray,
)
import partitura.musicanalysis as analysis

__all__ = ["load_score_midi", "load_performance_midi", "midi_to_notearray"]


# as key for the dict use channel * 128 (max number of pitches) + pitch
def note_hash(channel: int, pitch: int) -> int:
    """Generate a note hash."""
    return channel * 128 + pitch


@deprecated_alias(fn="filename")
def midi_to_notearray(filename: PathLike) -> np.ndarray:
    """Load a MIDI file in a note_array.

    This function should be used to load MIDI files into an
    array of MIDI notes given by onset and duration (in seconds),
    pitch, velocity, and ID.

    Sustain pedal, program changes, control changes, track and
    channel information as well as mpq and ppq are discarded.

    Parameters
    ----------
    filename : str
        Path to MIDI file
    Returns
    -------
    np.ndarray :
        Structured array with onset, duration, pitch, velocity, and
        ID fields.
    """
    perf = load_performance_midi(filename, merge_tracks=True)
    # set sustain pedal threshold to 128 to disable sustain adjusted offsets

    for ppart in perf:
        ppart.sustain_pedal_threshold = 128

    note_array = ensure_notearray(perf)
    return note_array


@deprecated_alias(fn="filename")
def load_performance_midi(
    filename: Union[PathLike, mido.MidiFile],
    default_bpm: Union[int, float] = 120,
    merge_tracks: bool = False,
) -> performance.Performance:
    """Load a musical performance from a MIDI file.

    This function should be used for MIDI files that encode
    performances, such as those obtained from a capture of a MIDI
    instrument. This function loads note on/off events as well as
    control events, but ignores other data such as time and key
    signatures. Furthermore, the PerformedPart instance that the
    function returns does not retain the ticks_per_beat or tempo
    events. The timing of all events is represented in seconds. If you
    wish to retain this information consider using the
    `load_score_midi` function.

    Parameters
    ----------
    filename : str
        Path to MIDI file
    default_bpm : number, optional
        Tempo to use wherever the MIDI does not specify a tempo.
        Defaults to 120.
    merge_tracks: bool, optional
        For MIDI files, merges all tracks into a single track.

    Returns
    -------
    :class:`partitura.performance.Performance`
        A Performance instance.
    """

    if isinstance(filename, mido.MidiFile):
        mid = filename
        doc_name = filename.filename
    else:
        mid = mido.MidiFile(filename)
        doc_name = get_document_name(filename)

    # parts per quarter
    ppq = mid.ticks_per_beat
    # microseconds per quarter
    mpq = 60 * (10**6 / default_bpm)

    # convert MIDI ticks in seconds
    time_conversion_factor = mpq / (ppq * 10**6)

    pps = list()

    if merge_tracks:
        mid_merge = mido.merge_tracks(mid.tracks)
        tracks = [(0, mid_merge)]
    else:
        tracks = [(i, u) for i, u in enumerate(mid.tracks)]
    for i, track in tracks:
        notes = []
        controls = []
        programs = []
        # This information is just for completeness,
        # but loading a MIDI file as a performance
        # assumes that key and time signature information
        # is not reliable (e.g., a performance recorded with
        # a MIDI keyboard, without metronome)
        key_signatures = []
        time_signatures = []
        # other MetaMessages (not including key and time_signature)
        meta_other = []

        t = 0
        ttick = 0

        sounding_notes = {}

        for msg in track:
            # update time deltas when they arrive
            t = t + msg.time * time_conversion_factor
            ttick = ttick + msg.time

            if isinstance(msg, mido.MetaMessage):
                # Meta Messages apply to all channels in the track

                # The tempo is set globally in PerformedParts,
                # i.e., the tempo_conversion_factor is adjusted
                # with every tempo change, rather than creating new
                # tempo events.
                if msg.type == "set_tempo":
                    mpq = msg.tempo
                    time_conversion_factor = mpq / (ppq * 10**6)

                elif msg.type == "time_signature":
                    time_signatures.append(
                        dict(
                            time=t,
                            time_tick=ttick,
                            beats=int(msg.numerator),
                            beat_type=int(msg.denominator),
                            track=i,
                        )
                    )
                elif msg.type == "key_signature":
                    key_name = str(msg.key)
                    fifths, mode = key_name_to_fifths_mode(key_name)
                    key_signatures.append(
                        dict(
                            time=t,
                            time_tick=ttick,
                            key_name=str(msg.key),
                            fifths=fifths,
                            mode=mode,
                            track=i,
                        )
                    )

                else:
                    # Other MetaMessages
                    # For more info, see
                    # https://mido.readthedocs.io/en/latest/meta_message_types.html
                    msg_dict = dict(
                        [
                            ("time", t),
                            ("time_tick", ttick),
                            ("track", i),
                        ]
                        + [
                            (key, val)
                            for key, val in msg.__dict__.items()
                            if key not in ("time", "track", "time_tick")
                        ]
                    )

                    meta_other.append(msg_dict)

            elif msg.type == "control_change":
                controls.append(
                    dict(
                        time=t,
                        time_tick=ttick,
                        number=msg.control,
                        value=msg.value,
                        track=i,
                        channel=msg.channel,
                    )
                )

            elif msg.type == "program_change":
                programs.append(
                    dict(
                        time=t,
                        time_tick=ttick,
                        program=msg.program,
                        track=i,
                        channel=msg.channel,
                    )
                )

            else:
                note_on = msg.type == "note_on"
                note_off = msg.type == "note_off"

                if not (note_on or note_off):
                    continue

                # hash sounding note
                note = note_hash(msg.channel, msg.note)

                # start note if it's a 'note on' event with velocity > 0
                if note_on and msg.velocity > 0:
                    # save the onset time and velocity
                    sounding_notes[note] = (t, ttick, msg.velocity)

                # end note if it's a 'note off' event or 'note on' with velocity 0
                elif note_off or (note_on and msg.velocity == 0):
                    if note not in sounding_notes:
                        warnings.warn(f"ignoring MIDI message {msg}")
                        continue

                    # append the note to the list associated with the channel
                    notes.append(
                        dict(
                            # id=f"n{len(notes)}",
                            midi_pitch=msg.note,
                            note_on=(sounding_notes[note][0]),
                            note_on_tick=(sounding_notes[note][1]),
                            note_off=(t),
                            note_off_tick=(ttick),
                            track=i,
                            channel=msg.channel,
                            velocity=sounding_notes[note][2],
                        )
                    )
                    # remove hash from dict
                    del sounding_notes[note]

        # fix note ids so that it is sorted lexicographically
        # by onset, pitch, offset, channel and track
        notes.sort(
            key=lambda x: (
                x["note_on"],
                x["midi_pitch"],
                x["note_off"],
                x["channel"],
                x["track"],
            )
        )

        # add note id to every note
        for k, note in enumerate(notes):
            note["id"] = f"n{k}"

        if len(notes) > 0 or len(controls) > 0 or len(programs) > 0:
            pp = performance.PerformedPart(
                notes,
                controls=controls,
                programs=programs,
                key_signatures=key_signatures,
                time_signatures=time_signatures,
                meta_other=meta_other,
                ppq=ppq,
                mpq=mpq,
                track=i,
            )

            pps.append(pp)

    perf = performance.Performance(
        id=doc_name,
        performedparts=pps,
    )
    return perf


@deprecated_parameter("ensure_list")
@deprecated_alias(fn="filename")
def load_score_midi(
    filename: Union[PathLike, mido.MidiFile],
    part_voice_assign_mode: Optional[int] = 0,
    quantization_unit: Optional[int] = None,
    estimate_voice_info: bool = False,
    estimate_key: bool = False,
    assign_note_ids: bool = True,
) -> score.Score:
    """Load a musical score from a MIDI file and return it as a Part
    instance.

    This function interprets MIDI information as describing a score.
    Pitch names are estimated using Meredith's PS13 algorithm [1]_.
    Assignment of notes to voices can either be done using Chew and
    Wu's voice separation algorithm [2]_, or by choosing one of the
    part/voice assignment modes that assign voices based on
    track/channel information. Furthermore, the key signature can be
    estimated based on Krumhansl's 1990 key profiles [3]_.

    This function expects times to be metrical/quantized. Optionally a
    quantization unit may be specified. If you wish to access the non-
    quantized time of MIDI events you may wish to used the
    `load_performance_midi` function instead.

    Parameters
    ----------
    filename : PathLike or mido.MidiFile
        Path to MIDI file or mido.MidiFile object.
    part_voice_assign_mode : {0, 1, 2, 3, 4, 5}, optional
        This keyword controls how part and voice information is
        associated to track and channel information in the MIDI file.
        The semantics of the modes is as follows:

        0
            Return one Part per track, with voices assigned by channel
        1
            Return one PartGroup per track, with Parts assigned by channel
            (no voices)
        2
            Return single Part with voices assigned by track (tracks are
            combined, channel info is ignored)
        3
            Return one Part per track, without voices (channel info is
            ignored)
        4
            Return single Part without voices (channel and track info is
            ignored)
        5
            Return one Part per <track, channel> combination, without
            voices  Defaults to 0.
    quantization_unit : integer or None, optional
        Quantize MIDI times to multiples of this unit. If None, the
        quantization unit is chosen automatically as the smallest
        division of the parts per quarter (MIDI "ticks") that can be
        represented as a symbolic duration. Defaults to None.
    estimate_key : bool, optional
        When True use Krumhansl's 1990 key profiles [3]_ to determine
        the most likely global key, discarding any key information in
        the MIDI file.
    estimate_voice_info : bool, optional
        When True use Chew and Wu's voice separation algorithm [2]_ to
        estimate voice information. If the voice information was imported
        from the file, it will be overridden. Defaults to False.

    Returns
    -------
    :class:`partitura.score.Part`, :class:`partitura.score.PartGroup`, \
or a list of these
        One or more part or partgroup objects

    References
    ----------
    .. [1] Meredith, D. (2006). "The ps13 Pitch Spelling Algorithm". Journal
           of New Music Research, 35(2):121.
    .. [2] Chew, E. and Wu, Xiaodan (2004) "Separating Voices in
           Polyphonic Music: A Contig Mapping Approach". In Uffe Kock,
           editor, Computer Music Modeling and Retrieval (CMMR), pp. 1–20,
           Springer Berlin Heidelberg.
    .. [3] Krumhansl, Carol L. (1990) "Cognitive foundations of musical pitch",
           Oxford University Press, New York.

    """

    if isinstance(filename, mido.MidiFile):
        mid = filename
        doc_name = filename.filename
    else:
        mid = mido.MidiFile(filename)
        doc_name = get_document_name(filename)

    divs = mid.ticks_per_beat

    # these lists will contain information from dedicated tracks for meta
    # information (i.e. without notes)
    global_time_sigs = []
    global_key_sigs = []
    global_tempos = []

    # these dictionaries will contain meta information indexed by track (only
    # for tracks that contain notes)
    time_sigs_by_track = {}
    key_sigs_by_track = {}
    track_names_by_track = {}
    # notes are indexed by (track, channel) tuples
    notes_by_track_ch = {}
    relevant = {
        "time_signature",
        "key_signature",
        "set_tempo",
        "note_on",
        "note_off",
    }
    for track_nr, track in enumerate(mid.tracks):
        time_sigs = []
        key_sigs = []
        # tempos = []
        notes = defaultdict(list)
        # dictionary for storing the last onset time and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}
        # current time (will be updated by delta times in messages)
        t_raw = 0

        for msg in track:
            t_raw = t_raw + msg.time

            if msg.type not in relevant:
                continue

            if quantization_unit:
                t = quantize(t_raw, quantization_unit)
            else:
                t = t_raw

            if msg.type == "time_signature":
                time_sigs.append((t, msg.numerator, msg.denominator))
            if msg.type == "key_signature":
                key_sigs.append((t, msg.key))
            if msg.type == "set_tempo":
                global_tempos.append((t, 60 * 10**6 / msg.tempo))
            else:
                note_on = msg.type == "note_on"
                note_off = msg.type == "note_off"

                if not (note_on or note_off):
                    continue

                # hash sounding note
                note = note_hash(msg.channel, msg.note)

                # start note if it's a 'note on' event with velocity > 0
                if note_on and msg.velocity > 0:
                    # save the onset time and velocity
                    sounding_notes[note] = (t, msg.velocity)

                # end note if it's a 'note off' event or 'note on' with velocity 0
                elif note_off or (note_on and msg.velocity == 0):
                    if note not in sounding_notes:
                        warnings.warn("ignoring MIDI message %s" % msg)
                        continue

                    # append the note to the list associated with the channel
                    notes[msg.channel].append(
                        (sounding_notes[note][0], msg.note, t - sounding_notes[note][0])
                    )
                    # sounding_notes[note][1]])
                    # remove hash from dict
                    del sounding_notes[note]

        # if a track has no notes, we assume it may contain global time/key sigs
        if not notes:
            global_time_sigs.extend(time_sigs)
            global_key_sigs.extend(key_sigs)
        else:
            # if there are note, we store the info under the track number
            time_sigs_by_track[track_nr] = time_sigs
            key_sigs_by_track[track_nr] = key_sigs
            track_names_by_track[track_nr] = track.name

        for ch, ch_notes in notes.items():
            # if there are any notes, store the notes along with key sig / time
            # sig / tempo information under the key (track_nr, ch_nr)
            if len(ch_notes) > 0:
                notes_by_track_ch[(track_nr, ch)] = ch_notes

    tr_ch_keys = sorted(notes_by_track_ch.keys())
    group_part_voice_keys, part_names, group_names = assign_group_part_voice(
        part_voice_assign_mode, tr_ch_keys, track_names_by_track
    )

    # for key and time sigs:
    track_to_part_mapping = make_track_to_part_mapping(
        tr_ch_keys, group_part_voice_keys
    )

    # pairs of (part, voice) for each note
    part_voice_list = [
        [part, voice]
        for tr_ch, (_, part, voice) in zip(tr_ch_keys, group_part_voice_keys)
        for i in range(len(notes_by_track_ch[tr_ch]))
    ]

    # pitch spelling, voice estimation and key estimation are done on a
    # structured array (onset, pitch, duration) of all notes in the piece
    # jointly, so we concatenate all notes
    # note_list = sorted(note for notes in
    # (notes_by_track_ch[key] for key in tr_ch_keys) for note in notes)
    note_list = [
        note
        for notes in (notes_by_track_ch[key] for key in tr_ch_keys)
        for note in notes
    ]
    note_array = np.array(
        note_list,
        dtype=[("onset_div", int), ("pitch", int), ("duration_div", int)],
    )

    warnings.warn("pitch spelling")
    spelling_global = analysis.estimate_spelling(note_array)

    if estimate_voice_info:
        warnings.warn("voice estimation", stacklevel=2)
        # TODO: deal with zero duration notes in note_array.
        # Zero duration notes are currently deleted
        estimated_voices = analysis.estimate_voices(note_array)
        assert len(part_voice_list) == len(estimated_voices)
        for part_voice, voice_est in zip(part_voice_list, estimated_voices):
            if part_voice[1] is None:
                part_voice[1] = voice_est

    if estimate_key:
        warnings.warn("key estimation", stacklevel=2)
        _, mode, fifths = analysis.estimate_key(note_array)
        key_sigs_by_track = {}
        global_key_sigs = [(0, fifths_mode_to_key_name(fifths, mode))]

    if assign_note_ids:
        note_ids = ["n{}".format(i) for i in range(len(note_array))]
    else:
        note_ids = [None for i in range(len(note_array))]

    ## sanitize time signature, when they are only present in one track, and no global is set
    # find the number of ts per each track
    number_of_time_sig_per_track = [
        len(time_sigs_by_track[t]) for t in key_sigs_by_track.keys()
    ]
    # if one track has 0 ts, and another has !=0 ts, and no global_time_sigs is present, sanitize
    # all key signatures are copied to global, and the track ts are removed
    if (
        len(global_time_sigs) == 0
        and min(number_of_time_sig_per_track) == 0
        and max(number_of_time_sig_per_track) != 0
    ):
        warnings.warn(
            "Sanitizing time signatures. They will be shared across all tracks."
        )
        for ts in [
            ts for ts_track in time_sigs_by_track.values() for ts in ts_track
        ]:  # flattening all track time signatures to a list of ts
            global_time_sigs.append(ts)
        # now clear all track_ts
        time_sigs_by_track.clear()

    time_sigs_by_part = defaultdict(set)
    for tr, ts_list in time_sigs_by_track.items():
        for ts in ts_list:
            for part in track_to_part_mapping[tr]:
                time_sigs_by_part[part].add(ts)
    for ts in global_time_sigs:
        for part in set(part for _, part, _ in group_part_voice_keys):
            time_sigs_by_part[part].add(ts)

    key_sigs_by_part = defaultdict(set)
    for tr, ks_list in key_sigs_by_track.items():
        for ks in ks_list:
            for part in track_to_part_mapping[tr]:
                key_sigs_by_part[part].add(ks)
    for ks in global_key_sigs:
        for part in set(part for _, part, _ in group_part_voice_keys):
            key_sigs_by_part[part].add(ks)

    # names_by_part = defaultdict(set)
    # for tr_ch, pg_p_v in zip(tr_ch_keys, group_part_voice_keys):
    #     print(tr_ch, pg_p_v)
    # for tr, name in track_names_by_track.items():
    #     print(tr, track_to_part_mapping, name)
    #     for part in track_to_part_mapping[tr]:
    #         names_by_part[part] = name

    notes_by_part = defaultdict(list)
    for (part, voice), note, spelling, note_id in zip(
        part_voice_list, note_list, spelling_global, note_ids
    ):
        notes_by_part[part].append((note, voice, spelling, note_id))

    partlist = []
    part_to_part_group = dict((p, pg) for pg, p, _ in group_part_voice_keys)
    part_groups = {}
    for part_nr, note_info in notes_by_part.items():
        notes, voices, spellings, note_ids = zip(*note_info)
        part = create_part(
            divs,
            notes,
            spellings,
            voices,
            note_ids,
            sorted(time_sigs_by_part[part_nr]),
            sorted(key_sigs_by_part[part_nr]),
            part_id="P{}".format(part_nr + 1),
            part_name=part_names.get(part_nr, None),
        )

        # print(part.pretty())
        # if this part has an associated part_group number we create a PartGroup
        # if necessary, and add the part to that. The newly created PartGroup is
        # then added to the partlist.
        pg_nr = part_to_part_group[part_nr]
        if pg_nr is None:
            partlist.append(part)
        else:
            if pg_nr not in part_groups:
                part_groups[pg_nr] = score.PartGroup(
                    group_name=group_names.get(pg_nr, None)
                )
                partlist.append(part_groups[pg_nr])
            part_groups[pg_nr].children.append(part)

    # add tempos to first part
    part = next(score.iter_parts(partlist))
    for t, qpm in global_tempos:
        part.add(score.Tempo(qpm, unit="q"), t)

    # TODO: Add info (composer, etc.)
    scr = score.Score(
        id=doc_name,
        partlist=partlist,
    )

    return scr


def make_track_to_part_mapping(tr_ch_keys, group_part_voice_keys):
    """Return a mapping from track numbers to one or more parts. This mapping tells
    us where to put meta event info like time and key sigs.
    """
    track_to_part_keys = defaultdict(set)
    for (tr, _), (_, part, _) in zip(tr_ch_keys, group_part_voice_keys):
        track_to_part_keys[tr].add(part)
    return track_to_part_keys


def assign_group_part_voice(
    mode: int,
    track_ch_combis: Dict[Tuple[int, int], List],
    track_names: Dict[int, str],
) -> Tuple[List[Tuple], Dict, Dict]:
    """
    0: return one Part per track, with voices assigned by channel
    1. return one PartGroup per track, with Parts assigned by channel (no voices)
    2. return single Part with voices assigned by track (tracks are combined,
       channel info is ignored)
    3. return one Part per track, without voices (channel info is ignored)
    4. return single Part without voices (channel and track info is ignored)
    5. return one Part per <track, channel> combination, without voices
    """
    part_group = {}
    part = {}
    voice = {}
    part_helper = {}
    voice_helper = {}
    part_group_helper = {}

    part_names = {}
    group_names = {}
    for tr, ch in track_ch_combis:
        if mode == 0:
            prt = part_helper.setdefault(tr, len(part_helper))
            vc1 = voice_helper.setdefault(tr, {})
            vc2 = vc1.setdefault(ch, len(vc1) + 1)
            part_names[prt] = "{}".format(
                track_names.get(tr, "Track {}".format(tr + 1))
            )
            part[(tr, ch)] = prt
            voice[(tr, ch)] = vc2
        elif mode == 1:
            pg = part_group_helper.setdefault(tr, len(part_group_helper))
            prt = part_helper.setdefault(ch, len(part_helper))
            part_group.setdefault((tr, ch), pg)
            group_names[pg] = track_names.get(tr, "Track {}".format(tr + 1))
            part_names[prt] = "ch={}".format(ch)
            part[(tr, ch)] = prt
        elif mode == 2:
            vc = voice_helper.setdefault(tr, len(voice_helper) + 1)
            part.setdefault((tr, ch), 0)
            voice[(tr, ch)] = vc
        elif mode == 3:
            prt = part_helper.setdefault(tr, len(part_helper))
            part_names[prt] = "{}".format(
                track_names.get(tr, "Track {}".format(tr + 1))
            )
            part[(tr, ch)] = prt
        elif mode == 4:
            part.setdefault((tr, ch), 0)
        elif mode == 5:
            part_names[(tr, ch)] = "{} ch={}".format(
                track_names.get(tr, "Track {}".format(tr + 1)), ch
            )
            part.setdefault((tr, ch), len(part))

    return (
        [
            (part_group.get(tr_ch), part.get(tr_ch), voice.get(tr_ch))
            for tr_ch in track_ch_combis
        ],
        part_names,
        group_names,
    )


def create_part(
    ticks: int,
    notes: List[Tuple[int, int, int]],
    spellings: List[Tuple[str, str, int]],
    voices: List[int],
    note_ids: List[str],
    time_sigs: List[Tuple[int, int, int]],
    key_sigs: List[Tuple[int, str]],
    part_id: Optional[str] = None,
    part_name: Optional[str] = None,
) -> score.Part:
    """
    Create score part object

    Parameters
    ----------
    ticks: int
        Integer unit to represent onset and duration information
        in the score in a lossless way.
    notes: List[Tuple[int, int, int]]
        Note information (onset, pitch, duration)
    spellings: List[Tuple[str, str, int]]
    voices: List[str]
    note_ids: List[str]
    time_sigs: List[Tuple[int, int, int]]
    key_sigs:
    part_id
    part_name

    Returns
    -------
    part: partitura.score.Part
        An object representing a Part in the score
    """
    warnings.warn("create_part", stacklevel=2)

    part = score.Part(part_id, part_name=part_name)
    part.set_quarter_duration(0, ticks)

    clef = score.Clef(
        staff=1, **estimate_clef_properties([pitch for _, pitch, _ in notes])
    )
    part.add(clef, 0)
    for t, name in key_sigs:
        fifths, mode = key_name_to_fifths_mode(name)
        part.add(score.KeySignature(fifths, mode), t)

    warnings.warn("add notes", stacklevel=2)

    for (onset, pitch, duration), (step, alter, octave), voice, note_id in zip(
        notes, spellings, voices, note_ids
    ):
        if duration > 0:
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                voice=int(voice or 0),
                id=note_id,
                symbolic_duration=estimate_symbolic_duration(duration, ticks),
            )
        else:
            note = score.GraceNote(
                grace_type="appoggiatura",
                step=step,
                octave=octave,
                alter=alter,
                voice=int(voice or 0),
                id=note_id,
                symbolic_duration=dict(type="quarter"),
            )

        part.add(note, onset, onset + duration)

    if not time_sigs:
        warnings.warn("No time signatures found, assuming 4/4")
        time_sigs = [(0, 4, 4)]

    time_sigs = np.array(time_sigs, dtype=int)

    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.iinfo(int).max]
    time_sigs = np.column_stack((time_sigs, ts_end_times))

    warnings.warn("add time sigs and measures", stacklevel=2)

    for ts_start, num, den, ts_end in time_sigs:
        time_sig = score.TimeSignature(num.item(), den.item())
        part.add(time_sig, ts_start.item())

    score.add_measures(part)

    # this is the old way to add measures. Since part comes from MIDI we
    # only have a single global divs value, which makes add it easier to compute
    # measure durations:

    # measure_counter = 1
    # # we call item() on numpy numbers to get the value in the equivalent python type
    # for ts_start, num, den, ts_end in time_sigs:
    #     time_sig = score.TimeSignature(num.item(), den.item())
    #     part.add(time_sig, ts_start.item())
    #     measure_duration = (num.item() * ticks * 4) // den.item()
    #     measure_start_limit = min(ts_end.item(), part.last_point.t)
    #     for m_start in range(ts_start, measure_start_limit, measure_duration):
    #         measure = score.Measure(number=measure_counter)
    #         m_end = min(m_start+measure_duration, ts_end)
    #         part.add(measure, m_start, m_end)
    #         measure_counter += 1
    #     if np.isinf(ts_end):
    #         ts_end = m_end

    warnings.warn("tie notes", stacklevel=2)
    # tie notes where necessary (across measure boundaries, and within measures
    # notes with compound duration)
    score.tie_notes(part)

    warnings.warn("find tuplets", stacklevel=2)
    # apply simplistic tuplet finding heuristic
    score.find_tuplets(part)

    warnings.warn("done create_part", stacklevel=2)
    return part


def quantize(
    v: Union[np.ndarray, float, int],
    unit: Union[float, int],
) -> Union[np.ndarray, float, int]:
    """Quantize value `v` to a multiple of `unit`. When `unit` is an integer,
    the return value will be integer as well, otherwise the function will
    return a float.

    Parameters
    ----------
    v : ndarray or number
        Number to be quantized
    unit : number
        The quantization unit

    Returns
    -------
    number
        The quantized number

    Examples
    --------
    >>> quantize(13.3, 4)
    12
    >>> quantize(3.3, .5)
    3.5

    """

    r = unit * np.round(v / unit)
    if isinstance(unit, int):
        return int(r)
    else:
        return r


if __name__ == "__main__":
    import doctest

    doctest.testmod()
