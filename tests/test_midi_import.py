#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for importing MIDI files.
"""
import logging
from collections import defaultdict, Counter
from operator import itemgetter
import unittest
from tempfile import NamedTemporaryFile
import mido

from partitura import load_score_midi
from partitura.utils import partition
import partitura.score as score
from partitura import load_performance_midi
from tests import MIDIINPORT_TESTFILES

LOGGER = logging.getLogger(__name__)


def fill_track(track, notes, divs, vel=64):
    # add on/off events for note specification (`notes`) to `track`
    onoffs = defaultdict(list)
    for on, off, pitch, ch in notes:
        on = int(divs * on)
        off = int(divs * off)
        onoffs[on].append(("note_on", pitch, ch))
        onoffs[off].append(("note_off", pitch, ch))

    times = sorted(onoffs.keys())
    prev = 0
    for t in times:
        dt = t - prev
        for msg, pitch, ch in onoffs[t]:
            track.append(
                mido.Message(msg, note=pitch, velocity=vel, channel=ch, time=dt)
            )
            dt = 0
        prev = t


def make_assignment_mode_example():
    # create a midi file on which to test the assignment modes in load_score_midi
    divs = 10
    mid = mido.MidiFile(ticks_per_beat=divs)
    track_1 = mido.MidiTrack()
    mid.tracks.append(track_1)
    tempo = int(0.5 * 10**6)
    track_1.append(mido.MetaMessage("track_name", name="Track 1"))
    # track_1.append(mido.MetaMessage('instrument_name', name='Instrument 1'))
    # track_1.append(mido.Message('program_change', channel=0, program=1))
    # track_1.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track_1.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
    )

    track_2 = mido.MidiTrack()
    mid.tracks.append(track_2)
    track_2.append(mido.MetaMessage("track_name", name="Track 2"))
    # track_2.append(mido.MetaMessage('instrument_name', name='Instrument 2'))
    # track_2.append(mido.Message('program_change', channel=1, program=57))
    # track_2.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track_2.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
    )

    # on off pitch ch, (times in quarter)
    notes_1 = [
        (0, 1, 76, 0),
        (1, 2, 76, 0),
        (2, 3, 76, 0),
        (0, 4, 69, 2),
    ]
    notes_2 = [
        (0, 2, 80, 1),
        (2, 4, 69, 1),
    ]

    fill_track(track_1, notes_1, divs)
    fill_track(track_2, notes_2, divs)
    return mid


def get_track_voice_numbers(mid):
    tc_counter = Counter()
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "note_on":
                tc_counter.update(((i, msg.channel),))
    return tc_counter


def make_triplets_example_1():
    divs = 120
    mid = mido.MidiFile(ticks_per_beat=divs)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # define two consecutive quarter triplets
    # on off pitch ch, (times in quarter)
    notes = [
        (0, 2 / 3, 76, 0),
        (2 / 3, 4 / 3, 76, 0),
        (4 / 3, 2, 76, 0),
        (2, 8 / 3, 76, 0),
        (8 / 3, 10 / 3, 76, 0),
        (10 / 3, 4, 76, 0),
    ]
    fill_track(track, notes, divs)
    # target:
    actual_notes = [3] * 6
    normal_notes = [2] * 6
    return mid, actual_notes, normal_notes


def make_triplets_example_2():
    divs = 120
    mid = mido.MidiFile(ticks_per_beat=divs)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # define two consecutive quarter triplets
    # on off pitch ch, (times in quarter)
    notes = [
        (0, 2 / 5, 76, 0),
        (2 / 5, 4 / 5, 76, 0),
        (4 / 5, 6 / 5, 76, 0),
        (6 / 5, 8 / 5, 76, 0),
        (8 / 5, 2, 76, 0),
        (2, 8 / 3, 76, 0),
        (8 / 3, 10 / 3, 76, 0),
        (10 / 3, 4, 76, 0),
    ]
    fill_track(track, notes, divs)
    # target:
    actual_notes = [5] * 5 + [3] * 3
    normal_notes = [4] * 5 + [2] * 3
    return mid, actual_notes, normal_notes


class TestMIDITuplets(unittest.TestCase):

    example_gen_funcs = (make_triplets_example_1, make_triplets_example_2)

    def test_tuplets(self):
        for example_gen_func in self.example_gen_funcs:
            mid, actual, normal = example_gen_func()
            with NamedTemporaryFile(suffix=".mid", delete=False) as fh:
                mid.save(fh.name)
                scr = load_score_midi(fh.name, part_voice_assign_mode=0)
                part = scr[0]
                notes = part.notes

                if len(actual) != len(normal):
                    LOGGER.warning("Error in example case, skipping test")
                    return
                msg = "Example Part has an unexpected number of notes (expected {}, got {})".format(
                    len(actual), len(notes)
                )
                self.assertEqual(len(notes), len(actual), msg)
                sym_durs = [n.symbolic_duration for n in notes]
                msg = "Incorrectly detected tuplets"
                self.assertEqual(
                    actual, [sd.get("actual_notes") for sd in sym_durs], msg
                )
                self.assertEqual(
                    normal, [sd.get("normal_notes") for sd in sym_durs], msg
                )


class TestMIDIImportModes(unittest.TestCase):
    def setUp(self):
        self.mid = make_assignment_mode_example()
        self.notes_per_tr_ch = get_track_voice_numbers(self.mid)
        self.tmpfile = NamedTemporaryFile(suffix=".mid", delete=False)
        self.mid.save(self.tmpfile.name)

    def test_midi_import_mode_0(self):
        parts = load_score_midi(self.tmpfile.name, part_voice_assign_mode=0)
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())

        msg = (
            "Number of parts {} does not equal number of tracks {} while "
            "testing part_voice_assign_mode=0 in load_score_midi"
        ).format(len(parts), len(by_track))
        self.assertEqual(len(parts), len(by_track), msg)

        for part, tr in zip(parts, by_track):

            msg = "{} should be a Part instance but it is not".format(part)
            self.assertTrue(isinstance(part, score.Part), msg)

            n_track_notes = sum(self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr])
            part_notes = part.notes
            n_part_notes = len(part_notes)
            msg = "Part should have {} notes but it has".format(
                n_track_notes, n_part_notes
            )
            self.assertEqual(n_track_notes, n_part_notes, msg)

            n_ch_notes = [self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr]]
            n_voice_notes = [
                len(vn)
                for v, vn in partition(
                    lambda x: x, [n.voice for n in part_notes]
                ).items()
            ]
            msg = "Part voices should have {} notes respectively, but they have {}".format(
                n_ch_notes, n_voice_notes
            )
            self.assertEqual(n_ch_notes, n_voice_notes, msg)

    def test_midi_import_mode_1(self):
        scr = load_score_midi(self.tmpfile.name, part_voice_assign_mode=1)

        parts = scr.part_structure
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())
        msg = (
            "Number of partgroups {} does not equal number of tracks {} while "
            "testing part_voice_assign_mode=0 in load_score_midi"
        ).format(len(parts), len(by_track))
        self.assertEqual(len(parts), len(by_track), msg)

        for part_group, tr in zip(parts, by_track):

            msg = "{} should be a PartGroup instance but it is not"
            self.assertTrue(isinstance(part_group, score.PartGroup), msg)
            n_parts = len(part_group.children)
            n_channels = len(by_track[tr])
            msg = (
                "PartGroup should have as many parts as there are "
                "channels in the corresponding track {}, but it has {}".format(
                    n_channels, n_parts
                )
            )
            self.assertEqual(n_parts, n_channels, msg)

            for part, tr_ch in zip(part_group.children, by_track[tr]):
                notes_in_track = self.notes_per_tr_ch[tr_ch]
                notes_in_part = len(part.notes)
                msg = "Part should have {} notes but it has {}".format(
                    notes_in_track, notes_in_part
                )
                self.assertEqual(notes_in_part, notes_in_track)

    def test_midi_import_mode_2(self):
        scr = load_score_midi(self.tmpfile.name, part_voice_assign_mode=2)
        self.assertTrue(len(scr) == 1)
        part = scr.part_structure[0]
        msg = "{} should be a Part instance but it is not".format(part)
        self.assertTrue(isinstance(part, score.Part), msg)
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())
        by_voice = partition(lambda x: x.voice, part.notes)
        n_track_notes = [
            sum(self.notes_per_tr_ch[tr_ch] for tr_ch in tr_chs)
            for tr_chs in by_track.values()
        ]
        n_voice_notes = [len(notes) for notes in by_voice.values()]
        msg = (
            "Number of notes per voice {} does not match number of "
            "notes per track {}".format(n_voice_notes, n_track_notes)
        )
        self.assertEqual(n_voice_notes, n_track_notes, msg)

    def test_midi_import_mode_3(self):
        parts = load_score_midi(self.tmpfile.name, part_voice_assign_mode=3)
        by_track = partition(itemgetter(0), self.notes_per_tr_ch.keys())

        msg = "Number of parts {} does not equal number of tracks {}".format(
            len(parts), len(by_track)
        )
        self.assertEqual(len(parts), len(by_track), msg)

        for part, tr in zip(parts, by_track):

            msg = "{} should be a Part instance but it is not".format(part)
            self.assertTrue(isinstance(part, score.Part), msg)

            n_track_notes = sum(self.notes_per_tr_ch[tr_ch] for tr_ch in by_track[tr])
            part_notes = part.notes
            n_part_notes = len(part_notes)
            msg = "Part should have {} notes but it has".format(
                n_track_notes, n_part_notes
            )
            self.assertEqual(n_track_notes, n_part_notes, msg)

    def test_midi_import_mode_4(self):
        scr = load_score_midi(self.tmpfile.name, part_voice_assign_mode=4)
        # this shold be a part
        part = scr.part_structure[0]
        msg = "{} should be a Part instance but it is not".format(part)
        self.assertTrue(isinstance(part, score.Part), msg)
        midi_notes = sum(self.notes_per_tr_ch.values())
        part_notes = len(part.notes)
        msg = "Part should have {} notes but it has".format(midi_notes, part_notes)
        self.assertEqual(midi_notes, part_notes, msg)

    def test_midi_import_mode_5(self):
        parts = load_score_midi(self.tmpfile.name, part_voice_assign_mode=5)
        msg = "Number of parts should be {} but it is {}".format(
            len(self.notes_per_tr_ch), len(parts)
        )
        self.assertEqual(len(parts), len(self.notes_per_tr_ch), msg)
        for part, trch_notes in zip(parts, self.notes_per_tr_ch.values()):
            part_notes = len(part.notes)
            msg = "Part should have {} notes but it has".format(trch_notes, part_notes)
            self.assertEqual(part_notes, trch_notes, msg)

    def tearDown(self):
        # remove tmp file
        self.tmpfile = None

class TestScoreMidi(unittest.TestCase):
    def test_time_signature(self):
        score = load_score_midi(MIDIINPORT_TESTFILES[0])
        self.assertEqual(score.note_array()["onset_beat"][2], 0.5)
        na = score.note_array(include_time_signature=True)
        self.assertTrue(all([n==3 for n in na["ts_beats"]]))
        self.assertTrue(all([d==8 for d in na["ts_beat_type"]]))
        
class TestLoadPerformanceMIDI(unittest.TestCase):
    def setUp(self):
        # create test MIDI file with tempo changes, time signatures, and key signatures
        self.midi_file = mido.MidiFile()
        track = mido.MidiTrack()
        self.midi_file.tracks.append(track)

        # set initial tempo (120 BPM)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        # set initial time signature (4/4)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        # set initial key signature (C major)
        track.append(mido.MetaMessage('key_signature', key='C', time=0))
        # add note
        track.append(mido.Message('note_on', note=60, velocity=64, time=0))
        track.append(mido.Message('note_off', note=60, velocity=64, time=480))
        # change tempo to 60 BPM
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(60), time=0))
        # change time signature to 3/4
        track.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
        # change key signature to G major
        track.append(mido.MetaMessage('key_signature', key='G', time=0))
        # add another note
        track.append(mido.Message('note_on', note=62, velocity=64, time=0))
        track.append(mido.Message('note_off', note=62, velocity=64, time=480))

    def test_adjust_time(self):
        
        performance = load_performance_midi(self.midi_file)

        notes = performance.performedparts[0].notes
        self.assertAlmostEqual(notes[0]['note_on'], 0.0, places=6)
        self.assertAlmostEqual(notes[0]['note_off'], 0.5, places=6)  # 120 BPM -> 0.5 seconds
        self.assertAlmostEqual(notes[1]['note_on'], 0.5, places=6)
        self.assertAlmostEqual(notes[1]['note_off'], 1.5, places=6)  # 60 BPM -> 1 second

    def test_time_signature_and_key_signature(self):
        
        performance = load_performance_midi(self.midi_file).performedparts[0]

        time_signatures = performance.time_signatures
        key_signatures = performance.key_signatures

        self.assertEqual(time_signatures[0]['beats'], 4)
        self.assertEqual(time_signatures[0]['beat_type'], 4)
        self.assertAlmostEqual(time_signatures[0]['time'], 0.0, places=6)

        self.assertEqual(time_signatures[1]['beats'], 3)
        self.assertEqual(time_signatures[1]['beat_type'], 4)
        self.assertAlmostEqual(time_signatures[1]['time'], 0.5, places=6)  # 120 BPM -> 0.5 seconds

        self.assertEqual(key_signatures[0]['key_name'], 'C')
        self.assertAlmostEqual(key_signatures[0]['time'], 0.0, places=6)

        self.assertEqual(key_signatures[1]['key_name'], 'G')
        self.assertAlmostEqual(key_signatures[1]['time'], 0.5, places=6)  # 120 BPM -> 0.5 seconds