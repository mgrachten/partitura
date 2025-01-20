#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for MusicXML import and export.
"""

import logging
import unittest
from tempfile import TemporaryFile

from tests import (
    MUSICXML_IMPORT_EXPORT_TESTFILES,
    MUSICXML_SCORE_OBJECT_TESTFILES,
    MUSICXML_TUPLET_ATTRIBUTES_TESTFILES,
    MUSICXML_UNFOLD_TESTPAIRS,
    MUSICXML_UNFOLD_COMPLEX,
    MUSICXML_UNFOLD_VOLTA,
    MUSICXML_UNFOLD_DACAPO,
)

from partitura import load_musicxml, save_musicxml
from partitura.directions import parse_direction
from partitura.utils import show_diff
import partitura.score as score
from lxml import etree

LOGGER = logging.getLogger(__name__)


class TestDirectionParser(unittest.TestCase):
    """
    Test if the direction parser gives the expected results for some directions
    """

    cases = [
        ("Andante", [score.ConstantTempoDirection]),
        ("ligato", [score.ConstantArticulationDirection]),
        ("sempre cresc", [score.IncreasingLoudnessDirection]),
        ("poco a poco rallentando", [score.DecreasingTempoDirection]),
    ]

    def test_parser(self):
        for words, target in self.cases:
            result = parse_direction(words)
            self.assertEqual(
                len(result),
                len(target),
                '"{}" not parsed correctly into directions'.format(words),
            )
            for res, trg in zip(result, target):
                self.assertEqual(type(res), trg, "")


class TestMusicXML(unittest.TestCase):
    """
    Test if importing and subsequent exporting restores the original input character by character
    """

    def test_import_export(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            with open(fn) as f:
                parts = load_musicxml(f, validate=False)
                result = save_musicxml(parts).decode("UTF-8")
                f.seek(0)
                target = f.read()
                equal = target == result
                if not equal:
                    show_diff(result, target)
                msg = "Import and export of MusicXML of file {} does not yield identical result".format(
                    fn
                )
                self.assertTrue(equal, msg)

    def test_unfold_timeline(self):
        for fn, fn_target_1, fn_target_2 in MUSICXML_UNFOLD_TESTPAIRS:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part, update_ids=False)
            # Load Target
            with open(fn_target_1) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

            # check unfold with update_id
            part = score.unfold_part_maximal(part, update_ids=True)
            result = save_musicxml(part).decode("UTF-8")
            with open(fn_target_2) as f:
                target = f.read()
            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_complex(self):
        for fn, fn_target in MUSICXML_UNFOLD_COMPLEX:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part)
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding complex part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_volta(self):
        for fn, fn_target in MUSICXML_UNFOLD_VOLTA:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part)
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding volta part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_dacapo(self):
        for fn, fn_target in MUSICXML_UNFOLD_DACAPO:
            sc = load_musicxml(fn, validate=False)
            part = score.unfold_part_maximal(sc[0])
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding volta part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_export_import_pprint(self):
        # create a part
        part1 = score.Part("My Part")

        # create contents
        divs = 10
        ts = score.TimeSignature(3, 4)
        page1 = score.Page(1)
        system1 = score.System(1)
        measure1 = score.Measure(number=1, name='1')
        note1 = score.Note(step="A", octave=4, voice=1, staff=1)
        rest1 = score.Rest(voice=1, staff=1)
        note2 = score.Note(step="C", octave=5, alter=-1, voice=2, staff=1)

        # and add the contents to the part:
        part1.set_quarter_duration(0, divs)
        part1.add(ts, 0)
        part1.add(measure1, 0, 30)
        part1.add(page1, 0)
        part1.add(system1, 0)
        part1.add(note1, 0, 15)
        part1.add(rest1, 15, 30)
        part1.add(note2, 0, 30)

        score.set_end_times(part1)

        # pretty print the part
        pstring1 = part1.pretty()

        with TemporaryFile() as f:
            # save part to musicxml
            save_musicxml(part1, f)
            f.flush()
            f.seek(0)
            # load part from musicxml
            part2 = load_musicxml(f)[0]

        # pretty print saved/loaded part:
        pstring2 = part2.pretty()

        # test pretty printed strings for equality
        equal = pstring1 == pstring2
        print('equal:', equal)

        if not equal:
            show_diff(pstring1, pstring2)
        msg = "Exported and imported score does not yield identical pretty printed representations"
        self.assertTrue(equal, msg)

    def test_export_import_tuplet(self):
        part1 = make_part_tuplet()
        self._pretty_export_import_pretty_test(part1)

    def test_export_import_slur(self):
        part1 = make_part_slur()
        self._pretty_export_import_pretty_test(part1)

    def test_stem_direction_import(self):
        # test if stem direction is imported correctly for the first note of test_note_ties.xml
        part = load_musicxml(MUSICXML_IMPORT_EXPORT_TESTFILES[0])[0]
        self.assertEqual(part.notes_tied[0].stem_direction, "up")

    def test_tuplet_attributes(self):
        part = load_musicxml(MUSICXML_TUPLET_ATTRIBUTES_TESTFILES[0])[0]
        tuplets = list(part.iter_all(cls=score.Tuplet))
        real_values = [
            (3, 2, "eighth"),
            (5, 4, "eighth"),
            (3, 2, "16th"),
            (2, 3, "eighth"),
            (5, 3, "eighth"),
        ]
        for tuplet, (actual, normal, note_type) in zip(tuplets, real_values):
            self.assertEqual(tuplet.actual_notes, actual)
            self.assertEqual(tuplet.normal_notes, normal)
            self.assertEqual(tuplet.type, note_type)

    def _pretty_export_import_pretty_test(self, part1):
        # pretty print the part
        pstring1 = part1.pretty()
        with TemporaryFile() as f:
            # save part to musicxml
            save_musicxml(part1, f)
            f.flush()
            f.seek(0)
            _tmp = f.read().decode("utf8")
            f.seek(0)
            # load part from musicxml
            part2 = load_musicxml(f)[0]

        # pretty print saved/loaded part:
        pstring2 = part2.pretty()

        # test pretty printed strings for equality
        equal = pstring1 == pstring2

        if not equal:
            print("pretty original:")
            print(pstring1)
            print("pretty reloaded:")
            print(pstring2)
            print("saved xml:")
            print(_tmp)
            print("diff:")
            show_diff(pstring1, pstring2)
        msg = "Exported and imported score does not yield identical pretty printed representations"
        self.assertTrue(equal, msg)
        
    def test_score_attribute(self):
        score = load_musicxml(MUSICXML_SCORE_OBJECT_TESTFILES[0])
        test_work_title = "Test Title"
        test_work_number = "Test Opus 1"

        self.assertTrue(score.work_title == test_work_title)
        self.assertTrue(score.work_number == test_work_number)


def make_part_slur():
    # create a part
    part = score.Part("My Part")
    # create contents
    divs = 12
    ts = score.TimeSignature(3, 4)
    page1 = score.Page(1)
    system1 = score.System(1)

    note0 = score.Note(id="n0", step="A", octave=4, voice=1, staff=1)
    note1 = score.Note(id="n1", step="A", octave=4, voice=1, staff=1)
    note2 = score.Note(id="n2", step="A", octave=4, voice=1, staff=1)
    note3 = score.Note(id="n3", step="A", octave=4, voice=1, staff=1)

    note4 = score.Note(id="n4", step="A", octave=3, voice=2, staff=1)
    note5 = score.Note(id="n5", step="A", octave=3, voice=2, staff=1)

    slur1 = score.Slur(start_note=note0, end_note=note5)
    slur2 = score.Slur(start_note=note4, end_note=note3)

    # and add the contents to the part:
    part.set_quarter_duration(0, divs)
    part.add(ts, 0)
    part.add(page1, 0)
    part.add(system1, 0)
    part.add(note0, 0, 12)
    part.add(note1, 12, 24)
    part.add(note2, 24, 36)
    part.add(note3, 36, 48)
    part.add(note4, 0, 6)
    part.add(note5, 6, 33)

    part.add(slur1, slur1.start_note.start.t, slur1.end_note.end.t)
    part.add(slur2, slur2.start_note.start.t, slur2.end_note.end.t)

    score.add_measures(part)
    score.tie_notes(part)
    score.set_end_times(part)
    return part


def make_part_tuplet():
    # create a part
    part = score.Part("My Part")

    # create contents
    divs = 12
    ts = score.TimeSignature(3, 4)
    page1 = score.Page(1)
    system1 = score.System(1)

    note1 = score.Note(id="n0", step="A", octave=4, voice=1, staff=1)
    rest1 = score.Rest(voice=1, staff=1)
    note2 = score.Note(id="n2", step="C", octave=4, voice=1, staff=1)
    rest2 = score.Rest(id="r0", voice=1, staff=1)

    # and add the contents to the part:
    part.set_quarter_duration(0, divs)
    part.add(ts, 0)
    part.add(page1, 0)
    part.add(system1, 0)
    part.add(note1, 0, 8)
    part.add(rest1, 8, 16)
    part.add(note2, 16, 24)
    part.add(rest2, 24, 36)

    score.add_measures(part)
    score.find_tuplets(part)
    score.set_end_times(part)

    return part


if __name__ == "__main__":
    unittest.main()
