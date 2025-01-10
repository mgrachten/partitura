#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for the metrical position computation
"""

import unittest
from partitura import load_musicxml
from partitura.utils.music import note_array_from_part
from partitura.score import TimeSignature

import numpy as np

from tests import (
    METRICAL_POSITION_TESTFILES,
    TIME_SIGNATURE_MAP_EDGECASES_TESTFILES,
)


class TestMetricalPosition(unittest.TestCase):
    def test_measure_map(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[0])
        notes = score[0].notes
        measures_for_notes = score[0].measure_map([n.start.t for n in notes])
        expected_measures_for_notes = (
            np.array(  # the start and end of measures that we expect
                [(16 - 48, 16) for i in range(2)]
                + [(16, 64) for i in range(6)]
                + [(64, 128) for i in range(5)]
                + [(128, 200) for i in range(8)]
                + [(200, 232) for i in range(2)]
            )
        )
        self.assertTrue(np.array_equal(measures_for_notes, expected_measures_for_notes))

    def test_metrical_position_map(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[0])
        note_array = note_array_from_part(score[0], include_metrical_position=True)
        expected_notes_on_downbeat = [2, 3, 8, 9, 13, 21]
        expected_rel_positions = [
            32,
            40,
            0,
            0,
            24,
            28,
            32,
            40,
            0,
            0,
            24,
            48,
            56,
            0,
            16,
            32,
            33,
            34,
            36,
            40,
            48,
            0,
            16,
        ]
        expected_measure_duration = (
            [48 for i in range(2)]
            + [48 for i in range(6)]
            + [64 for i in range(5)]
            + [72 for i in range(8)]
            + [32 for i in range(2)]
        )
        # test if all downbeats are in the right place
        self.assertTrue(
            np.array_equal(
                np.flatnonzero(note_array["is_downbeat"]), expected_notes_on_downbeat
            )
        )
        self.assertTrue(
            np.array_equal(note_array["rel_onset_div"], expected_rel_positions)
        )
        self.assertTrue(
            np.array_equal(
                note_array["tot_measure_div"],
                expected_measure_duration,
            )
        )

    def test_measure_number_map(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[0])
        notes = score[0].notes
        measure_numbers_for_notes = score[0].measure_number_map(
            [n.start.t for n in notes]
        )
        expected_measure_numbers_for_notes = (
            np.array(  # the start and end of measures that we expect
                [1 for i in range(2)]
                + [2 for i in range(6)]
                + [3 for i in range(5)]
                + [4 for i in range(8)]
                + [5 for i in range(2)]
            )
        )
        self.assertTrue(
            np.array_equal(
                measure_numbers_for_notes, expected_measure_numbers_for_notes
            )
        )

    def test_anacrusis_downbeat(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = note_array_from_part(score[0], include_metrical_position=True)
        # first note on the anacrusis is not a downbeat
        self.assertTrue(note_array["is_downbeat"][0] == 0)
        self.assertTrue(note_array["rel_onset_div"][0] == 3)


class TestTimeSignatureMap(unittest.TestCase):
    def test_time_signature_map(self):
        for fn in TIME_SIGNATURE_MAP_EDGECASES_TESTFILES:
            score = load_musicxml(fn)

            for part in score:

                tss = np.array(
                    [
                        (ts.start.t, ts.beats, ts.beat_type, ts.musical_beats)
                        for ts in part.iter_all(TimeSignature)
                    ]
                )

                self.assertTrue(
                    np.all(part.time_signature_map(part.first_point.t) == tss[0, 1:])
                )




if __name__ == "__main__":
    unittest.main()
