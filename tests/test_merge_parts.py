#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utilities for merging parts.
"""
import numpy as np
import logging
import unittest
from pathlib import Path

from partitura import load_musicxml, load_mei
from partitura.score import merge_parts, Part, iter_parts
from partitura.utils.music import ensure_notearray

from tests import MERGE_PARTS_TESTFILES

LOGGER = logging.getLogger(__name__)


class TestMergeParts(unittest.TestCase):
    """
    Test merge parts utility
    """

    def test_list_of_parts_and_partgroup(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[1])
        merged_part = merge_parts(score.parts)
        note_array = merged_part.note_array()
        expected_onsets = [0, 0, 0, 0, 0, 0, 12, 15, 24, 24, 32, 40, 48]
        expected_pitches = [48, 50, 53, 62, 67, 69, 64, 65, 47, 69, 67, 64, 60]
        expected_duration = [24, 24, 24, 12, 24, 24, 3, 3, 24, 8, 8, 8, 24]
        self.assertTrue(np.array_equal(note_array["onset_div"], expected_onsets))
        self.assertTrue(np.array_equal(note_array["pitch"], expected_pitches))
        self.assertTrue(np.array_equal(note_array["duration_div"], expected_duration))

    def test_different_divs(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[1])
        merged_part = merge_parts(score.parts)
        note_array = merged_part.note_array()
        expected_onsets = [0, 0, 0, 0, 0, 0, 12, 15, 24, 24, 32, 40, 48]
        expected_pitches = [48, 50, 53, 62, 67, 69, 64, 65, 47, 69, 67, 64, 60]
        self.assertTrue(np.array_equal(note_array["onset_div"], expected_onsets))
        self.assertTrue(np.array_equal(note_array["pitch"], expected_pitches))

    def test_compare_normal_and_different_divs(self):
        score_normal = load_musicxml(MERGE_PARTS_TESTFILES[1])
        score_diff = load_musicxml(MERGE_PARTS_TESTFILES[2])
        merged_part_normal = merge_parts(score_normal.parts)
        merged_part_diff = merge_parts(score_diff.parts)
        note_array_normal = merged_part_normal.note_array()
        note_array_diff = merged_part_diff.note_array()
        self.assertTrue(
            np.array_equal(
                note_array_normal["onset_beat"], note_array_diff["onset_beat"]
            )
        )

    def test_merge_single_part(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[3])
        merged_part = merge_parts(score.parts)
        self.assertTrue(merged_part == score[0])

    # def test_merge_interpolation(self):
    #     parts = load_musicxml(MERGE_PARTS_TESTFILES[4])
    #     merged_part = merge_parts(parts)
    #     self.assertTrue(isinstance(merged_part, Part))

    def test_reassign_voices(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[6])
        merged_part = merge_parts(score.parts, reassign="voice")
        note_array = merged_part.note_array(include_staff=True)
        expected_voices = [3, 2, 1, 1]
        expected_staves = [1, 1, 1, 1]
        self.assertTrue(note_array["voice"].tolist() == expected_voices)
        self.assertTrue(note_array["staff"].tolist() == expected_staves)

    def test_reassign_voices2(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[7])
        merged_part = merge_parts(score.parts, reassign="voice")
        note_array = merged_part.note_array(include_staff=True)
        expected_voices = [4, 4, 3, 2, 1, 1]
        expected_staves = [2, 1, 1, 1, 1, 1]
        self.assertTrue(note_array["voice"].tolist() == expected_voices)
        self.assertTrue(note_array["staff"].tolist() == expected_staves)

    def test_reassign_staves(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[6])
        merged_part = merge_parts(score.parts, reassign="staff")
        note_array = merged_part.note_array(include_staff=True)
        expected_voices = [1, 2, 1, 1]
        expected_staves = [2, 1, 1, 1]
        self.assertTrue(note_array["voice"].tolist() == expected_voices)
        self.assertTrue(note_array["staff"].tolist() == expected_staves)

    def test_reassign_staves2(self):
        score = load_musicxml(MERGE_PARTS_TESTFILES[7])
        merged_part = merge_parts(score.parts, reassign="staff")
        note_array = merged_part.note_array(include_staff=True)
        expected_voices = [1, 1, 1, 2, 1, 1]
        expected_staves = [4, 3, 2, 1, 1, 1]
        self.assertTrue(note_array["voice"].tolist() == expected_voices)
        self.assertTrue(note_array["staff"].tolist() == expected_staves)

    def test_reassign_auto(self):
        score = load_mei(MERGE_PARTS_TESTFILES[8])
        merged_part = merge_parts(score.parts, reassign="auto")
        note_array = merged_part.note_array(include_staff=True)
        expected_voices = [13, 9, 5, 2, 1, 1]
        expected_staves = [4, 3, 2, 1, 1, 1 ]
        self.assertEqual(note_array["voice"].tolist(),expected_voices)
        self.assertEqual(note_array["staff"].tolist(),expected_staves)
