#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for methods for generating note-level features.
"""
import unittest
from tests import (
    METRICAL_POSITION_TESTFILES,
    MUSICXML_IMPORT_EXPORT_TESTFILES,
    MEI_TESTFILES,
    MUSICXML_NOTE_FEATURES,
)
from partitura import load_musicxml, load_mei
from partitura.musicanalysis import make_note_feats, compute_note_array
import numpy as np


class TestingNoteFeatureExtraction(unittest.TestCase):
    def test_metrical_basis(self):
        for fn in METRICAL_POSITION_TESTFILES:
            score = load_musicxml(fn, force_note_ids = "keep")
            make_note_feats(score[0], ["metrical_feature"])

    def test_grace_basis(self):
        fn = [f for f in MEI_TESTFILES if f.endswith("test_grace_note.mei")][0]
        part = load_mei(fn)
        make_note_feats(part, ["grace_feature"])

    def test_all_basis(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            score = load_musicxml(fn)
            make_note_feats(score[0], "all")

    def test_slur_grace_art_dyn_orn(self):
        for fn in MUSICXML_NOTE_FEATURES:
            score = load_musicxml(fn, force_note_ids=True)
            feats = [
                "ornament_feature",
                "articulation_feature",
                "grace_feature",
                "loudness_direction_feature",
                "slur_feature",
            ]
            na = compute_note_array(score[0], feature_functions=feats)
            stactest = na["articulation_feature.staccato"] == np.array(
                [1, 0, 0, 0, 0, 0]
            )
            tentest = na["articulation_feature.tenuto"] == np.array([0, 1, 0, 0, 0, 0])
            trilltest = na["ornament_feature.trill-mark"] == np.array(
                [0, 0, 1, 0, 0, 0]
            )
            gracetest = na["grace_feature.grace_note"] == np.array([0, 0, 0, 1, 0, 1])
            dyntest = na["loudness_direction_feature.f"] == np.array([0, 0, 0, 1, 1, 1])
            slurtest = na["slur_feature.slur_decr"] == np.array([0, 0, 0, 1, 1, 1])
            self.assertTrue(np.all(stactest), "staccato feature does not match")
            self.assertTrue(np.all(tentest), "tenuto feature does not match")
            self.assertTrue(np.all(trilltest), "trill feature does not match")
            self.assertTrue(np.all(gracetest), "grace feature does not match")
            self.assertTrue(np.all(dyntest), "forte feature does not match")
            self.assertTrue(np.all(slurtest), "slur feature does not match")

    def test_measure_feature(self):
        for fn in MUSICXML_NOTE_FEATURES:
            score = load_musicxml(fn, force_note_ids=True)
            feats = [
                "measure_feature"
            ]
            na = compute_note_array(score[0], feature_functions=feats)

            numtest = na["measure_feature.measure_number"] == np.array([1, 1, 1, 2, 2, 2])
            starttest = na["measure_feature.measure_start_beat"] == np.array([0, 0, 0, 4, 4, 4])
            endtest = na["measure_feature.measure_end_beat"] == np.array([4, 4, 4, 8, 8, 8])
            self.assertTrue(np.all(numtest), "measure number feature does not match")
            self.assertTrue(np.all(starttest), "measure start feature does not match")
            self.assertTrue(np.all(endtest), "measure end feature does not match")



if __name__ == "__main__":
    unittest.main()
