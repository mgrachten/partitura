#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for clef related methods.
"""
import unittest
from tests import (
    CLEF_TESTFILES,
    CLEF_MAP_TESTFILES
)
import numpy as np
from partitura import load_musicxml
from partitura.musicanalysis import compute_note_array
from partitura.musicanalysis.note_features import clef_feature
from partitura.score import merge_parts
import partitura

class TestingClefFeatureExtraction(unittest.TestCase):
    def test_clef_feature_exctraction(self):
        for fn in CLEF_TESTFILES:
            score = load_musicxml(fn, force_note_ids = "keep")
            sna1 = compute_note_array(score.parts[0],
                               feature_functions=["clef_feature"])
            sna2 = compute_note_array(score.parts[1],
                               feature_functions=["clef_feature"])
            mpart = merge_parts(score.parts, reassign="staff")
            sna3 = compute_note_array(mpart,
                               feature_functions=["clef_feature"])
            
            sna1test1 = sna1["clef_feature.clef_sign"] == np.array([1., 0., 2., 0., 0., 2., 0., 0., 1., 0.])
            sna1test2 = sna1["clef_feature.clef_line"] == np.array([4., 2., 3., 2., 2., 4., 2., 2., 4., 2.])
            sna1test3 = sna1["clef_feature.clef_octave_change"] == np.array([0.,  0.,  0.,  1.,  1.,  0., -1.,  0.,  0.,  0.])
            self.assertTrue(np.all(sna1test1), "clef sign does not match")
            self.assertTrue(np.all(sna1test2), "clef line does not match")
            self.assertTrue(np.all(sna1test3), "clef octave does not match")


            sna2test1 = sna2["clef_feature.clef_sign"] == np.array([1., 0.])
            sna2test2 = sna2["clef_feature.clef_line"] == np.array([4., 2.])
            sna2test3 = sna2["clef_feature.clef_octave_change"] == np.array([0.,  0.])
            self.assertTrue(np.all(sna2test1), "clef sign does not match")
            self.assertTrue(np.all(sna2test2), "clef line does not match")
            self.assertTrue(np.all(sna2test3), "clef octave does not match")

            sna3test1 = sna3["clef_feature.clef_sign"] == np.array([1., 0., 2., 0., 0., 1., 2., 0., 0., 0., 1., 0.])
            sna3test2 = sna3["clef_feature.clef_line"] == np.array([4., 2., 3., 2., 2., 4., 4., 2., 2., 2.,  4., 2.])
            sna3test3 = sna3["clef_feature.clef_octave_change"] == np.array([0.,  0.,  0.,  1.,  1., 0., 0., -1.,  0., 0., 0.,  0.])
            self.assertTrue(np.all(sna3test1), "clef sign does not match")
            self.assertTrue(np.all(sna3test2), "clef line does not match")
            self.assertTrue(np.all(sna3test3), "clef octave does not match")




class TestClefMap(unittest.TestCase):
    def test_clef_map(self):
        score = load_musicxml(CLEF_MAP_TESTFILES[0])
        for part in score:
            # clef = (staff_no, sign_shape, line, octave_shift)
            map_fn = part.clef_map
            self.assertTrue(
                np.all(map_fn(part.first_point.t) == np.array([[1, 0, 2, 0], [2, 1, 4, 0]]))  # treble / bass
            )
            self.assertTrue(
                np.all(map_fn(7) == np.array([[1, 0, 2, -2], [2, 0, 2, 0]]))  # treble15vb / treble
            )
            self.assertTrue(
                np.all(map_fn(8) == np.array([[1, 0, 2, 1], [2, 1, 4, 0]]))  # treble8va / bass
            )
            self.assertTrue(
                np.all(map_fn(11) == np.array([[1, 2, 3, 0], [2, 1, 4, 0]]))  # ut3 / bass
            )
            self.assertTrue(
                np.all(map_fn(12) == np.array([[1, 2, 3, 0], [2, 1, 3, 0]]))  # ut3 / bass3
            )
            self.assertTrue(
                np.all(map_fn(13) == np.array([[1, 2, 4, 0], [2, 1, 3, 0]]))  # ut4 / bass3
            )
            self.assertTrue(
                np.all(map_fn(part.last_point.t) == np.array([[1, 2, 4, 0], [2, 1, 3, 0]]))  # ut4 / bass3
            )


    def test_clef_map_multipart(self):
        score = load_musicxml(CLEF_TESTFILES[0])
        p1 = score.parts[0]
        p2 = score.parts[1]
        
        t = np.arange(16)
        target_p1_octave_change = np.array([ 0,  0,  0,  0,  1,  1,  1,  1, -1, -1,  0,  0,  0,  0,  0,  0])
        target_p1_line = np.array([4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4])
        map_fn = p1.clef_map
        self.assertTrue(np.all(map_fn(t)[0,:,3] == target_p1_octave_change)) 
        self.assertTrue(np.all(map_fn(t)[1,:,2] == target_p1_line)) 

        target_p2_sign = np.zeros(16) # 16 stepgs G clef, imputed missing clef in the beginning
        map_fn = p2.clef_map
        self.assertTrue(np.all(map_fn(t)[1,:,1] == target_p2_sign)) 


        p3 = merge_parts(score.parts, reassign="staff")
        map_fn = p3.clef_map
        self.assertTrue(np.all(map_fn(t)[0,:,3] == target_p1_octave_change)) 
        self.assertTrue(np.all(map_fn(t)[1,:,2] == target_p1_line)) 
        self.assertTrue(np.all(map_fn(t)[3,:,1] == target_p2_sign)) 
