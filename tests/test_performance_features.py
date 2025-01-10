#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Performance Array Calculations
"""
import unittest
import numpy as np
from partitura import load_match
from tests import MATCH_EXPRESSIVE_FEATURES_TESTFILES
from partitura.musicanalysis.performance_features import make_performance_features
import os


class TestPerformanceFeatures(unittest.TestCase):
    def test_performance_features(self):
        fields = ['id',
                  'articulation_feature.kor', 'pedal_feature.onset_value', 'pedal_feature.offset_value', 'pedal_feature.to_prev_release',
                  'pedal_feature.to_next_release', 'onset', 'duration', 'pitch', 'p_onset', 'p_duration', 'velocity', 'beat_period']
        True_array = np.array(
            [('n1', 0.23374297,  89.74999,  62.000057, 0., 0.16015087, -0.5, 0.5, 59, 4.9925, 0.8775, 44, 1.4700003),
            ('n4', 0.03011051, 114.25004,  61.000244, 0.,
            0.4027142,  0., 1., 40, 5.7025, 2.4375, 22, 2.8474998),
            ('n3', 0.8451489,  87.500046, 113., 0., 1.5302137,  0., 0.25, 56, 5.77625, 1.23625, 26, 2.8474998)],
                              dtype=[('id', '<U256'),
                                     ('articulation_feature.kor', '<f4'),
                                     ('pedal_feature.onset_value', '<f4'),
                                     ('pedal_feature.offset_value', '<f4'),
                                     ('pedal_feature.to_prev_release', '<f4'),
                                     ('pedal_feature.to_next_release', '<f4'),
                                     ('onset', '<f4'),
                                     ('duration', '<f4'),
                                     ('pitch', '<i4'),
                                     ('p_onset', '<f4'),
                                     ('p_duration', '<f4'),
                                     ('velocity', '<i4'),
                                     ('beat_period', '<f4')])
        fn = MATCH_EXPRESSIVE_FEATURES_TESTFILES[0]
        perf, alignment, score = load_match(filename=fn, create_score=True)
        features = make_performance_features(score,
                                             perf,
                                             alignment,
                                             feature_functions="all")

        self.assertTrue(np.all(True_array[fields] == features[fields][:3]),
                        f"The expression features don't match the original.")

