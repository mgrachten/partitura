import unittest
from partitura import load_musicxml, load_mei
import numpy as np
from tests import CROSS_STAFF_TESTFILES


class CrossStaffBeaming(unittest.TestCase):
    def test_cross_staff_single_part_musicxml(self):
        score = load_musicxml(CROSS_STAFF_TESTFILES[0])
        note_array = score.note_array(include_staff=True)
        expected_staff = np.array([1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1])
        cross_staff_mask = (note_array["pitch"] > 52) & (note_array["pitch"] < 72)
        note_array_staff = note_array[cross_staff_mask]["staff"]
        expected_voice = np.ones(len(note_array_staff), dtype=int)
        note_array_voice = note_array[cross_staff_mask]["voice"]
        self.assertTrue(np.all(note_array_staff == expected_staff))
        self.assertTrue(np.all(note_array_voice == expected_voice))

class CrossStaffVoices(unittest.TestCase):
    def test_music_xml(self):
        score = load_musicxml(CROSS_STAFF_TESTFILES[1])
        note_array = score.note_array(include_staff=True)
        expected_staff = [2,1,1,2,1,2,1,1]
        expected_voice = [5,2,1,5,5,5,5,5]
        self.assertEqual(note_array["staff"].tolist(), expected_staff)
        self.assertEqual(note_array["voice"].tolist(), expected_voice)
    
    def test_mei(self):
        score = load_mei(CROSS_STAFF_TESTFILES[2])
        note_array = score.note_array(include_staff=True)
        expected_staff = [2,1,1,2,1,2,1,1]
        expected_voice = [5,2,1,5,5,5,5,5]
        self.assertEqual(note_array["staff"].tolist(), expected_staff)
        self.assertEqual(note_array["voice"].tolist(), expected_voice)