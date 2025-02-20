import unittest
from partitura import load_score
import numpy as np


class TestImport(unittest.TestCase):
    def test_load_kern(self):
        score = load_score("https://raw.githubusercontent.com/CPJKU/partitura/refs/heads/main/partitura/assets/score_example.krn")
        note_array = score.note_array()
        self.assertTrue(np.all(note_array["pitch"] == [69, 72, 76]))

    def test_load_mei(self):
        score = load_score("https://raw.githubusercontent.com/CPJKU/partitura/refs/heads/main/partitura/assets/score_example.mei")
        note_array = score.note_array()
        self.assertTrue(np.all(note_array["pitch"] == [69, 72, 76]))

    def test_load_midi(self):
        score = load_score("https://raw.githubusercontent.com/CPJKU/partitura/refs/heads/main/partitura/assets/score_example.mid")
        note_array = score.note_array()
        self.assertTrue(np.all(note_array["pitch"] == [69, 72, 76]))

    def test_load_musicxml(self):
        score = load_score("https://raw.githubusercontent.com/CPJKU/partitura/refs/heads/main/partitura/assets/score_example.musicxml")
        note_array = score.note_array()
        self.assertTrue(np.all(note_array["pitch"] == [69, 72, 76]))

