#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains test functions for MEI import
"""

import unittest

from tests import MEI_TESTFILES, MUSICXML_PATH
from partitura import load_musicxml, load_mei, EXAMPLE_MEI, save_mei
import partitura.score as score
from partitura.io.importmei import MeiParser
from partitura.utils import compute_pianoroll
from lxml import etree
from tempfile import TemporaryDirectory
from xmlschema.names import XML_NAMESPACE
import os
import numpy as np
from partitura.score import merge_parts


# class TestSaveMEI(unittest.TestCase):

#     def test_save_mei(self):

#         with open(EXAMPLE_MEI, 'r') as f:
#             target_mei = f.read()

#         mei = save_mei(load_musicxml(EXAMPLE_MUSICXML), title_text='score_example')
#         msg = "Export of MEI of file {} does not yield identical result".format(EXAMPLE_MEI)

#         self.assertTrue(mei.decode('utf-8') == target_mei, msg)

class TestExportMEI(unittest.TestCase):
    def test_export_mei_simple(self):
        import_score = load_mei(EXAMPLE_MEI)
        ina = import_score.note_array()
        with TemporaryDirectory() as tmpdir:
            tmp_mei = os.path.join(tmpdir, "test.mei")
            merged_mei = merge_parts(import_score.parts, reassign="auto")
            save_mei(merged_mei, tmp_mei)
            export_score = load_mei(tmp_mei)
            ena = export_score.note_array()
            self.assertTrue(np.all(ina["onset_beat"] == ena["onset_beat"]))
            self.assertTrue(np.all(ina["duration_beat"] == ena["duration_beat"]))
            self.assertTrue(np.all(ina["pitch"] == ena["pitch"]))
            self.assertTrue(np.all(ina["voice"] == ena["voice"]))
            self.assertTrue(np.all(ina["id"] == ena["id"]))

    def test_export_mei(self):
        import_score = load_musicxml(os.path.join(MUSICXML_PATH, "test_chew_vosa_example.xml"), force_note_ids=True)
        ina = import_score.note_array()
        with TemporaryDirectory() as tmpdir:
            tmp_mei = os.path.join(tmpdir, "test.mei")
            save_mei(import_score, tmp_mei)
            export_score = load_mei(tmp_mei)
            ena = export_score.note_array()
            self.assertTrue(np.all(ina["onset_beat"] == ena["onset_beat"]))
            self.assertTrue(np.all(ina["duration_beat"] == ena["duration_beat"]))
            self.assertTrue(np.all(ina["pitch"] == ena["pitch"]))

    def test_export_with_harmony(self):
        score_fn = os.path.join(MUSICXML_PATH, "test_harmony.musicxml")
        import_score = load_musicxml(score_fn)
        with TemporaryDirectory() as tmpdir:
            tmp_mei = os.path.join(tmpdir, "test.mei")
            save_mei(import_score, tmp_mei)


class TestImportMEI(unittest.TestCase):
    def test_main_part_group1(self):
        parser = MeiParser(MEI_TESTFILES[1])
        main_partgroup_el = parser.document.find(parser._ns_name("staffGrp", all=True))
        part_list = parser._handle_main_staff_group(main_partgroup_el)
        self.assertTrue(len(part_list) == 2)
        # first partgroup
        self.assertTrue(isinstance(part_list[0], score.PartGroup))
        self.assertTrue(part_list[0].group_symbol == "bracket")
        self.assertTrue(part_list[0].group_name is None)
        self.assertTrue(part_list[0].id == "sl1ipm2")
        # first partgroup first part
        self.assertTrue(part_list[0].children[0].id == "P1")
        self.assertTrue(part_list[0].children[0].part_name == "S")
        self.assertTrue(part_list[0].children[0]._quarter_durations[0] == 12)
        # first partgroup second part
        self.assertTrue(part_list[0].children[1].id == "P2")
        self.assertTrue(part_list[0].children[1].part_name == "A")
        self.assertTrue(part_list[0].children[1]._quarter_durations[0] == 12)
        # first partgroup third part
        self.assertTrue(part_list[0].children[2].id == "P3")
        self.assertTrue(part_list[0].children[2].part_name == "T")
        self.assertTrue(part_list[0].children[2]._quarter_durations[0] == 12)
        # first partgroup fourth part
        self.assertTrue(part_list[0].children[3].id == "P4")
        self.assertTrue(part_list[0].children[3].part_name == "B")
        self.assertTrue(part_list[0].children[3]._quarter_durations[0] == 12)
        # second partgroup
        self.assertTrue(isinstance(part_list[1], score.PartGroup))
        self.assertTrue(part_list[1].group_symbol == "brace")
        self.assertTrue(part_list[1].group_name == "Piano")
        self.assertTrue(part_list[1].id == "P5")

    def test_main_part_group2(self):
        parser = MeiParser(MEI_TESTFILES[0])
        main_partgroup_el = parser.document.find(parser._ns_name("staffGrp", all=True))
        part_list = parser._handle_main_staff_group(main_partgroup_el)
        self.assertTrue(len(part_list) == 1)
        self.assertTrue(isinstance(part_list[0], score.PartGroup))

    def test_handle_layer1(self):
        parser = MeiParser(MEI_TESTFILES[1])
        layer_el = [
            e
            for e in parser.document.findall(parser._ns_name("layer", all=True))
            if e.attrib[parser._ns_name("id", XML_NAMESPACE)] == "l3ss4q5"
        ][0]
        part = score.Part("dummyid", quarter_duration=12)
        parser._handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part)
        self.assertTrue(len(part.note_array()) == 3)

    def test_handle_layer2(self):
        parser = MeiParser(MEI_TESTFILES[1])
        layer_el = [
            e
            for e in parser.document.findall(parser._ns_name("layer", all=True))
            if e.attrib[parser._ns_name("id", XML_NAMESPACE)] == "l95j799"
        ][0]
        part = score.Part("dummyid", quarter_duration=12)
        parser._handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part)
        self.assertTrue(len(part.note_array()) == 3)

    def test_handle_layer_tuplets(self):
        parser = MeiParser(MEI_TESTFILES[2])
        layer_el = [
            e
            for e in parser.document.findall(parser._ns_name("layer", all=True))
            if e.attrib[parser._ns_name("id", XML_NAMESPACE)] == "l7hooah"
        ][0]
        part = score.Part("dummyid", quarter_duration=15)
        parser._handle_layer_in_staff_in_measure(layer_el, 1, 1, 0, part)
        self.assertTrue(len(part.note_array()) == 10)

    def test_ties1(self):
        scr = load_mei(MEI_TESTFILES[3])
        part_list = scr.parts
        note_array = list(score.iter_parts(part_list))[0].note_array()
        self.assertTrue(len(note_array) == 4)

    def test_time_signatures(self):
        scr = load_mei(MEI_TESTFILES[4])
        part_list = scr.parts
        part0 = list(score.iter_parts(part_list))[0]
        time_signatures = list(part0.iter_all(score.TimeSignature))
        self.assertTrue(len(time_signatures) == 3)
        self.assertTrue(time_signatures[0].start.t == 0)
        self.assertTrue(time_signatures[1].start.t == 8 * 16)
        self.assertTrue(time_signatures[2].start.t == 12.5 * 16)

    def test_clef(self):
        part_list = load_mei(MEI_TESTFILES[5]).parts
        # test on part 2
        part2 = list(score.iter_parts(part_list))[2]
        clefs2 = list(part2.iter_all(score.Clef))
        self.assertTrue(len(clefs2) == 2)
        self.assertTrue(clefs2[0].start.t == 0)
        self.assertTrue(clefs2[0].sign == "C")
        self.assertTrue(clefs2[0].line == 3)
        self.assertTrue(clefs2[0].staff == 3)
        self.assertTrue(clefs2[0].octave_change == 0)
        self.assertTrue(clefs2[1].start.t == 8)
        self.assertTrue(clefs2[1].sign == "F")
        self.assertTrue(clefs2[1].line == 4)
        self.assertTrue(clefs2[1].staff == 3)
        self.assertTrue(clefs2[1].octave_change == 0)
        # test on part 3
        part3 = list(score.iter_parts(part_list))[3]
        clefs3 = list(part3.iter_all(score.Clef))
        self.assertTrue(len(clefs3) == 2)
        self.assertTrue(clefs3[0].start.t == 0)
        self.assertTrue(clefs3[1].start.t == 4)
        self.assertTrue(clefs3[1].sign == "G")
        self.assertTrue(clefs3[1].line == 2)
        self.assertTrue(clefs3[1].staff == 4)
        self.assertTrue(clefs3[1].octave_change == -1)

    def test_key_signature1(self):
        part_list = load_mei(MEI_TESTFILES[5]).parts
        for part in score.iter_parts(part_list):
            kss = list(part.iter_all(score.KeySignature))
            self.assertTrue(len(kss) == 2)
            self.assertTrue(kss[0].fifths == 2)
            self.assertTrue(kss[1].fifths == 4)

    def test_key_signature2(self):
        part_list = load_mei(MEI_TESTFILES[6]).parts
        for part in score.iter_parts(part_list):
            kss = list(part.iter_all(score.KeySignature))
            self.assertTrue(len(kss) == 1)
            self.assertTrue(kss[0].fifths == -1)

    def test_grace_note(self):
        part_list = load_mei(MEI_TESTFILES[6]).parts
        part = list(score.iter_parts(part_list))[0]
        grace_notes = list(part.iter_all(score.GraceNote))
        self.assertTrue(len(part.note_array()) == 7)
        self.assertTrue(len(grace_notes) == 4)
        self.assertTrue(grace_notes[0].grace_type == "acciaccatura")
        self.assertTrue(grace_notes[1].grace_type == "appoggiatura")

    def test_meter_in_scoredef(self):
        part_list = load_mei(MEI_TESTFILES[7]).parts
        self.assertTrue(True)

    def test_infer_ppq(self):
        parser = MeiParser(MEI_TESTFILES[8])
        inferred_ppq = parser._find_ppq()
        self.assertTrue(inferred_ppq == 15)

    def test_no_ppq(self):
        # compare the same piece with and without ppq annotations
        parts_ppq = load_mei(MEI_TESTFILES[2]).parts
        part_ppq = list(score.iter_parts(parts_ppq))[0]
        note_array_ppq = part_ppq.note_array()

        parts_no_ppq = load_mei(MEI_TESTFILES[8]).parts
        part_no_ppq = list(score.iter_parts(parts_no_ppq))[0]
        note_array_no_ppq = part_no_ppq.note_array()

        self.assertTrue(np.array_equal(note_array_ppq, note_array_no_ppq))

    def test_part_duration(self):
        parts_no_ppq = load_mei(MEI_TESTFILES[10]).parts
        part_no_ppq = list(score.iter_parts(parts_no_ppq))[0]
        note_array_no_ppq = part_no_ppq.note_array()
        self.assertTrue(part_no_ppq._quarter_durations[0] == 4)
        self.assertTrue(sorted(part_no_ppq._points)[-1].t == 12)

    def test_part_duration2(self):
        parts_no_ppq = load_mei(MEI_TESTFILES[11]).parts
        part_no_ppq = list(score.iter_parts(parts_no_ppq))[0]
        note_array_no_ppq = part_no_ppq.note_array()
        self.assertTrue(part_no_ppq._quarter_durations[0] == 8)
        self.assertTrue(sorted(part_no_ppq._points)[-1].t == 22)

    def test_barline(self):
        parts = load_mei(MEI_TESTFILES[12]).parts
        part = list(score.iter_parts(parts))[0]
        barlines = list(part.iter_all(score.Barline))
        expected_barlines_times = [0, 8, 8, 16, 20, 24, 28]
        expected_barlines_style = [
            "heavy-light",
            "light-heavy",
            "heavy-light",
            "light-heavy",
            "light-light",
            "dashed",
            "light-heavy",
        ]
        self.assertTrue([bl.start.t for bl in barlines] == expected_barlines_times)
        self.assertTrue([bl.style for bl in barlines] == expected_barlines_style)

    def test_repetition1(self):
        parts = load_mei(MEI_TESTFILES[12]).parts
        part = list(score.iter_parts(parts))[0]
        repetitions = list(part.iter_all(score.Repeat))
        expected_repeat_starts = [0, 8]
        expected_repeat_ends = [8, 16]
        self.assertTrue([rp.start.t for rp in repetitions] == expected_repeat_starts)
        self.assertTrue([rp.end.t for rp in repetitions] == expected_repeat_ends)

    def test_repetition2(self):
        parts = load_mei(MEI_TESTFILES[13]).parts
        part = list(score.iter_parts(parts))[0]
        fine_els = list(part.iter_all(score.Fine))
        self.assertTrue(len(fine_els) == 1)
        self.assertTrue(fine_els[0].start.t == 12)
        dacapo_els = list(part.iter_all(score.DaCapo))
        self.assertTrue(len(dacapo_els) == 1)
        self.assertTrue(dacapo_els[0].start.t == 26)

    # def test_articulation(self):
    #     parts = load_mei(MEI_TESTFILES[17])
    #     part = list(score.iter_parts(parts))[0]
    #     self.assertTrue(False)

    def test_parse_mei_example(self):
        part_list = load_mei(EXAMPLE_MEI).parts
        self.assertTrue(True)

    def test_parse_mei(self):
        # check if all test files load correctly
        for mei in MEI_TESTFILES:
            print("loading {}".format(mei))
            part_list = load_mei(mei).parts
        self.assertTrue(True)

    def test_voice(self):
        parts = load_mei(MEI_TESTFILES[15])
        merged_part = score.merge_parts(parts, reassign="voice")
        voices = merged_part.note_array()["voice"]
        expected_voices = [5, 4, 3, 2, 1, 1]
        self.assertTrue(np.array_equal(voices, expected_voices))

    def test_staff(self):
        score = load_mei(MEI_TESTFILES[15])
        staves = score.note_array(include_staff=True)["staff"]
        expected_staves = [4, 3, 2, 1, 1, 1]
        self.assertTrue(np.array_equal(staves, expected_staves))

    def test_nopart(self):
        my_score = load_mei(MEI_TESTFILES[16])
        last_measure_duration = [
            list(p.iter_all(score.Barline))[-1].start.t
            - list(p.iter_all(score.Measure))[-1].start.t
            for p in my_score.parts
        ]
        self.assertTrue(all([d == 4096 for d in last_measure_duration]))

    def test_tuplet_div(self):
        score = load_mei(MEI_TESTFILES[17])
        self.assertTrue(np.array_equal(score.note_array()["duration_div"],[3,3,3,3,3,3,3,3,24]))

    def test_measure_number(self):
        score = load_mei(MEI_TESTFILES[0])
        measure_number_map = score.parts[0].measure_number_map
        onsets = score.note_array()["onset_div"]
        measure_number_per_each_onset = measure_number_map(onsets)
        self.assertTrue(measure_number_per_each_onset[0].dtype == int)
        self.assertTrue(min(measure_number_per_each_onset) == 1)
        self.assertTrue(max(measure_number_per_each_onset) == 34)

    def test_measure_number2(self):
        score = load_mei(MEI_TESTFILES[13])
        measure_number_map = score.parts[0].measure_number_map
        measure_number_per_each_onset = measure_number_map(score.note_array()["onset_div"])
        self.assertTrue(measure_number_per_each_onset.tolist()==[1,2,2,3,4,5,6,8])

if __name__ == "__main__":
    unittest.main()
