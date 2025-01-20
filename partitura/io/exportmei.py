#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting MEI files.
"""
import math
from collections import defaultdict
from lxml import etree
import lxml
import partitura.score as spt
from operator import itemgetter
from itertools import groupby
from typing import Optional
from partitura.utils import (
    partition,
    iter_current_next,
    to_quarter_tempo,
    fifths_mode_to_key_name,
)
import numpy as np
import warnings
from partitura.utils.misc import deprecated_alias, PathLike
from partitura.utils.music import MEI_DURS_TO_SYMBOLIC, estimate_symbolic_duration


__all__ = ["save_mei"]

XMLNS_ID = "{http://www.w3.org/XML/1998/namespace}id"

ALTER_TO_MEI = {
    -2: "ff",
    -1: "f",
    0: "n",
    1: "s",
    2: "ss",
}

SYMBOLIC_TYPES_TO_MEI_DURS = {v: k for k, v in MEI_DURS_TO_SYMBOLIC.items()}
SYMBOLIC_TYPES_TO_MEI_DURS["h"] = "2"
SYMBOLIC_TYPES_TO_MEI_DURS["e"] = "8"
SYMBOLIC_TYPES_TO_MEI_DURS["q"] = "4"

DOCTYPE = '<?xml-model href="https://music-encoding.org/schema/4.0.1/mei-CMN.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>\n<?xml-model href="https://music-encoding.org/schema/4.0.1/mei-CMN.rng" type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>'


class MEIExporter:
    def __init__(self, part, title=None):
        self.part = part
        self.qdivs = part._quarter_durations[0]
        self.num_staves = part.number_of_staves
        self.title = title
        self.element_counter = 0
        self.current_key_signature = []
        self.flats = ["bf", "ef", "af", "df", "gf", "cf", "ff"]
        self.sharps = ["fs", "cs", "gs", "ds", "as", "es", "bs"]

    def elc_id(self):
        # transforms an integer number to 8-digit string
        # The number is right aligned and padded with zeros
        self.element_counter += 1
        out = str(self.element_counter).zfill(10)
        return out

    def export_to_mei(self):
        # Create root MEI element
        etree.register_namespace("xml", "http://www.w3.org/XML/1998/namespace")
        etree.register_namespace("mei", "http://www.music-encoding.org/ns/mei")
        mei = etree.Element(
            "mei",
            nsmap={
                "xml": "http://www.w3.org/XML/1998/namespace",
                None: "http://www.music-encoding.org/ns/mei",
            },
        )
        # mei.set('xmlns', "http://www.music-encoding.org/ns/mei")
        mei.set("meiversion", "4.0.1")
        # Create child elements
        mei_head = etree.SubElement(mei, "meiHead")
        file_desc = etree.SubElement(mei_head, "fileDesc")
        # write the title
        title_stmt = etree.SubElement(file_desc, "titleStmt")
        title = etree.SubElement(title_stmt, "title")
        if self.title is not None:
            title.text = self.title
        else:
            title.text = self.part.id if self.part.id is not None else "Untitled"
        music = etree.SubElement(mei, "music")
        body = etree.SubElement(music, "body")
        mdiv = etree.SubElement(body, "mdiv")
        score = etree.SubElement(mdiv, "score")
        score.set(XMLNS_ID, "score-" + self.elc_id())
        score_def = etree.SubElement(score, "scoreDef")
        score_def.set(XMLNS_ID, "scoredef-" + self.elc_id())
        staff_grp = etree.SubElement(score_def, "staffGrp")
        staff_grp.set(XMLNS_ID, "staffgrp-" + self.elc_id())
        staff_grp.set("bar.thru", "true")
        self._handle_staffs(staff_grp)
        section = etree.SubElement(score, "section")
        section.set(XMLNS_ID, "section-" + self.elc_id())
        # Iterate over part's timeline
        for measure in self.part.measures:
            # Create measure element
            xml_el = etree.SubElement(section, "measure")
            self._handle_measure(measure, xml_el)

        return mei

    def _handle_staffs(self, xml_el):
        clefs = self.part.iter_all(spt.Clef, start=0, end=1)
        clefs = {c.staff: c for c in clefs}
        key_sigs = list(self.part.iter_all(spt.KeySignature, start=0, end=1))
        keys_sig = key_sigs[0] if len(key_sigs) > 0 else None
        time_sigs = list(self.part.iter_all(spt.TimeSignature, start=0, end=1))
        time_sig = time_sigs[0] if len(time_sigs) > 0 else None
        for staff_num in range(self.part.number_of_staves):
            staff_num += 1
            staff_def = etree.SubElement(xml_el, "staffDef")
            staff_def.set("n", str(staff_num))
            staff_def.set(XMLNS_ID, "staffdef-" + self.elc_id())
            staff_def.set("lines", "5")
            # Get clef for this staff If no cleff is available for this staff, default to "G2"
            clef_def = etree.SubElement(staff_def, "clef")
            clef_def.set(XMLNS_ID, "clef-" + self.elc_id())
            clef_shape = clefs[staff_num].sign if staff_num in clefs.keys() else "G"
            clef_def.set("shape", str(clef_shape))
            (
                clef_def.set("line", str(clefs[staff_num].line))
                if staff_num in clefs.keys()
                else clef_def.set("line", "2")
            )
            # Get key signature for this staff
            if keys_sig is not None:
                ks_def = etree.SubElement(staff_def, "keySig")
                ks_def.set(XMLNS_ID, "keysig-" + self.elc_id())
                (
                    ks_def.set("mode", keys_sig.mode)
                    if keys_sig.mode is not None
                    else ks_def.set("mode", "major")
                )
                if keys_sig.fifths == 0:
                    ks_def.set("sig", "0")
                elif keys_sig.fifths > 0:
                    ks_def.set("sig", str(keys_sig.fifths) + "s")
                    self.current_key_signature = self.sharps[: keys_sig.fifths]
                else:
                    ks_def.set("sig", str(abs(keys_sig.fifths)) + "f")
                    self.current_key_signature = self.flats[: abs(keys_sig.fifths)]
                # Find the pname from the number of sharps or flats and the mode
                ks_def.set(
                    "pname",
                    fifths_mode_to_key_name(keys_sig.fifths, keys_sig.mode).lower()[
                        0
                    ],  # only the first letter
                )

            if time_sig is not None:
                ts_def = etree.SubElement(staff_def, "meterSig")
                ts_def.set(XMLNS_ID, "msig-" + self.elc_id())
                ts_def.set("count", str(time_sig.beats))
                ts_def.set("unit", str(time_sig.beat_type))

    def _handle_measure(self, measure, measure_el):
        # Add measure number
        measure_el.set("n", str(measure.number))
        measure_el.set(XMLNS_ID, "measure-" + self.elc_id())
        note_or_rest_elements = np.array(
            list(
                self.part.iter_all(
                    spt.GenericNote,
                    start=measure.start.t,
                    end=measure.end.t,
                    include_subclasses=True,
                )
            )
        )
        # Separate by staff
        staffs = np.vectorize(lambda x: x.staff)(note_or_rest_elements)
        voices = np.vectorize(lambda x: x.voice)(note_or_rest_elements)
        unique_staffs, staff_inverse_map = np.unique(staffs, return_inverse=True)
        unique_voices_par = np.unique(voices)
        voice_staff_map = {
            v: {
                "mask": voices == v,
                "staff": np.bincount(
                    staffs[voices == v], minlength=self.num_staves
                ).argmax(),
            }
            for v in unique_voices_par
        }
        for i in range(self.num_staves):
            staff = i + 1
            staff_el = etree.SubElement(measure_el, "staff")
            # Add staff number
            staff_el.set("n", str(staff))
            staff_el.set(XMLNS_ID, "staff-" + self.elc_id())
            if staff not in unique_staffs:
                continue
            staff_notes = note_or_rest_elements[staff_inverse_map == i]
            # Separate by voice
            voices = np.vectorize(lambda x: x.voice)(staff_notes)
            unique_voices, voice_inverse_map = np.unique(voices, return_inverse=True)
            for j, voice in enumerate(unique_voices):
                voice_el = etree.SubElement(staff_el, "layer")
                voice_el.set("n", str(voice))
                voice_el.set(XMLNS_ID, "voice-" + self.elc_id())
                # try to handle cross-staff beaming
                if voice_staff_map[voice]["staff"] != staff:
                    continue
                voice_notes = note_or_rest_elements[voice_staff_map[voice]["mask"]]
                # Sort by onset
                note_start_times = np.vectorize(lambda x: x.start.t)(voice_notes)
                unique_onsets = np.unique(note_start_times)
                for onset in unique_onsets:
                    # group by start time
                    notes = voice_notes[note_start_times == onset]
                    if len(notes) > 1:
                        self._handle_chord(notes, voice_el)
                    else:
                        self._handle_note_or_rest(notes[0], voice_el)

        self._handle_tuplets(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_beams(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_clef_changes(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_ks_changes(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_ts_changes(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_harmony(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_fermata(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_barline(measure_el, start=measure.start.t, end=measure.end.t)
        self._handle_fingering(measure_el, start=measure.start.t, end=measure.end.t)
        return measure_el

    def _handle_chord(self, chord, xml_voice_el):
        chord_el = etree.SubElement(xml_voice_el, "chord")
        chord_el.set(XMLNS_ID, "chord-" + self.elc_id())
        for note in chord:
            duration = self._handle_note_or_rest(note, chord_el)
        chord_el.set("dur", duration)

    def _handle_note_or_rest(self, note, xml_voice_el):
        if isinstance(note, spt.Rest):
            duration = self._handle_rest(note, xml_voice_el)
        else:
            duration = self._handle_note(note, xml_voice_el)
        return duration

    def _handle_rest(self, rest, xml_voice_el):
        rest_el = etree.SubElement(xml_voice_el, "rest")
        if "type" not in rest.symbolic_duration.keys():
            rest.symbolic_duration = estimate_symbolic_duration(
                rest.end.t - rest.start.t, div=self.qdivs
            )
        duration = SYMBOLIC_TYPES_TO_MEI_DURS[rest.symbolic_duration["type"]]
        rest_el.set("dur", duration)
        if "dots" in rest.symbolic_duration:
            rest_el.set("dots", str(rest.symbolic_duration["dots"]))
        if rest.id is None:
            rest.id = "rest-" + self.elc_id()
        rest_el.set(XMLNS_ID, rest.id)
        return duration

    def _handle_note(self, note, xml_voice_el):
        note_el = etree.SubElement(xml_voice_el, "note")
        duration = SYMBOLIC_TYPES_TO_MEI_DURS[note.symbolic_duration["type"]]
        note_el.set("dur", duration)
        (
            note_el.set(XMLNS_ID, "note-" + self.elc_id())
            if note.id is None
            else note_el.set(XMLNS_ID, note.id)
        )
        if "dots" in note.symbolic_duration:
            note_el.set("dots", str(note.symbolic_duration["dots"]))
        note_el.set("oct", str(note.octave))
        note_el.set("pname", note.step.lower())
        note_el.set("staff", str(note.staff))
        if note.tie_next is not None and note.tie_prev is not None:
            note_el.set("tie", "m")
        elif note.tie_next is not None:
            note_el.set("tie", "i")
        elif note.tie_prev is not None:
            note_el.set("tie", "t")

        if note.stem_direction in ["up", "down"]:
            note_el.set("stem.dir", note.stem_direction)

        if note.alter is not None:
            if (
                note.step.lower() + ALTER_TO_MEI[note.alter]
                in self.current_key_signature
            ):
                note_el.set("accid.ges", ALTER_TO_MEI[note.alter])
            else:
                accidental = etree.SubElement(note_el, "accid")
                accidental.set(XMLNS_ID, "accid-" + self.elc_id())
                accidental.set("accid", ALTER_TO_MEI[note.alter])

        if isinstance(note, spt.GraceNote):
            note_el.set("grace", "acc")
        return duration

    def _handle_tuplets(self, measure_el: lxml.etree._Element, start: int, end: int):
        for tuplet in self.part.iter_all(spt.Tuplet, start=start, end=end):
            start_note = tuplet.start_note
            end_note = tuplet.end_note
            if start_note.start.t < start or end_note.end.t > end:
                warnings.warn(
                    "Tuplet start or end note is outside of the measure. Skipping tuplet element."
                )
                continue
            if start_note.start.t > end_note.start.t:
                warnings.warn(
                    "Tuplet start note is after end note. Skipping tuplet element."
                )
                continue
            # Skip if start and end notes are in different voices or staves
            if start_note.voice != end_note.voice or start_note.staff != end_note.staff:
                warnings.warn(
                    "Tuplet start and end notes are in different voices or staves. Skipping tuplet element."
                )
                continue
            # Find the note element corresponding to the start note i.e. has the same id value
            start_note_el = measure_el.xpath(f".//*[@xml:id='{start_note.id}']")[0]
            # Find the note element corresponding to the end note i.e. has the same id value
            end_note_el = measure_el.xpath(f".//*[@xml:id='{end_note.id}']")[0]
            # if start or note element parents are chords, tuplet element should be added as parent of the chord element
            start_note_el = (
                start_note_el.getparent()
                if start_note_el.getparent().tag == "chord"
                else start_note_el
            )
            end_note_el = (
                end_note_el.getparent()
                if end_note_el.getparent().tag == "chord"
                else end_note_el
            )
            # Create the tuplet element as parent of the start and end note elements
            # Make it start at the same index as the start note element
            tuplet_el = etree.Element("tuplet")
            layer_el = start_note_el.getparent()
            layer_el.insert(layer_el.index(start_note_el), tuplet_el)
            tuplet_el.set(XMLNS_ID, "tuplet-" + self.elc_id())
            tuplet_el.set("num", str(start_note.symbolic_duration["actual_notes"]))
            tuplet_el.set("numbase", str(start_note.symbolic_duration["normal_notes"]))
            # Add all elements between the start and end note elements to the tuplet element as childen
            # Find them from the xml tree
            start_note_index = start_note_el.getparent().index(start_note_el)
            end_note_index = end_note_el.getparent().index(end_note_el)
            # If the start and end note elements are not in order skip (it a weird bug that happens sometimes)
            if start_note_index > end_note_index:
                continue
            xml_el_within_tuplet = [
                start_note_el.getparent()[i]
                for i in range(start_note_index, end_note_index + 1)
            ]
            for el in xml_el_within_tuplet:
                tuplet_el.append(el)

    def _handle_beams(self, measure_el: lxml.etree._Element, start: int, end: int):
        for beam in self.part.iter_all(spt.Beam, start=start, end=end):
            # If the beam has only one note, skip it
            if len(beam.notes) < 2:
                continue
            start_note = beam.notes[np.argmin([n.start.t for n in beam.notes])]
            # Beam element is parent of the note element
            note_el = measure_el.xpath(f".//*[@xml:id='{start_note.id}']")[0]
            layer_el = note_el.getparent()
            insert_index = layer_el.index(note_el)
            # If the parent is a tuplet, the beam element should be added as parent of the tuplet element
            if layer_el.tag == "tuplet":
                parent_el = layer_el.getparent()
                insert_index = parent_el.index(layer_el)
                layer_el = parent_el
            # If the parent is a chord, the beam element should be added as parent of the chord element
            if layer_el.tag == "chord":
                parent_el = layer_el.getparent()
                if parent_el.tag == "tuplet":
                    parent_el = parent_el.getparent()
                    insert_index = parent_el.index(layer_el.getparent())
                    layer_el = parent_el
                else:
                    insert_index = parent_el.index(layer_el)
                    layer_el = parent_el

            # Create the beam element
            beam_el = etree.Element("beam")
            layer_el.insert(insert_index, beam_el)
            beam_el.set(XMLNS_ID, "beam-" + self.elc_id())
            for note in beam.notes:
                # Find the note element corresponding to the start note i.e. has the same id value
                note_el = measure_el.xpath(f".//*[@xml:id='{note.id}']")
                if len(note_el) > 0:
                    note_el = note_el[0]
                    # Add the note element to the beam element but if the parent is a tuplet, the note element should be added as child of the tuplet element
                    if note_el.getparent().tag == "tuplet":
                        beam_el.append(note_el.getparent())
                    elif note_el.getparent().tag == "chord":
                        if note_el.getparent().getparent().tag == "tuplet":
                            beam_el.append(note_el.getparent().getparent())
                        else:
                            beam_el.append(note_el.getparent())
                    else:
                        # verify that the note element is not already a child of the beam element
                        if note_el.getparent() != beam_el:
                            beam_el.append(note_el)

    def _handle_clef_changes(
        self, measure_el: lxml.etree._Element, start: int, end: int
    ):
        for clef in self.part.iter_all(spt.Clef, start=start, end=end):
            # Clef element is parent of the note element
            if clef.start.t == 0:
                continue
            # Find the note element corresponding to the start note i.e. has the same id value
            for note in self.part.iter_all(
                spt.GenericNote, start=clef.start.t, end=clef.start.t
            ):
                note_el = measure_el.xpath(f".//*[@xml:id='{note.id}']")
                if len(note_el) > 0:
                    note_el = note_el[0]
                    layer_el = note_el.getparent()
                    insert_index = layer_el.index(note_el)
                    # Create the clef element
                    clef_el = etree.Element("clef")
                    layer_el.insert(insert_index, clef_el)
                    clef_el.set(XMLNS_ID, "clef-" + self.elc_id())
                    clef_el.set("shape", str(clef.sign))
                    clef_el.set("line", str(clef.line))

    def _handle_ks_changes(self, measure_el: lxml.etree._Element, start: int, end: int):
        # For key signature changes, we add a new scoreDef element at the beginning of the measure
        # and add the key signature element as attributes of the scoreDef element
        for key_sig in self.part.iter_all(spt.KeySignature, start=start, end=end):
            if key_sig.start.t == 0:
                continue
            # Create the scoreDef element
            score_def_el = etree.Element("scoreDef")
            score_def_el.set(XMLNS_ID, "scoredef-" + self.elc_id())
            (
                score_def_el.set("mode", key_sig.mode)
                if key_sig.mode is not None
                else score_def_el.set("mode", "major")
            )
            if key_sig.fifths == 0:
                score_def_el.set("sig", "0")
            elif key_sig.fifths > 0:
                score_def_el.set("sig", str(key_sig.fifths) + "s")
                self.current_key_signature = self.sharps[: key_sig.fifths]
            else:
                score_def_el.set("sig", str(abs(key_sig.fifths)) + "f")
                self.current_key_signature = self.flats[: abs(key_sig.fifths)]
            # Find the pname from the number of sharps or flats and the mode
            score_def_el.set(
                "pname", fifths_mode_to_key_name(key_sig.fifths, key_sig.mode).lower()
            )
            # Add the scoreDef element at before the measure element starts
            parent = measure_el.getparent()
            parent.insert(parent.index(measure_el), score_def_el)

    def _handle_ts_changes(self, measure_el: lxml.etree._Element, start: int, end: int):
        # For key signature changes, we add a new scoreDef element at the beginning of the measure
        # and add the key signature element as attributes of the scoreDef element
        for time_sig in self.part.iter_all(spt.TimeSignature, start=start, end=end):
            if time_sig.start.t == 0:
                continue
            # Create the scoreDef element
            score_def_el = etree.Element("scoreDef")
            score_def_el.set(XMLNS_ID, "scoredef-" + self.elc_id())

            # Add the scoreDef element at before the measure element starts
            parent = measure_el.getparent()
            parent.insert(parent.index(measure_el), score_def_el)
            score_def_el.set("count", str(time_sig.beats))
            score_def_el.set("unit", str(time_sig.beat_type))

    def _handle_harmony(self, measure_el: lxml.etree._Element, start: int, end: int):
        """
        For harmonies we add a new harm element at the beginning of the measure.
        The position doesn't really matter since the tstamp attribute will place it correctly
        The harmonies will be displayed below the lowest staff.
        """
        for harmony in self.part.iter_all(spt.RomanNumeral, start=start, end=end):
            harm_el = etree.SubElement(measure_el, "harm")
            harm_el.set(XMLNS_ID, "harm-" + self.elc_id())
            harm_el.set("staff", str(self.part.number_of_staves))
            harm_el.set(
                "tstamp",
                str(np.diff(self.part.quarter_map([start, harmony.start.t]))[0] + 1),
            )
            harm_el.set("place", "below")
            # text is a child element of harmony but not a xml element
            harm_el.text = harmony.text

        for harmony in self.part.iter_all(spt.Cadence, start=start, end=end):
            # if there is already a harmony at the same position, add the cadence to the text of the harmony
            harm_els = measure_el.xpath(
                f".//harm[@tstamp='{np.diff(self.part.quarter_map([start, harmony.start.t]))[0] + 1}']"
            )
            if len(harm_els) > 0:
                harm_el = harm_els[0]
                harm_el.text += " |" + harmony.text
            else:

                harm_el = etree.SubElement(measure_el, "harm")
                harm_el.set(XMLNS_ID, "harm-" + self.elc_id())
                harm_el.set("staff", str(self.part.number_of_staves))
                harm_el.set(
                    "tstamp",
                    str(
                        np.diff(self.part.quarter_map([start, harmony.start.t]))[0] + 1
                    ),
                )
                harm_el.set("place", "below")
                # text is a child element of harmony but not a xml element
                harm_el.text = "|" + harmony.text

    def _handle_fermata(self, measure_el: lxml.etree._Element, start: int, end: int):
        for fermata in self.part.iter_all(spt.Fermata, start=start, end=end):
            if fermata.ref is not None:
                note = fermata.ref
                note_el = measure_el.xpath(f".//*[@xml:id='{note.id}']")
                if len(note_el) > 0:
                    note_el[0].set("fermata", "above")
            else:
                fermata_el = etree.SubElement(measure_el, "fermata")
                fermata_el.set(XMLNS_ID, "fermata-" + self.elc_id())
                fermata_el.set(
                    "tstamp",
                    str(
                        np.diff(self.part.quarter_map([start, fermata.start.t]))[0] + 1
                    ),
                )
                # Set the fermata to be above the staff (the highest staff)
                fermata_el.set("staff", "1")

    def _handle_barline(self, measure_el: lxml.etree._Element, start: int, end: int):
        for end_barline in self.part.iter_all(
            spt.Ending, start=end, end=end + 1, mode="ending"
        ):
            measure_el.set("right", "end")
        for end_barline in self.part.iter_all(
            spt.Barline, start=end, end=end + 1, mode="starting"
        ):
            if end_barline.style == "light-heavy":
                measure_el.set("right", "end")
        for end_repeat in self.part.iter_all(
            spt.Repeat, start=end, end=end + 1, mode="ending"
        ):
            measure_el.set("right", "rptend")
        for start_repeat in self.part.iter_all(
            spt.Repeat, start=start, end=start + 1, mode="starting"
        ):
            measure_el.set("left", "rptstart")

    def _handle_fingering(self, measure_el: lxml.etree._Element, start: int, end: int):
        """
        For fingering we add a new fing element at the end of the measure.
        The position doesn't really matter since the startid attribute will place it correctly
        """
        for note in self.part.iter_all(spt.Note, start=start, end=end):
            if note.technical is not None:
                for technical_notation in note.technical:
                    if (
                        isinstance(technical_notation, score.Fingering)
                        and note.id is not None
                    ):
                        fing_el = etree.SubElement(measure_el, "fing")
                        fing_el.set(XMLNS_ID, "fing-" + self.elc_id())
                        fing_el.set("startid", note.id)
                        # Naive way to place the fingering notation
                        fing_el.set("place", ("above" if note.staff == 1 else "below"))
                        # text is a child element of fingering but not a xml element
                        fing_el.text = technical_notation.fingering


@deprecated_alias(parts="score_data")
def save_mei(
    score_data: spt.ScoreLike,
    out: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Save a one or more Part or PartGroup instances in MEI format.

    Parameters
    ----------
    score_data : Score, list, Part, or PartGroup
        The musical score to be saved. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    out: str, file-like object, or None, optional
        Output file

    Returns
    -------
    None or str
        If no output file is specified using `out` the function returns the
        MEI data as a string. Otherwise the function returns None.
    """

    if isinstance(score_data, spt.Score):
        parts = score_data.parts
    elif isinstance(score_data, list):
        parts = score_data
    else:
        parts = [score_data]

    if len(parts) > 1:
        raise ValueError("Partitura supports only one part or PartGroup per MEI file.")

    score_data = parts[0]

    exporter = MEIExporter(score_data, title=title)
    root = exporter.export_to_mei()

    if out:
        if hasattr(out, "write"):
            out.write(
                etree.tostring(
                    root.getroottree(),
                    encoding="UTF-8",
                    xml_declaration=True,
                    pretty_print=True,
                    doctype=DOCTYPE,
                )
            )

        else:
            with open(out, "wb") as f:
                f.write(
                    etree.tostring(
                        root.getroottree(),
                        encoding="UTF-8",
                        xml_declaration=True,
                        pretty_print=True,
                        doctype=DOCTYPE,
                    )
                )

    else:
        return etree.tostring(
            root.getroottree(),
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=DOCTYPE,
        )
