#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting Kern files.
"""
import math
from collections import defaultdict

import numpy

import partitura.score as spt
from operator import itemgetter
from typing import Optional
import numpy as np
import warnings
from partitura.utils import partition, iter_current_next, to_quarter_tempo
from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_kern"]


ACC_TO_SIGN = {
    0: "n",
    -1: "-",
    1: "#",
    -2: "--",
    2: "##",
}

# Kern notes encoding has a dedicated octave for each note.
KERN_NOTES = {
    ("C", 3): "C",
    ("D", 3): "D",
    ("E", 3): "E",
    ("F", 3): "F",
    ("G", 3): "G",
    ("A", 3): "A",
    ("B", 3): "B",
    ("C", 4): "c",
    ("D", 4): "d",
    ("E", 4): "e",
    ("F", 4): "f",
    ("G", 4): "g",
    ("A", 4): "a",
    ("B", 4): "b",
}

KERN_DURS = {
    "maxima": "000",
    "long": "00",
    "breve": "0",
    "whole": "1",
    "half": "2",
    "quarter": "4",
    "eighth": "8",
    "16th": "16",
    "32nd": "32",
    "64th": "64",
    "128th": "128",
    "256th": "256",
}

KEYS = ["f", "c", "g", "d", "a", "e", "b"]


class KernExporter(object):
    """
    Class for exporting a partitura score to Kern format.

    Parameters
    ----------
    part: spt.Part
        Part to export to Kern format.
    """

    def __init__(self, part):
        self.part = part
        note_array = part.note_array(include_staff=True)
        num_measures = len(part.measures)
        num_notes = len(part.notes)
        num_rests = len(part.rests)
        self.unique_voc_staff = np.unique(note_array[["voice", "staff"]], axis=0)
        self.vocstaff_map_dict = {
            f"{self.unique_voc_staff[i][0]}-{self.unique_voc_staff[i][1]}": i
            for i in range(self.unique_voc_staff.shape[0])
        }
        # Part elements is really the maximum number of lines we could have in the kern file
        # we add some to account for the **kern and the *- encoding at beginning and end of file and also tandem elements
        # that might be added. We also add the number of measures to account for the measure encoding
        total_elements_ish = num_measures + num_notes + num_rests + 2 + 10
        self.out_data = np.empty(
            (total_elements_ish, len(self.unique_voc_staff)), dtype=object
        )
        self.unique_times = np.array([p.t for p in part._points])
        # Fill all values with the "." character to filter afterwards
        self.out_data.fill(".")
        self.out_data[0] = "**kern"
        self.out_data[-1] = "*-"
        # Add the staff element to the second line
        for i in range(self.unique_voc_staff.shape[0]):
            self.out_data[1, i] = f"*staff{self.unique_voc_staff[i][1]}"
        self.prev_note_time = None
        self.prev_note_col_idx = None
        self.prev_note_row_idx = None

    def parse(self):
        """
        Parse the partitura score to Kern format.

        This method iterates over all elements in the partitura score and converts them to Kern format.
        To better process the elements, the method first groups them by start time and then processes them in order.
        It first finds notes and then processes structural elements (clefs, time signatures, etc.) and finally measures.

        Returns
        -------
        self.out_data: np.ndarray
            Kern file as a numpy array of strings.
        """
        row_idx = 2
        for start_time in self.unique_times:
            end_time = start_time + 1
            # Get all elements starting at this time
            elements_starting = np.array(
                list(self.part.iter_all(start=start_time, end=end_time)), dtype=object
            )
            # Find notes
            note_mask = np.array(
                [isinstance(el, spt.GenericNote) for el in elements_starting]
            )
            if np.any(~note_mask):
                bar_mask = np.array(
                    [
                        isinstance(el, spt.Measure)
                        for el in elements_starting[~note_mask]
                    ]
                )
                tandem_mask = ~bar_mask
                structural_elements = elements_starting[~note_mask]
                structural_elements = np.hstack(
                    (structural_elements[tandem_mask], structural_elements[bar_mask])
                )
            else:
                structural_elements = elements_starting[~note_mask]
            # Put structural elements first (start with tandem elements, then measure elements, then notes and rests)
            elements_starting = np.hstack(
                (structural_elements, elements_starting[note_mask])
            )
            for el in elements_starting:
                add_row = True
                if isinstance(el, spt.GenericNote):
                    self._handle_note(el, row_idx)
                elif isinstance(el, spt.Clef):
                    # Apply clef to all voices of the same staff
                    currect_staff = el.staff
                    for staff_idx in range(self.unique_voc_staff.shape[0]):
                        if self.unique_voc_staff[staff_idx][1] == currect_staff:
                            kern_el = f"*clef{el.sign.upper()}{el.line}"
                            self.out_data[row_idx, staff_idx] = kern_el
                elif isinstance(el, spt.Tempo):
                    # Apply tempo to all splines
                    kern_el = f"*MM{to_quarter_tempo(el.qpm)}"
                    self.out_data[row_idx] = kern_el
                elif isinstance(el, spt.Measure):
                    # Apply measure to all splines
                    kern_el = f"={el.number}"
                    self.out_data[row_idx] = kern_el
                elif isinstance(el, spt.TimeSignature):
                    # Apply element to all splines
                    kern_el = f"*M{el.beats}/{el.beat_type}"
                    self.out_data[row_idx] = kern_el
                elif isinstance(el, spt.KeySignature):
                    # Apply element to all splines
                    if el.fifths < 0:
                        alters = "-".join(KEYS[: el.fifths])
                    elif el.fifths > 0:
                        alters = "#".join(KEYS[: el.fifths])
                    else:
                        alters = ""
                    kern_el = f"*k[{alters}]"
                    self.out_data[row_idx] = kern_el
                else:
                    add_row = False
                    warnings.warn(f"Element {el} is not supported for kern export yet.")
                if add_row:
                    row_idx += 1
        return self.out_data

    def trim(self, data):
        # if an entire row is filled with "." elements remove it.
        out_data = data[~np.all(data == ".", axis=1)]
        return out_data

    def sym_dur_to_kern(self, symbolic_duration: dict) -> str:
        kern_base = KERN_DURS[symbolic_duration["type"]]
        dots = (
            "." * symbolic_duration["dots"]
            if "dots" in symbolic_duration.keys()
            else ""
        )
        if "actual_notes" in symbolic_duration.keys() and "normal_notes":
            kern_base = (
                int(kern_base)
                * symbolic_duration["actual_notes"]
                / symbolic_duration["normal_notes"]
            )
            kern_base = str(kern_base)
        return kern_base + dots

    def duration_to_kern(self, element: spt.GenericNote) -> str:
        if isinstance(element, spt.GraceNote):
            if element.grace_type == "acciaccatura":
                return "p"
            else:
                return "q"
        else:
            if "type" not in element.symbolic_duration.keys():
                warnings.warn(f"Element {element} has no symbolic duration type")
                return "4"
            return self.sym_dur_to_kern(element.symbolic_duration)

    def pitch_to_kern(self, element: spt.GenericNote) -> str:
        """
        Transform a Partitura Note object to a kern note string (only pitch).

        To encode pitch correctly in kern we need to take into account that the octave
        duplication of the step in kern can either move the note up or down an octave

        """
        if isinstance(element, spt.Rest):
            return "r"
        step, alter, octave = element.step, element.alter, element.octave
        # Check if we need to have duplication of the step character
        if octave > 4:
            multiply_character = octave - 3
            octave = 4
        elif octave < 3:
            multiply_character = 4 - octave
            octave = 3
        else:
            multiply_character = 1
        # Fetch the correct string for the step and multiply it if needed
        kern_step = KERN_NOTES[(step, octave)] * multiply_character
        kern_alter = ACC_TO_SIGN[alter] if alter is not None else ""
        return kern_step + kern_alter

    def markings_to_kern(self, element: spt.GenericNote) -> str:
        symbols = ""
        if not isinstance(element, spt.Rest):
            if element.tie_next and element.tie_prev:
                symbols += "-"
            elif element.tie_next:
                symbols += "["
            elif element.tie_prev:
                symbols += "]"
            if element.slur_starts:
                symbols += "("
            if element.slur_stops:
                symbols += ")"
        if isinstance(element, spt.Note):
            if element.beam is not None:
                symbols += (
                    "L"
                    if element.beam == "begin"
                    else "J" if element.beam == "end" else "K"
                )
        return symbols

    def _handle_note(self, el: spt.GenericNote, row_idx) -> str:
        voice = el.voice
        staff = el.staff
        duration = self.duration_to_kern(el)
        pitch = self.pitch_to_kern(el)
        col_idx = self.vocstaff_map_dict[f"{voice}-{staff}"]
        markings = self.markings_to_kern(el)
        kern_el = duration + pitch + markings
        if self.prev_note_time == el.start.t:
            if self.prev_note_col_idx == col_idx:
                # Chords in Kern
                self.out_data[self.prev_note_row_idx, self.prev_note_col_idx] = (
                    self.out_data[self.prev_note_row_idx, self.prev_note_col_idx]
                    + " "
                    + kern_el
                )
            else:
                # Same row (start.t) other spline
                self.out_data[self.prev_note_row_idx, col_idx] = kern_el
        else:
            # New line
            self.out_data[row_idx, col_idx] = kern_el
            self.prev_note_row_idx = row_idx
        self.prev_note_col_idx = col_idx
        self.prev_note_time = el.start.t


def save_kern(
    score_data: spt.ScoreLike,
    out: Optional[PathLike] = None,
) -> Optional[np.ndarray]:
    """
    Save a score in Kern format.

    Parameters
    ----------
    score_data: spt.ScoreLike
        Score to save in Kern format

    out: Optional[PathLike]
        Path to save the Kern file. If None, the function returns the Kern file as a numpy array.

    Returns
    -------
    Optional[np.ndarray]
        If out is None, the Kern file is returned as a numpy array.
    """
    # Header extracts meta information about the score
    header = "Here is some random piece"
    # Kern can output only from part so first let's merge parts (we need a timewise representation)
    if isinstance(score_data, spt.Score):
        # TODO check that divisions are the same
        part = spt.merge_parts(score_data.parts)
    else:
        part = score_data
    if not part.measures:
        spt.add_measures(part)
    spt.fill_rests(part, measurewise=False)
    exporter = KernExporter(part)
    out_data = exporter.parse()
    out_data = exporter.trim(out_data)
    # Use numpy savetxt to save the file
    footer = "Encoded using the Partitura Python package, version 1.6.0"
    if out is not None:
        np.savetxt(
            fname=out,
            X=out_data,
            fmt="%1.26s",
            delimiter="\t",
            newline="\n",
            header=header,
            footer=footer,
            comments="!!!",
            encoding="utf-8",
        )
    else:
        return out_data
