#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing Humdrum Kern files.
"""
import copy
import re, sys
import warnings
from typing import Union, Optional
import numpy as np
from math import inf, ceil
import partitura.score as spt
from partitura.utils import PathLike, get_document_name, symbolic_to_numeric_duration


SIGN_TO_ACC = {
    "nn": 0,
    "n": 0,
    "#": 1,
    "s": 1,
    "ss": 2,
    "x": 2,
    "n#": 1,
    "#n": 1,
    "##": 2,
    "###": 3,
    "b": -1,
    "f": -1,
    "bb": -2,
    "ff": -2,
    "bbb": -3,
    "-": -1,
    "n-": -1,
    "-n": -1,
    "--": -2,
}

KERN_NOTES = {
    "C": ("C", 3),
    "D": ("D", 3),
    "E": ("E", 3),
    "F": ("F", 3),
    "G": ("G", 3),
    "A": ("A", 3),
    "B": ("B", 3),
    "c": ("C", 4),
    "d": ("D", 4),
    "e": ("E", 4),
    "f": ("F", 4),
    "g": ("G", 4),
    "a": ("A", 4),
    "b": ("B", 4),
}

KERN_DURS = {
    "000": {"type": "maxima"},
    "00": {"type": "long"},
    "0": {"type": "breve"},
    "1": {"type": "whole"},
    "2": {"type": "half"},
    "4": {"type": "quarter"},
    "8": {"type": "eighth"},
    "16": {"type": "16th"},
    "32": {"type": "32nd"},
    "64": {"type": "64th"},
    "128": {"type": "128th"},
    "256": {"type": "256th"},
}


class KernElement(object):
    def __init__(self, element):
        self.editorial_start = True if "ossia" in element else False
        self.editorial_end = True if "Xstrophe" in element else False
        self.voice_end = True if "*v" in element else False
        self.voice_start = True if "*^" in element else False
        self.element = element.replace("*", "")


def add_durations(a, b):
    return a * b / (a + b)


def dot_function(duration: int, dots: int):
    if dots == 0:
        return duration
    elif duration == 0:
        return 0
    else:
        return add_durations((2**dots) * duration, dot_function(duration, dots - 1))


def parse_by_voice(file: list, dtype=np.object_):
    indices_to_remove = []
    voices = 1
    for i, line in enumerate(file):
        for v in range(voices):
            indices_to_remove.append([i, v])
        if any([line[v] == "*^" for v in range(voices)]):
            voices += 1
        elif sum([(line[v] == "*v") for v in range(voices)]):
            sum_vred = sum([line[v] == "*v" for v in range(voices)]) // 2
            voices = voices - sum_vred

    voice_indices = np.array(indices_to_remove)
    num_voices = voice_indices[:, 1].max() + 1
    data = np.empty((len(file), num_voices), dtype=dtype)
    for line, voice in voice_indices:
        data[line, voice] = file[line][voice]
    data = data.T
    if num_voices > 1:
        # Copy global lines from the first voice to all other voices unless they are the string "*S/ossia"
        cp_idx = np.char.startswith(data[0], "*")
        un_idx = np.char.startswith(data[0], "*S/ossia")
        cp_idx = np.logical_and(cp_idx, ~un_idx)
        for i in range(1, num_voices):
            data[i][cp_idx] = data[0][cp_idx]
        # Copy Measure Lines from the first voice to all other voices
        cp_idx = np.char.startswith(data[0], "=")
        for i in range(1, num_voices):
            data[i][cp_idx] = data[0][cp_idx]
    return data, voice_indices, num_voices


def _handle_kern_with_spine_splitting(kern_path: PathLike):
    """
    Parse a kern file with spine splitting.

    A special case of kern files is when the file contains multiple spines that are split by voice. In this case, this
    function will restructure the data in a way that it can be parsed by the kern parser.

    Parameters
    ----------
    kern_path: str

    Returns
    -------
    data: np.array
        The data to be parsed.
    parsing_idxs: np.array
        The indices of the data that are being parsed indicating the assignment of voices.
    """
    # org_file = np.loadtxt(kern_path, dtype="U", delimiter="\n", comments="!!!", encoding="cp437")
    org_file = np.genfromtxt(
        kern_path, dtype="U", delimiter="\n", comments="!!!", encoding="cp437"
    )
    # Get Main Number of parts and Spline Types
    spline_types = org_file[0].split("\t")
    parsing_idxs = []
    dtype = org_file.dtype
    data = []
    file = org_file.tolist()
    file = [line.split("\t") for line in file if not line.startswith("!")]
    for i in range(len(spline_types)):
        # Parse by voice
        d, voice_indices, num_voices = parse_by_voice(file, dtype=dtype)
        data.append(d)
        parsing_idxs.append([i for _ in range(num_voices)])
        # Remove all parsed cells from the file
        voice_indices = voice_indices[
            np.lexsort((voice_indices[:, 1] * -1, voice_indices[:, 0]))
        ]
        for line, voice in voice_indices:
            if voice < len(file[line]):
                file[line].pop(voice)
            else:
                print(
                    "Line {} does not have a voice {} from original line {}".format(
                        line, voice, org_file[line]
                    )
                )
    data = np.vstack(data).T
    parsing_idxs = np.hstack(parsing_idxs).T
    return data, parsing_idxs


def element_parsing(
    part: spt.Part,
    elements: np.array,
    total_duration_values: np.array,
    same_part: bool,
    doc_lines: np.array,
    line2pos: dict,
):
    """
    Parse and add musical elements to a part.

    Parameters
    ----------
    part : spt.Part
        The partitura part to which elements will be added.
    elements : np.array
        Array of musical elements to be parsed and added.
    total_duration_values : np.array
        Array of total duration values for each element.
    same_part : bool
        Flag indicating if elements are being added to the same part.
    doc_lines : np.array
        Array of document lines corresponding to the elements.
    line2pos : dict
        Dictionary mapping document lines to their part start positions.

    Returns
    -------
    line2pos : dict
        Updated dictionary mapping document lines to their positions.
        This is used to handle cases with split spines.
    """
    divs_pq = part._quarter_durations[0]
    line2pos = line2pos if same_part else {}
    current_tl_pos = 0
    editorial = False
    measure_mapping = {m.number: m.start.t for m in part.iter_all(spt.Measure)}

    for i in range(elements.shape[0]):
        element = elements[i]
        if i < len(doc_lines):
            current_tl_pos = line2pos.get(doc_lines[i], current_tl_pos)

        # Handle editorial elements
        if isinstance(element, KernElement):
            if element.editorial_start:
                editorial = True
            if element.editorial_end:
                editorial = False

        if element is None or editorial:
            continue

        # Handle generic notes
        if isinstance(element, spt.GenericNote):
            if total_duration_values[i] == 0:
                duration_divs = symbolic_to_numeric_duration(
                    element.symbolic_duration, divs_pq
                )
            else:
                quarter_duration = 4 / total_duration_values[i]
                duration_divs = ceil(quarter_duration * divs_pq)
            el_end = current_tl_pos + duration_divs
            part.add(element, start=current_tl_pos, end=el_end)
            line2pos[doc_lines[i]] = current_tl_pos
            current_tl_pos = el_end

        # Handle chords
        elif isinstance(element, tuple):
            quarter_duration = 4 / total_duration_values[i]
            duration_divs = ceil(quarter_duration * divs_pq)
            el_end = current_tl_pos + duration_divs
            for note in element[1]:
                part.add(note, start=current_tl_pos, end=el_end)
            line2pos[doc_lines[i]] = current_tl_pos
            current_tl_pos = el_end

        # Handle slurs
        elif isinstance(element, spt.Slur):
            start_sl = element.start_note.start.t
            end_sl = element.end_note.start.t
            part.add(element, start=start_sl, end=end_sl)

        # Handle other elements
        else:
            # Do not repeat structural elements if they are being added to the same part.
            if not same_part:
                part.add(element, start=current_tl_pos)
                line2pos[doc_lines[i]] = current_tl_pos
            else:
                if isinstance(element, spt.Measure):
                    current_tl_pos = measure_mapping[element.number]

    return line2pos


# functions to initialize the kern parser
def load_kern(
    filename: PathLike,
    force_note_ids: Optional[Union[bool, str]] = None,
    force_same_part: Optional[bool] = False,
) -> spt.Score:
    """
    Parses an KERN file from path to Part.

    Parameters
    ----------
    filename : PathLike
        The path of the KERN document.
    force_note_ids : (None, bool or "keep")
        When True each Note in the returned Part(s) will have a newly assigned unique id attribute.
    Returns
    -------
    score : partitura.score.Score
        The score object containing the parts.
    """
    try:
        # Attempt to load the file using a faster parser that does not support spine splitting
        file = np.loadtxt(
            filename, dtype="U", delimiter="\t", comments="!!", encoding="cp437"
        )
        parsing_idxs = np.arange(file.shape[0])
    except ValueError:
        # Fallback to a slower parser that supports spine splitting
        file, parsing_idxs = _handle_kern_with_spine_splitting(filename)

    partlist = []
    # Get the main number of parts and spline types
    spline_types = file[0]

    # Identify parsable parts that start with "**kern" or "**notes"
    note_parts = np.char.startswith(spline_types, "**kern") | np.char.startswith(
        spline_types, "**notes"
    )
    # Extract splines for the identified parts
    splines = file[1:].T[note_parts]

    has_instrument = np.char.startswith(splines, "*I")
    # Determine if all parts have the same instrument
    p_same_part = (
        np.all(splines[has_instrument] == splines[has_instrument][0], axis=0)
        if np.any(has_instrument)
        else False
    )
    # Determine if all parts have the same *part
    has_part_global = np.char.startswith(splines, "*part")
    p_same_part = (
        np.all(splines[has_part_global] == splines[has_part_global][0], axis=0)
        if np.any(has_part_global)
        else p_same_part
    )
    # Assign all splines to the same part if necessary
    if p_same_part or force_same_part:
        parsing_idxs[:] = 0

    # Initialize lists to store parsed data
    total_durations_list = []
    elements_list = []
    part_assignments = []
    copy_partlist = []
    doc_lines_per_spline = []
    # Initialize staff and voice numbers
    prev_staff = 1
    pvoice = 1

    # Iterate over each spline (musical data stream)
    for j, spline in enumerate(splines):
        # Create a new SplineParser for the current spline
        parser = SplineParser(
            size=spline.shape[-1],
            id="P{}".format(parsing_idxs[j]),
            staff=prev_staff,
            voice=pvoice,
        )
        # Flag to indicate if the part is the same as a previous one
        same_part = parser.id in [p.id for p in copy_partlist]
        if same_part:
            # If the part already exists, add to the previous part
            warnings.warn(
                "Part {} already exists. Adding to previous Part.".format(parser.id)
            )
            part = next(p for p in copy_partlist if p.id == parser.id)
            # Check for staff information in the spline
            has_staff = np.char.startswith(spline, "*staff")
            staff = (
                int(spline[has_staff][0][6:])
                if np.count_nonzero(has_staff)
                else prev_staff
            )
            prev_staff = staff
            if parser.staff != staff:
                parser.staff = staff
            # Update the voice number
            parser.voice = pvoice + 1
        else:
            # If the part does not exist, create a new part
            has_staff = np.char.startswith(spline, "*staff")
            staff = int(spline[has_staff][0][6:]) if np.count_nonzero(has_staff) else 1
            parser.staff = staff
            prev_staff = staff
            parser.voice = 1
            pvoice = 1

        # Parse the spline into musical elements
        elements, lines = parser.parse(spline)
        # Calculate unique durations and ensure they are integers
        unique_durs = np.unique(parser.total_duration_values)
        unique_durs = unique_durs[np.isfinite(unique_durs)]
        d_mul = 2
        while not np.all(np.isclose(unique_durs % 1, 0.0)):
            unique_durs *= d_mul
            d_mul += 1
        unique_durs = unique_durs.astype(int)
        divs_pq = np.lcm.reduce(unique_durs)
        divs_pq = max(divs_pq, 4)

        if same_part:
            divs_pq = np.lcm.reduce([divs_pq, part._quarter_durations[0]])
            part.set_quarter_duration(0, divs_pq)
            pvoice = parser.voice
        else:
            part = spt.Part(
                id=parser.id, quarter_duration=divs_pq, part_name=parser.name
            )

        part_assignments.append(same_part)
        doc_lines_per_spline.append(lines)
        total_durations_list.append(parser.total_duration_values)
        elements_list.append(elements)
        copy_partlist.append(part)

    # Ensure all parts have the same divs per quarter
    divs_pq = np.lcm.reduce([p._quarter_durations[0] for p in copy_partlist])
    for part in copy_partlist:
        part.set_quarter_duration(0, divs_pq)

    line2pos = {}
    for part, elements, total_duration_values, same_part, doc_lines in zip(
        copy_partlist,
        elements_list,
        total_durations_list,
        part_assignments,
        doc_lines_per_spline,
    ):
        line2pos = element_parsing(
            part, elements, total_duration_values, same_part, doc_lines, line2pos
        )

    for i, part in enumerate(copy_partlist):
        if part_assignments[i]:
            continue
        # For all measures add end time as beginning time of next measure
        measures = part.measures
        for i in range(len(measures) - 1):
            measures[i].end = measures[i + 1].start
        measures[-1].end = part.last_point
        # find and add pickup measure
        if part.measures[0].start.t != 0:
            part.add(spt.Measure(number=0), start=0, end=part.measures[0].start.t)

        if parser.id not in [p.id for p in partlist]:
            partlist.append(part)

    spt.assign_note_ids(
        partlist, keep=(force_note_ids is True or force_note_ids == "keep")
    )

    doc_name = get_document_name(filename)
    # Reverse the partlist to correct part order and visualization for exporting musicxml files
    score = spt.Score(partlist=partlist[::-1], id=doc_name)
    return score


class SplineParser(object):
    def __init__(self, id="P1", staff=1, voice=1, size=1, name=""):
        self.id = id
        self.name = name
        self.staff = staff
        self.voice = voice
        self.total_duration_values = []
        self.alters = []
        self.size = size
        self.total_parsed_elements = 0
        self.measure_enum = 1
        self.tie_prev = None
        self.tie_next = None
        self.slurs_start = []
        self.slurs_end = []

    def parse(self, spline: np.array):
        """
        Parse a spline line and return the elements.

        Parameters
        ----------
        spline: np.array
            The spline line to parse. It is a numpy array of strings.

        Returns
        -------
        elements: np.array
            The parsed elements of the spline line.
        """
        lines = np.arange(len(spline))
        # Remove "-" lines
        mask = (
            (spline == "-")
            | (spline == ".")
            | (spline == "")
            | (spline is None)
            | (np.char.startswith(spline, "!") == True)
        )
        mask = ~mask
        spline = spline[mask]
        lines = lines[mask]
        # Empty Numpy array with objects
        elements = np.empty(len(spline), dtype=object)
        self.total_duration_values = np.ones(len(spline))
        # Find Global indices, i.e. where spline cells start with "*" and process
        tandem_mask = np.char.find(spline, "*") != -1
        elements[tandem_mask] = np.vectorize(self.meta_tandem_line, otypes=[object])(
            spline[tandem_mask]
        )
        # Find Barline indices, i.e. where spline cells start with "="
        bar_mask = np.char.find(spline, "=") != -1
        elements[bar_mask] = np.vectorize(self.meta_barline_line, otypes=[object])(
            spline[bar_mask]
        )
        # Find Chord indices, i.e. where spline cells contain " "
        chord_mask = np.char.find(spline, " ") != -1
        chord_mask = np.logical_and(chord_mask, np.logical_and(~tandem_mask, ~bar_mask))
        self.total_parsed_elements = -1
        self.note_duration_values = np.ones(len(spline[chord_mask]))
        chord_num = np.count_nonzero(chord_mask)
        self.tie_next = np.zeros(chord_num, dtype=bool)
        self.tie_prev = np.zeros(chord_num, dtype=bool)
        elements[chord_mask] = np.vectorize(self.meta_chord_line, otypes=[object])(
            spline[chord_mask]
        )
        self.total_duration_values[chord_mask] = self.note_duration_values
        # TODO: figure out slurs for chords

        # All the rest are note indices
        note_mask = np.logical_and(~tandem_mask, np.logical_and(~bar_mask, ~chord_mask))
        self.total_parsed_elements = -1
        self.note_duration_values = np.ones(len(spline[note_mask]))
        note_num = np.count_nonzero(note_mask)
        self.tie_next = np.zeros(note_num, dtype=bool)
        self.tie_prev = np.zeros(note_num, dtype=bool)
        notes = np.vectorize(self.meta_note_line, otypes=[object])(spline[note_mask])
        self.total_duration_values[note_mask] = self.note_duration_values
        # Notes should appear in order within stream so shift tie_next by one to the right
        # and tie next and inversingly tie_prev also
        # Case of note to chord tie or chord to note tie is not handled yet
        for note, to_tie in np.c_[
            notes[self.tie_next], notes[np.roll(self.tie_next, -1)]
        ]:
            to_tie.tie_next = note
            note.tie_prev = to_tie

        elements[note_mask] = notes

        # Find Slur indices, i.e. where spline cells contain "(" or ")"
        open_slur_mask = np.char.find(spline[note_mask], "(") != -1
        close_slur_mask = np.char.find(spline[note_mask], ")") != -1
        self.slurs_start = np.where(open_slur_mask)[0]
        self.slurs_end = np.where(close_slur_mask)[0]
        # Only add slur if there is a start and end
        if len(self.slurs_start) == len(self.slurs_end):
            slurs = np.empty(len(self.slurs_start), dtype=object)
            for i, (start, end) in enumerate(zip(self.slurs_start, self.slurs_end)):
                slurs[i] = spt.Slur(notes[start], notes[end])
            # Add slurs to elements
            elements = np.append(elements, slurs)
        else:
            warnings.warn(
                "Slurs openings and closings do not match. Skipping parsing slurs for this part {}.".format(
                    self.id
                )
            )

        return elements, lines

    def meta_tandem_line(self, line: str):
        """
        Find all tandem lines
        """
        # find number and keep its index.
        self.total_parsed_elements += 1
        if line.startswith("*MM"):
            rest = line[3:]
            return self.process_tempo_line(rest)
        elif line.startswith("*I"):
            rest = line[2:]
            return self.process_istrument_line(rest)
        elif line.startswith("*clef"):
            rest = line[5:]
            return self.process_clef_line(rest)
        elif line.startswith("*M"):
            rest = line[2:]
            return self.process_meter_line(rest)
        elif line.startswith("*k"):
            rest = line[2:]
            return self.process_key_signature_line(rest)
        elif line.startswith("*IC"):
            rest = line[3:]
            return self.process_istrument_class_line(rest)
        elif line.startswith("*IG"):
            rest = line[3:]
            return self.process_istrument_group_line(rest)
        elif line.startswith("*tb"):
            rest = line[3:]
            return self.process_timebase_line(rest)
        elif line.startswith("*ITr"):
            rest = line[4:]
            return self.process_istrument_transpose_line(rest)
        elif line.startswith("*staff"):
            rest = line[6:]
            return self.process_staff_line(rest)
        elif line.endswith(":"):
            rest = line[1:]
            return self.process_key_line(rest)
        elif line.startswith("*-"):
            return self.process_fine()
        else:
            return KernElement(element=line)

    def process_tempo_line(self, line: str):
        return spt.Tempo(float(line))

    def process_fine(self):
        return spt.Fine()

    def process_istrument_line(self, line: str):
        # TODO: add support for instrument lines
        return

    def process_istrument_class_line(self, line: str):
        # TODO: add support for instrument class lines
        return

    def process_istrument_group_line(self, line: str):
        # TODO: add support for instrument group lines
        return

    def process_timebase_line(self, line: str):
        # TODO: add support for timebase lines
        return

    def process_istrument_transpose_line(self, line: str):
        # TODO: add support for instrument transpose lines
        return

    def process_key_line(self, line: str):
        find = re.search(r"([a-gA-G])", line).group(0)
        # check if the key is major or minor by checking if the key is in lower or upper case.
        self.mode = "minor" if find.islower() else "major"
        return

    def process_staff_line(self, line: str):
        self.staff = int(line)
        return spt.Staff(self.staff)

    def process_clef_line(self, line: str):
        # if the cleff line does not contain any of the following characters, ["G", "F", "C"], raise a ValueError.
        if not any(c in line for c in ["G", "F", "C"]):
            raise ValueError("Unrecognized clef: {}".format(line))
        # find the clef
        clef = re.search(r"([GFC])", line).group(0)
        # find the octave
        has_line = re.search(r"([0-9])", line)
        octave_change = "v" in line
        if has_line is None:
            if clef == "G":
                clef_line = 2
            elif clef == "F":
                clef_line = 4
            elif clef == "C":
                clef_line = 3
            elif clef == "X":
                clef = "percussion"
                clef_line = 1
            else:
                raise ValueError("Unrecognized clef line: {}".format(line))
        else:
            clef_line = int(has_line.group(0))
        if octave_change and clef_line == 2 and clef == "G":
            octave = -1
        elif octave_change:
            warnings.warn("Octave change not supported for clef: {}".format(line))
            octave = 0
        else:
            octave = 0

        return spt.Clef(
            sign=clef, staff=self.staff, line=int(clef_line), octave_change=octave
        )

    def process_key_signature_line(self, line: str):
        fifths = line.count("#") - line.count("-")
        alters = re.findall(r"([a-gA-G#\-]+)", line)
        alters = "".join(alters)
        # split alters by two characters
        self.alters = [alters[i : i + 2] for i in range(0, len(alters), 2)]
        # TODO retrieve the key mode
        mode = "major"
        return spt.KeySignature(fifths, mode)

    def process_meter_line(self, line: str):
        if " " in line:
            line = line.split(" ")[0]
        numerator, denominator = line.split("/")
        # Find digits in numerator and denominator and convert to int
        numerator = int(re.search(r"([0-9]+)", numerator).group(0))
        denominator = int(re.search(r"([0-9]+)", denominator).group(0))
        return spt.TimeSignature(numerator, denominator)

    def _process_kern_pitch(self, pitch: str):
        # find accidentals
        alter = re.search(r"([n#-]+)", pitch)
        # remove alter from pitch
        pitch = pitch.replace(alter.group(0), "") if alter else pitch
        step, octave = KERN_NOTES[pitch[0]]
        # do_alt = (step + alter.group(0)).lower() not in self.alters if alter else False
        if octave == 4:
            octave = octave + pitch.count(pitch[0]) - 1
        elif octave == 3:
            octave = octave - pitch.count(pitch[0]) + 1
        alter = SIGN_TO_ACC[alter.group(0)] if alter is not None else None
        return step, octave, alter

    def _process_kern_duration(self, duration: str, is_grace=False):
        """
        Process the duration of a note.

        Parameters
        ----------
        duration: str
            The duration of the note.
        is_grace: bool(default=False)
            If the note is a grace note.
        Returns
        -------
        symbolic_duration: dict
            A dictionary containing the symbolic duration of the note.
        """
        dots = duration.count(".")
        dur = duration.replace(".", "")
        if dur in KERN_DURS.keys():
            symbolic_duration = copy.deepcopy(KERN_DURS[dur])
        # support for extended kern durations
        elif "%" in dur:
            dur = dur.split("%")
            nom, den = int(dur[0]), int(dur[1])
            symbolic_duration = {
                "type": "whole",
                "dots": 0,
                "actual_notes": nom,
                "normal_notes": den,
            }
            dur = nom * den
        else:
            dur = float(dur)
            key_loolup = [2**i for i in range(0, 9)]
            diff = dict(
                (
                    map(
                        lambda x: (dur - x, str(x)) if dur > x else (dur + x, str(x)),
                        key_loolup,
                    )
                )
            )

            symbolic_duration = copy.deepcopy(KERN_DURS[diff[min(list(diff.keys()))]])
            symbolic_duration["actual_notes"] = int(dur // 4)
            symbolic_duration["normal_notes"] = int(diff[min(list(diff.keys()))]) // 4
        if dots:
            symbolic_duration["dots"] = dots
        self.note_duration_values[self.total_parsed_elements] = (
            dot_function((float(dur) if isinstance(dur, str) else dur), dots)
            if not is_grace
            else inf
        )
        return symbolic_duration

    def process_symbol(self, note: spt.Note, symbols: list):
        """
        Process the symbols of a note.

        Parameters
        ----------
        note: spt.Note
            The note to add the symbols to.
        symbols: list
            List of symbols to process.
        """
        # return if list is empty
        if symbols == []:
            return
        if "[" in symbols:
            self.tie_prev[self.total_parsed_elements] = True
            # pop symbol and call again
            symbols.pop(symbols.index("["))
            self.process_symbol(note, symbols)
        if "]" in symbols:
            self.tie_next[self.total_parsed_elements] = True
            symbols.pop(symbols.index("]"))
            self.process_symbol(note, symbols)
        if "_" in symbols:
            # continuing tie
            self.tie_prev[self.total_parsed_elements] = True
            self.tie_next[self.total_parsed_elements] = True
            symbols.pop(symbols.index("_"))
            self.process_symbol(note, symbols)
        return

    def meta_note_line(self, line: str, voice=None, add=True):
        """
        Grammar Defining a note line.

        A note line is specified by the following grammar:
        note_line = symbol | duration | pitch | symbol

        Parameters
        ----------
        line: str
            The line to parse containing a note element.
        voice: int
            The voice of the note.
        add: bool
            If True, the element is added to the number of parsed elements.

        Returns
        -------
        spt.Note object
        """
        self.total_parsed_elements += 1 if add else 0
        voice = self.voice if voice is None else voice
        # extract first occurence of one of the following: a-g A-G r # - n
        find_pitch = re.search(r"([a-gA-Gr\-n#]+)", line)
        if find_pitch is None:
            warnings.warn(
                "No pitch found in line: {}, transforming to a rest".format(line)
            )
            pitch = "r"
        else:
            pitch = find_pitch.group(0)
        # extract duration can be any of the following: 0-9 .
        dur_search = re.search(r"([0-9.%]+)", line)
        # if no duration is found, then the duration is 8 by default (for grace notes with no duration)
        duration = dur_search.group(0) if dur_search else "8"
        # extract symbol can be any of the following: _()[]{}<>|:
        symbols = re.findall(r"([_()\[\]{}<>|:])", line)
        symbolic_duration = self._process_kern_duration(duration, is_grace="q" in line)
        el_id = "{}-s{}-v{}-el{}".format(
            self.id, self.staff, voice, self.total_parsed_elements
        )
        if pitch.startswith("r"):
            return spt.Rest(
                symbolic_duration=symbolic_duration,
                staff=self.staff,
                voice=voice,
                id=el_id,
            )
        step, octave, alter = self._process_kern_pitch(pitch)
        # check if the note is a grace note
        if "q" in line:
            note = spt.GraceNote(
                grace_type="grace",
                step=step,
                octave=octave,
                alter=alter,
                symbolic_duration=symbolic_duration,
                staff=self.staff,
                voice=voice,
                id=el_id,
            )
        else:
            note = spt.Note(
                step,
                octave,
                alter,
                symbolic_duration=symbolic_duration,
                staff=self.staff,
                voice=voice,
                id=el_id,
            )
        if symbols:
            self.process_symbol(note, symbols)
        return note

    def meta_barline_line(self, line: str):
        """
        Grammar Defining a barline line.

        A barline line is specified by the following grammar:
        barline_line = repeat | barline | number | repeat

        Parameters
        ----------
        line: str
            The line to parse containing a barline.

        Returns
        -------
        spt.Measure object
        """
        # find number and keep its index.
        self.total_parsed_elements += 1
        number = re.findall(r"([0-9]+)", line)
        number_index = line.index(number[0]) if number else line.index("=")
        closing_repeat = re.findall(r"[:|]", line[:number_index])
        opening_repeat = re.findall(r"[|:]", line[number_index:])
        m = spt.Measure(
            number=self.measure_enum, name=int(number[0]) if number else None
        )
        self.measure_enum += 1
        return m

    def meta_chord_line(self, line: str):
        """
        Grammar Defining a chord line.

        A chord line is specified by the following grammar:
        chord_line = note | chord

        Parameters
        ----------
        line

        Returns
        -------

        """
        self.total_parsed_elements += 1
        chord = ("c", [self.meta_note_line(n, add=False) for n in line.split(" ")])
        return chord
