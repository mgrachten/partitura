import re
import partitura.score as spt
import partitura.io as sptio
import os.path as osp
import numpy as np
from urllib.parse import urlparse
import urllib.request
from partitura.utils.music import key_name_to_fifths_mode


def load_rntxt(path: spt.Path, part=None, return_part=False):
    if sptio.is_url(path):
        data = load_data_from_url(path)
        lines = data.split("\n")
    else:
        if not osp.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            lines = f.readlines()
            assert validate_rntxt(lines)

    # remove empty lines
    lines = [line for line in lines if line.strip()]

    parser = RntxtParser(part)
    parser.parse(lines)
    if return_part or part is None:
        return parser.part
    return


def validate_rntxt(lines):
    # TODO: Implement
    return True


def load_data_from_url(url: str):
    with urllib.request.urlopen(url) as response:
        data = response.read().decode()
    return data


class RntxtParser:
    """
    A parser for RNtxt format to a partitura Part.

    For full specification of the format visit:
    https://github.com/MarkGotham/When-in-Rome/blob/master/syntax.md
    """
    def __init__(self, score=None):
        if score is not None:
            self.ref_part = score.parts[0]
            quarter_duration = self.ref_part._quarter_durations[0]
            ref_measures = self.ref_part.measures
            ref_time_sigs = self.ref_part.time_sigs
            ref_keys = self.ref_part.key_sigs
        else:
            quarter_duration = 4
            ref_measures = []
            ref_time_sigs = []
            ref_keys = []
        self.part = spt.Part(id="rn", part_name="Rn", part_abbreviation="rnp", quarter_duration=quarter_duration)
        # include measures
        for measure in ref_measures:
            self.part.add(measure, measure.start.t, measure.end.t)
        # include time signatures
        for time_sig in ref_time_sigs:
            self.part.add(time_sig, time_sig.start.t)
        # include key signatures
        for key in ref_keys:
            self.part.add(key, key.start.t)
        self.measures = {m.number: m for m in self.part.measures}
        self.part.add(spt.Staff(number=1, lines=1), 0)
        self.current_measure = None
        self.current_position = 0
        self.measure_beat_position = 1
        self.current_voice = None
        self.current_note = None
        self.current_chord = None
        self.current_tie = None
        self.num_parsed_romans = 0
        self.key = "C"

    def parse(self, lines):
        # np_lines = np.array(lines)
        # potential_measure_lines = np.lines[np.char.startswith(np_lines, "m")]
        # for line in potential_measure_lines:
        #     self._handle_measure(line)
        for line in lines:
            if line.startswith("Time Signature:"):
                self.time_signature = line.split(":")[1].strip()
            elif line.startswith("Pedal:"):
                self.pedal = line.split(":")[1].strip()
            elif line.startswith("m"):
                self._handle_measure(line)

        self.currate_ending_times()

    def currate_ending_times(self):
        romans = list(self.part.iter_all(spt.RomanNumeral))
        starting_times = [rn.start.t for rn in romans]
        argsort_start = np.argsort(starting_times)
        for i, rn_idx in enumerate(argsort_start[:-1]):
            rn = romans[rn_idx]
            if rn.end is None:
                rn.end = romans[argsort_start[i+1]].start if rn.start.t < romans[argsort_start[i+1]].start.t else rn.start.t + 1

    def _handle_measure(self, line):
        if not self._validate_measure_line(line):
            return
        elements = line.split(" ")
        measure_number = elements[0].strip("m")
        if not measure_number.isnumeric():
            # TODO: complete check for variation measures
            if "var" in measure_number:
                return
            else:

                raise ValueError(f"Invalid measure number: {measure_number}")
        measure_number = int(measure_number)
        if measure_number not in self.measures.keys():
            self.current_measure = spt.Measure(number=measure_number)
            self.measures[measure_number] = self.current_measure
            self.part.add(self.current_measure, int(self.current_position))
        else:
            self.current_measure = self.measures[measure_number]

        self.current_position = self.current_measure.start.t
        # starts counting beats from 1
        self.measure_beat_position = 1
        for element in elements[1:]:
            self._handle_element(element)

    def _handle_element(self, element):
        # if element starts with "b" followed by a number ("float" or "int") it is a beat
        if element.startswith("b") and element[1:].replace(".", "").isnumeric():
            self.measure_beat_position = float(element[1:])
            if self.current_measure.number == 0:
                if (self.current_position == 0 and self.num_parsed_romans == 0):
                    self.current_position = 0
                else:
                    self.current_position = self.part.inv_beat_map(self.part.beat_map(self.current_position) + self.measure_beat_position - 1).item()
            else:
                self.current_position = self.part.inv_beat_map(self.part.beat_map(self.current_measure.start.t) + self.measure_beat_position - 1).item()

        # if element starts with [A-G] and it includes : it is a key
        elif len(re.findall(r"[A-Ga-g#b:]", element)) == len(element) and element[-1] == ":":
            self._handle_key(element)
        # if element only contains "|" or ":" (and combinations) it is a barline
        elif all(c in "|:" for c in element):
            self._handle_barline(element)
        # else it is a roman numeral
        else:
            self._handle_roman_numeral(element)

    def _handle_key(self, element):
        # key is in the format "C:" or "c:" for C major or c minor
        # for alterations use "C#:" or "c#:" for C# major or c# minor
        name = element[0]
        mode = "major" if name.isupper() else "minor"
        step = name.upper()
        # handle alterations
        alter = element[1:].strip(":")
        key_name = f"{step}{alter}{('m' if mode == 'minor' else '')}"
        # step and alter to fifths
        fifths, mode = key_name_to_fifths_mode(key_name)
        ks = spt.KeySignature(fifths=fifths, mode=mode)
        self.key = element.strip(":")
        self.part.add(ks, int(self.current_position))

    def _handle_barline(self, element):
        pass

    def _handle_roman_numeral(self, element):
        """
        The handling or roman numeral aims to translate rntxt notation to internal partitura notation.

        Parameters
        ----------
        element: txt
            The element is a rntxt notation string
        """
        # Remove line endings and spaces
        element = element.strip()
        # change strings such as RN6/5 to RN65 but keep RN65/RN for the secondary degree
        if "/" in element:
            # if all elements between "/" are either digits or one of [o, +] then remove "/" else leave it in place
            el_list = element.split("/")
            element = el_list[0]
            for el in el_list[1:]:
                if len(re.findall(r"[1-9\+o]", el)) == len(el):
                    element += el
                else:
                    element += "/" + el
        # Validity checks happen inside the Roman Numeral object
        # The checks include 1 & 2 Degree, Root, Bass, Inversion, and Quality extraction.
        rn = spt.RomanNumeral(text=element, local_key=self.key)

        try:
            self.part.add(rn, int(self.current_position))
        except ValueError:
            print(f"Could not add roman numeral {element} at position {self.current_position}")
            return
        # Set the end of the previous roman numeral
        # if self.previous_rn is not None:
        #     self.previous_rn.end = spt.TimePoint(t=self.current_position)
        self.num_parsed_romans += 1

    def _validate_measure_line(self, line):
        # does it have elements
        if not len(line.split(" ")) > 1:
            return False
        return True



