#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module defines an ontology of musical elements to represent
musical scores, such as measures, notes, slurs, words, tempo and
loudness directions. A score is defined at the highest level by a
`Part` object (or a hierarchy of `Part` objects, in a `PartGroup`
object). This object serves as a timeline at which musical elements
are registered in terms of their start and end times.
"""
from copy import copy, deepcopy
from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction
from numbers import Number
from partitura.utils.globals import (
    MUSICAL_BEATS,
    INTERVALCLASSES,
    INTERVAL_TO_SEMITONES,
    LABEL_DURS,
)
import warnings, sys
import numpy as np
import re
from scipy.interpolate import PPoly
from typing import Union, List, Optional, Iterator, Any, Iterable as Itertype
import difflib
from partitura.utils import (
    ComparableMixin,
    ReplaceRefMixin,
    iter_subclasses,
    iter_current_next,
    sorted_dict_items,
    PrettyPrintTree,
    ALTER_SIGNS,
    find_tie_split,
    format_symbolic_duration,
    estimate_symbolic_duration,
    symbolic_to_numeric_duration,
    fifths_mode_to_key_name,
    pitch_spelling_to_midi_pitch,
    note_array_from_part,
    rest_array_from_part,
    rest_array_from_part_list,
    note_array_from_part_list,
    to_quarter_tempo,
    key_mode_to_int,
    _OrderedSet,
    update_note_ids_after_unfolding,
    clef_sign_to_int,
)
from partitura.utils.generic import interp1d
from partitura.utils.music import transpose_note, step2pc
from partitura.utils.globals import (
    INT_TO_ALT,
    ALT_TO_INT,
    ACCEPTED_ROMANS,
    LOCAL_KEY_TRASPOSITIONS_DCML,
)


class Part(object):
    """Represents a score part, e.g. all notes of one single instrument
    (or multiple instruments written in the same staff). Note that
    there may be more than one staff per score part.

    Parameters
    ----------
    id : str
        The identifier of the part. In order to be compatible with
        MusicXML the identifier should not start with a number.
    part_name : str or None, optional
        Name for the part. Defaults to None
    part_abbreviation : str or None, optional
        Abbreviated name for part
    quarter_duration : int, optional
        The default quarter duration. See
        :meth:`~partitura.score.Part.set_quarter_duration` for
        details.

    Attributes
    ----------
    id : str
        See parameters
    part_name : str
        See parameters
    part_abbreviation : str
        See parameters

    """

    def __init__(self, id, part_name=None, part_abbreviation=None, quarter_duration=1):
        super().__init__()
        self.id = id
        self.parent = None
        self.part_name = part_name
        self.part_abbreviation = part_abbreviation

        # timeline init
        self._points = np.array([], dtype=TimePoint)
        self._quarter_times = [0]
        self._quarter_durations = [quarter_duration]
        self._quarter_map = self.quarter_duration_map

        # set beat reference
        self._use_musical_beat = False

        # store number of staves
        self._number_of_staves = None

    def __str__(self):
        return 'Part id="{}" name="{}"'.format(self.id, self.part_name)

    def _pp(self, tree):
        result = [self.__str__()]
        tree.push()
        N = len(self._points)
        for i, timepoint in enumerate(self._points):
            result.append("{}".format(tree).rstrip())
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(timepoint._pp(tree))
        tree.pop()
        return result

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation

        """
        return "\n".join(self._pp(PrettyPrintTree()))

    @property
    def time_signature_map(self):
        """A function mapping timeline times to the beats and beat_type
        of the time signature at that time. The function can take
        scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        tss = np.array(
            [
                (ts.start.t, ts.beats, ts.beat_type, ts.musical_beats)
                for ts in self.iter_all(TimeSignature)
            ]
        )

        if len(tss) == 0:
            # default time sig
            beats, beat_type, musical_beats = 4, 4, 4
            warnings.warn(
                "No time signatures found, assuming {}/{}".format(beats, beat_type)
            )
            if self.first_point is None:
                t0, tN = 0, 0
            else:
                t0 = self.first_point.t
                tN = self.last_point.t
            tss = np.array(
                [
                    (t0, beats, beat_type, musical_beats),
                    (tN, beats, beat_type, musical_beats),
                ]
            )
        elif len(tss) == 1:
            # If there is only a single time signature
            tss = np.array([tss[0, :], tss[0, :]])
        elif tss[0, 0] > self.first_point.t:
            tss = np.vstack(
                ((self.first_point.t, tss[0, 1], tss[0, 2], tss[0, 3]), tss)
            )

        return interp1d(
            tss[:, 0],
            tss[:, 1:],
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

    @property
    def key_signature_map(self):
        """A function mappting timeline times to the key and mode of
        the key signature at that time. The function can take scalar
        values or lists/arrays of values

        Returns
        -------
        function
            The mapping function
        """
        kss = np.array(
            [
                (ks.start.t, ks.fifths, key_mode_to_int(ks.mode))
                for ks in self.iter_all(KeySignature)
            ]
        )

        if len(kss) == 0:
            # default key signature
            fifths, mode = 0, 1
            warnings.warn("No key signature found, assuming C major")
            if self.first_point is None:
                t0, tN = 0, 0
            else:
                t0 = self.first_point.t
                tN = self.first_point.t

            kss = np.array([(t0, fifths, mode), (tN, fifths, mode)])

        elif kss[0, 0] > self.first_point.t:
            kss = np.vstack(((self.first_point.t, kss[0, 1], kss[0, 2]), kss))

        return interp1d(
            kss[:, 0],
            kss[:, 1:],
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

    @property
    def clef_map(self):
        """A function mapping timeline times to the clef in each
        staff at that time. The function can take scalar
        values or lists/arrays of values

        Returns
        -------
        function
            The mapping function
        """
        clefs = np.array(
            [
                (
                    c.start.t,
                    c.staff,
                    clef_sign_to_int(c.sign),
                    c.line,
                    c.octave_change if c.octave_change is not None else 0,
                )
                for c in self.iter_all(Clef)
            ]
        )

        interpolators = []
        for s in range(1, self.number_of_staves + 1):
            staff_clefs = clefs[clefs[:, 1] == s]
            if len(staff_clefs) == 0:
                # default "none" clef
                staff, clef, line, octave_change = s, clef_sign_to_int("none"), 0, 0

                warnings.warn(
                    "No clefs found on staff {}, assuming {} clef.".format(s, clef)
                )
                if self.first_point is None:
                    t0, tN = 0, 0
                else:
                    t0 = self.first_point.t
                    tN = self.last_point.t
                staff_clefs = np.array(
                    [
                        (t0, staff, clef, line, octave_change),
                        (tN, staff, clef, line, octave_change),
                    ]
                )
            elif len(staff_clefs) == 1:
                # If there is only a single clef
                staff_clefs = np.array([staff_clefs[0, :], staff_clefs[0, :]])

            if staff_clefs[0, 0] > self.first_point.t:
                staff_clefs = np.vstack(
                    ((self.first_point.t, *staff_clefs[0, 1:]), staff_clefs)
                )

            interpolators.append(
                interp1d(
                    staff_clefs[:, 0],
                    staff_clefs[:, 1:],
                    axis=0,
                    kind="previous",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            )

        def collator(time: Union[int, np.ndarray]) -> np.ndarray:
            return np.array(
                [interpolator(time) for interpolator in interpolators], dtype=int
            )

        return collator

    @property
    def measure_map(self):
        """A function mapping timeline times to the start and end of
        the measure they are contained in. The function can take
        scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        measures = np.array([(m.start.t, m.end.t) for m in self.iter_all(Measure)])

        # correct for anacrusis
        divs_per_beat = self.inv_beat_map(
            1 + self.beat_map(0)
        )  # find the divs per beat in the first measure
        if (
            measures[0][1] - measures[0][0]
            < self.time_signature_map(0)[0] * divs_per_beat
        ):
            measures[0][0] = (
                measures[0][1] - self.time_signature_map(0)[0] * divs_per_beat
            )

        if len(measures) == 0:  # no measures in the piece
            # default only one measure spanning the entire timeline
            warnings.warn("No measures found, assuming only one measure")
            if self.first_point is None:
                t0, tN = 0, 0
            else:
                t0 = self.first_point.t
                tN = self.last_point.t

            measures = np.array([(t0, tN)])

        inter_function = interp1d(
            measures[:, 0],
            measures[:, :].astype(int),
            kind="previous",
            axis=0,
            fill_value="extrapolate",
            dtype=int,
        )

        return inter_function

    @property
    def measure_number_map(self):
        """A function mapping timeline times to the measure number of
        the measure they are contained in. The function can take
        scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        # operations to avoid None values and filter them efficiently.
        m_it = self.measures

        measures = np.array(
            [
                [
                    m.start.t,
                    m.end.t,
                    (m_it[i - 1].number if m.number == None else m.number),
                ]
                for i, m in enumerate(m_it)
            ]
        )
        # correct for anacrusis
        divs_per_beat = self.inv_beat_map(
            1 + self.beat_map(0)
        )  # find the divs per beat in the first measure
        if (
            measures[0][1] - measures[0][0]
            < self.time_signature_map(0)[0] * divs_per_beat
        ):
            measures[0][0] = (
                measures[0][1] - self.time_signature_map(0)[0] * divs_per_beat
            )

        if len(measures) == 0:  # no measures in the piece
            # default only one measure spanning the entire timeline
            warnings.warn("No measures found, assuming only one measure")
            if self.first_point is None:
                t0, tN = 0, 0
            else:
                t0 = self.first_point.t
                tN = self.last_point.t

            measures = np.array([(t0, tN, 1)])

        inter_function = interp1d(
            measures[:, 0],
            measures[:, 2],
            kind="previous",
            fill_value="extrapolate",
            dtype=int,
        )

        return inter_function

    @property
    def metrical_position_map(self):
        """A function mapping timeline times to their relative position in
        the measure they are contained in. The function can take
        scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        measure_map = self.measure_map
        ms = [measure_map(m.start.t)[0] for m in self.iter_all(Measure)]
        me = [measure_map(m.start.t)[1] for m in self.iter_all(Measure)]

        if len(ms) < 2:
            warnings.warn("No or single measures found, metrical position 0 everywhere")
            zero_interpolator = interp1d(
                np.arange(0, 2),
                np.zeros((2, 2)),
                axis=0,
                kind="linear",
                fill_value="extrapolate",
                dtype=int,
            )

            return zero_interpolator
        else:
            barlines = np.array(ms + me[-1:])
            bar_durations = np.diff(barlines)
            measure_inter_function = interp1d(
                barlines[:-1],
                bar_durations,
                axis=0,
                kind="previous",
                fill_value="extrapolate",
            )

            lin_poly_coeff = np.row_stack(
                (np.ones(bar_durations.shape[0]), np.zeros(bar_durations.shape[0]))
            )
            inter_function = PPoly(lin_poly_coeff, barlines)

            def int_interp1d(input):
                if isinstance(input, Iterable):
                    return np.column_stack(
                        (
                            inter_function(input).astype(int),
                            measure_inter_function(input).astype(int),
                        )
                    )
                else:
                    return (
                        inter_function(input).astype(int),
                        measure_inter_function(input).astype(int),
                    )

            return int_interp1d

    def _time_interpolator(self, quarter=False, inv=False, musical_beat=False):
        if len(self._points) < 2:
            return lambda x: np.zeros(len(x))

        keypoints = defaultdict(lambda: [None, None])
        _ = keypoints[self.first_point.t]
        _ = keypoints[self.last_point.t]
        for t, q in zip(self._quarter_times, self._quarter_durations):
            keypoints[t][0] = q
        if not quarter:
            for ts in self.iter_all(TimeSignature):
                # keypoints[ts.start.t][1] = int(np.log2(ts.beat_type))
                if musical_beat:
                    keypoints[ts.start.t][1] = (ts.beat_type / 4) * (
                        ts.musical_beats / ts.beats
                    )
                else:
                    keypoints[ts.start.t][1] = ts.beat_type / 4
        cur_div = 1
        cur_bt = 1
        keypoints_list = []

        for t in sorted(keypoints.keys()):
            kp = keypoints[t]
            if kp[0] is None:
                kp[0] = cur_div
            else:
                cur_div = kp[0]
            if kp[1] is None:
                kp[1] = cur_bt
            else:
                cur_bt = kp[1]
            if not keypoints_list or kp != keypoints_list[-1]:
                keypoints_list.append([t] + kp)
        keypoints = np.array(keypoints_list, dtype=float)

        x = keypoints[:, 0]
        y = np.r_[
            0,
            np.cumsum(
                (keypoints[:-1, 2] * np.diff(keypoints[:, 0])) / keypoints[:-1, 1]
            ),
        ]

        m1 = next(self.first_point.iter_starting(Measure), None)

        if m1 and m1.start is not None and m1.end is not None:
            f = interp1d(x, y)
            actual_dur = np.diff(f((m1.start.t, m1.end.t)))[0]
            ts = next(m1.start.iter_starting(TimeSignature), None)

            if ts:
                normal_dur = ts.beats
                if quarter:
                    normal_dur *= 4 / ts.beat_type
                if musical_beat:
                    normal_dur = ts.musical_beats
                if actual_dur < normal_dur:
                    y -= actual_dur
            else:
                # warn
                pass

        if inv:
            return interp1d(y, x)
        else:
            return interp1d(x, y)

    @property
    def beat_map(self):
        """A function mapping timeline times to beat times. The function
        can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        if self._use_musical_beat:
            return self._time_interpolator(musical_beat=True)
        else:
            return self._time_interpolator()

    @property
    def inv_beat_map(self):
        """A function mapping beat times to timeline times. The function
        can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        if self._use_musical_beat:
            return self._time_interpolator(inv=True, musical_beat=True)
        else:
            return self._time_interpolator(inv=True)

    @property
    def quarter_map(self):
        """A function mapping timeline times to quarter times. The
        function can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        return self._time_interpolator(quarter=True)

    @property
    def inv_quarter_map(self):
        """A function mapping quarter times to timeline times. The
        function can take scalar values or lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        return self._time_interpolator(quarter=True, inv=True)

    @property
    def notes(self):
        """Return a list of all Note objects in the part. This list includes
        GraceNote objects but not Rest objects.

        Returns
        -------
        list
            list of Note objects

        """
        return list(self.iter_all(Note, include_subclasses=True))

    @property
    def notes_tied(self):
        """Return a list of all Note objects in the part that are
        either not tied, or the first note of a group of tied notes.
        This list includes GraceNote objects but not Rest objects.

        Returns
        -------
        list
            List of Note objects

        """
        return [
            note
            for note in self.iter_all(Note, include_subclasses=True)
            if note.tie_prev is None
        ]

    @property
    def measures(self):
        """Return a list of all Measure objects in the part

        Returns
        -------
        list
            List of Measure objects

        """
        return [e for e in self.iter_all(Measure, include_subclasses=False)]

    @property
    def rests(self):
        """Return a list of all rest objects in the part

        Returns
        -------
        list
            List of Rest objects

        """
        return [e for e in self.iter_all(Rest, include_subclasses=False)]

    @property
    def cadences(self):
        """Return a list of all cadence objects in the part

        Returns
        -------
        list
            List of Cadence objects

        """
        return [e for e in self.iter_all(Cadence)]

    @property
    def repeats(self):
        """Return a list of all Repeat objects in the part

        Returns
        -------
        list
            List of Repeat objects

        """
        return [e for e in self.iter_all(Repeat, include_subclasses=False)]

    @property
    def key_sigs(self):
        """Return a list of all Key Signature objects in the part

        Returns
        -------
        list
            List of Key Signature objects

        """
        return [e for e in self.iter_all(KeySignature, include_subclasses=False)]

    @property
    def time_sigs(self):
        """Return a list of all Time Signature objects in the part

        Returns
        -------
        list
            List of Time Signature objects

        """
        return [e for e in self.iter_all(TimeSignature, include_subclasses=False)]

    @property
    def dynamics(self):
        """Return a list of all Dynamics markings in the part

        Returns
        -------
        list
            List of Dynamics objects

        """
        return [e for e in self.iter_all(LoudnessDirection, include_subclasses=True)]

    @property
    def tempo_directions(self):
        """Return a list of all tempo direction in the part

        Returns
        -------
        list
            List of TempoDirection objects

        """
        return [e for e in self.iter_all(TempoDirection, include_subclasses=True)]

    @property
    def cadences(self):
        """Return a list of all cadences in the part

        Returns
        -------
        list
            List of Cadence objects

        """
        return [e for e in self.iter_all(Cadence, include_subclasses=False)]

    @property
    def harmony(self):
        """Return a list of all harmony in the part

        Returns
        -------
        list
            List of Harmony objects

        """
        return [e for e in self.iter_all(Harmony, include_subclasses=True)]

    @property
    def phrases(self):
        """Return a list of all phrases in the part

        Returns
        -------
        list
            List of Phrase objects
        """
        return [e for e in self.iter_all(Phrase, include_subclasses=False)]

    @property
    def articulations(self):
        """Return a list of all Articulation markings in the part

        Returns
        -------
        list
            List of Articulation objects

        """
        return [
            e for e in self.iter_all(ArticulationDirection, include_subclasses=True)
        ]

    @property
    def segments(self):
        """Return a list of all segments in the part

        Returns
        -------
        list
            List of Segment objects

        """
        add_segments(self)
        return [e for e in self.iter_all(Segment, include_subclasses=False)]

    def quarter_durations(self, start=None, end=None):
        """Return an Nx2 array with quarter duration (second column)
        and their respective times (first column).

        When a start and or end time is specified, the returned
        array will contain only the entries within those bounds.

        Parameters
        ----------
        start : number, optional
            Start of range
        end : number, optional
            End of range

        Returns
        -------
        ndarray
            An array with quarter durations and times

        """

        qd = np.column_stack((self._quarter_times, self._quarter_durations))
        if start is not None:
            qd = qd[qd[:, 0] >= start, :]
        if end is not None:
            qd = qd[qd[:, 0] < end, :]
        return qd

    @property
    def quarter_duration_map(self):
        """A function mapping timeline times to quarter durations in
        effect at those times. The function can take scalar values or
        lists/arrays of values.

        Returns
        -------
        function
            The mapping function

        """
        x = self._quarter_times
        y = self._quarter_durations
        if len(x) == 1:
            x = x + x
            y = y + y
        return interp1d(
            x, y, kind="previous", bounds_error=False, fill_value=(y[0], y[-1])
        )

    def set_quarter_duration(self, t, quarter):
        """Set the duration of a quarter note from timepoint `t`
        onwards.

        Setting the quarter note duration defines how intervals
        between timepoints are related to musical durations. For
        example when two timepoints `t1` and `t2` have associated
        times 10 and 20 respecively, then the interval between `t1`
        and `t2` corresponds to a half note when the quarter duration
        equals 5 during that interval.

        The quarter duration can vary throughout the part. When
        setting a quarter duration at time t, then that value takes
        effect until the time of the next quarter duration. If a
        different quarter duration was already set at time t, it wil
        be replaced.

        Note setting the quarter duration does not change the
        timepoints, only the relation to musical time. For
        illustration: in the example above, when changing the current
        quarter duration from 5 to 10, a note that starts at `t1` and
        ends at `t2` will change from being a half note to being a
        quarter note.

        Parameters
        ----------
        t : int
            Time at which to set the quarter duration
        quarter : int
            The quarter duration

        """

        # add quarter duration at time t, unless it is redundant. If another
        # quarter duration is at t, replace it.

        # shorthand
        times = self._quarter_times
        quarters = self._quarter_durations

        i = np.searchsorted(times, t)
        changed = False

        if i == 0 or quarters[i - 1] != quarter:
            # add or replace
            if i == len(times) or times[i] != t:
                # add
                times.insert(i, t)
                quarters.insert(i, quarter)
                changed = True
            elif quarters[i] != quarter:
                # replace
                quarters[i] = quarter
                changed = True
            else:
                # times[i] == t, quarters[i] == quarter
                pass

        if not changed:
            return

        if i + 1 == len(times):
            t_next = np.inf
        else:
            t_next = times[i + 1]

        # update quarter attribute of all timepoints in the range [t, t_next]
        start_idx = np.searchsorted(self._points, TimePoint(t))
        end_idx = np.searchsorted(self._points, TimePoint(t_next))
        for tp in self._points[start_idx:end_idx]:
            tp.quarter = quarter

        # update the interpolation function
        self._quarter_map = self.quarter_duration_map

    def _add_point(self, tp):
        # Add `TimePoint` object `tp` to the part, unless there is
        # already a timepoint at the same time.

        i = np.searchsorted(self._points, tp)
        if i == len(self._points) or self._points[i].t != tp.t:
            self._points = np.insert(self._points, i, tp)
            if i > 0:
                self._points[i - 1].next = self._points[i]
                self._points[i].prev = self._points[i - 1]
            if i < len(self._points) - 1:
                self._points[i].next = self._points[i + 1]
                self._points[i + 1].prev = self._points[i]

    @property
    def number_of_staves(self):
        if self._number_of_staves is not None:
            return self._number_of_staves
        else:
            return self.compute_number_of_staves()

    def compute_number_of_staves(self):
        max_staves = 1
        for e in self.iter_all(GenericNote, include_subclasses=True):
            if e.staff is not None and e.staff > max_staves:
                max_staves = e.staff
        for e in self.iter_all(Clef):
            if e.staff is not None and e.staff > max_staves:
                max_staves = e.staff
        for e in self.iter_all(Direction, include_subclasses=True):
            if e.staff is not None and e.staff > max_staves:
                max_staves = e.staff
        for e in self.iter_all(Words):
            if e.staff is not None and e.staff > max_staves:
                max_staves = e.staff

        self._number_of_staves = max_staves
        return max_staves

    def _remove_point(self, tp):
        i = np.searchsorted(self._points, tp)
        if self._points[i] == tp:
            self._points = np.delete(self._points, i)
            if i > 0:
                self._points[i - 1].next = self._points[i]
                self._points[i].prev = self._points[i - 1]
            if i < len(self._points) - 1:
                self._points[i].next = self._points[i + 1]
                self._points[i + 1].prev = self._points[i]

    def get_point(self, t):
        """Return the `TimePoint` object with time `t`, or None if
        there is no such object.

        """
        if t < 0:
            raise InvalidTimePointException(
                "TimePoints should have non-negative integer values"
            )

        i = np.searchsorted(self._points, TimePoint(t))
        if i < len(self._points) and self._points[i].t == t:
            return self._points[i]
        else:
            return None

    def get_or_add_point(self, t):
        """Return the `TimePoint` object with time `t`; if there is no
        such object, create it, add it to the time line, and return
        it.

        Parameters
        ----------
        t : int
            Time value `t`

        Returns
        -------
        :class:`TimePoint`
            a TimePoint object with time `t`

        """
        if t < 0:
            raise InvalidTimePointException(
                "TimePoints should have non-negative integer values"
            )

        tp = self.get_point(t)
        if tp is None:
            tp = TimePoint(t, int(self._quarter_map(t)))
            self._add_point(tp)
        return tp

    def add(self, o, start=None, end=None):
        """Add an object to the timeline.

        An object can be added by start time, end time, or both,
        depending on which of the `start` and `end` keywords are
        provided. If neither is provided this method does nothing.

        `start` and `end` should be non-negative integers.

        Parameters
        ----------
        o : :class:`TimedObject`
            Object to be removed
        start : int, optional
            The start time of the object
        end : int, optional
            The end time of the object

        """
        if start is not None:
            if start < 0:
                raise InvalidTimePointException(
                    "TimePoints should have non-negative integer values"
                )
            self.get_or_add_point(start).add_starting_object(o)
        if end is not None:
            if end < 0:
                raise InvalidTimePointException(
                    "TimePoints should have non-negative integer values"
                )
            self.get_or_add_point(end).add_ending_object(o)

    def remove(self, o, which="both"):
        """Remove an object from the timeline.

        An object can be removed by start time, end time, or both.

        Parameters
        ----------
        o : :class:`TimedObject`
            Object to be removed
        which : {'start', 'end', 'both'}, optional
            Whether to remove o as a starting object, an ending
            object, or both. Defaults to 'both'.

        """

        if which in ("start", "both") and o.start:
            try:
                o.start.starting_objects[o.__class__].remove(o)
            except (KeyError, ValueError):
                raise Exception(
                    "Not implemented: removing an object "
                    "that is registered by its superclass"
                )
            # cleanup timepoint if no starting/ending objects are left
            self._cleanup_point(o.start)
            o.start = None

        if which in ("end", "both") and o.end:
            try:
                o.end.ending_objects[o.__class__].remove(o)
            except (KeyError, ValueError):
                raise Exception(
                    "Not implemented: removing an object "
                    "that is registered by its superclass"
                )
            # cleanup timepoint if no starting/ending objects are left
            self._cleanup_point(o.end)
            o.end = None

    def _cleanup_point(self, tp):
        # remove tp when it has no starting or ending objects
        if (
            sum(len(oo) for oo in tp.starting_objects.values())
            + sum(len(oo) for oo in tp.ending_objects.values())
        ) == 0:
            self._remove_point(tp)

    def iter_all(
        self, cls=None, start=None, end=None, include_subclasses=False, mode="starting"
    ):
        """Iterate (in direction of increasing time) over all
        instances of `cls` that either start or end (depending on
        `mode`) in the interval `start` to `end`.  When `start` and
        `end` are omitted, the whole timeline is searched.

        Parameters
        ----------
        cls : class, optional
            The class of objects to iterate over. If omitted, iterate
            over all objects in the part.
        start : :class:`TimePoint`, optional
            The start of the interval to search. If omitted or None,
            the search starts at the start of the timeline. Defaults
            to None.
        end : :class:`TimePoint`, optional
            The end of the interval to search. If omitted or None, the
            search ends at the end of the timeline. Defaults to None.
        include_subclasses : bool, optional
            If True also return instances that are subclasses of
            `cls`. Defaults to False.
        mode : {'starting', 'ending'}, optional
            Flag indicating whether to search for starting or ending
            objects. Defaults to 'starting'.

        Yields
        ------
        object
            Instances of the specified type.

        """
        if mode not in ("starting", "ending"):
            warnings.warn('unknown mode "{}", using "starting" instead'.format(mode))
            mode = "starting"

        if start is None:
            start_idx = 0
        else:
            if not isinstance(start, TimePoint):
                start = TimePoint(start)
            start_idx = np.searchsorted(self._points, start)

        if end is None:
            end_idx = len(self._points)
        else:
            if not isinstance(end, TimePoint):
                end = TimePoint(end)
            end_idx = np.searchsorted(self._points, end)

        if cls is None:
            cls = object
            include_subclasses = True

        if mode == "ending":
            for tp in self._points[start_idx:end_idx]:
                yield from tp.iter_ending(cls, include_subclasses)
        else:
            for tp in self._points[start_idx:end_idx]:
                yield from tp.iter_starting(cls, include_subclasses)

    def apply(self):
        """Apply all changes to the timeline for objects like octave Shift."""
        pass

    @property
    def last_point(self):
        """The last TimePoint on the timeline, or None if the timeline
        is empty.

        Returns
        -------
        :class:`TimePoint`

        """
        return self._points[-1] if len(self._points) > 0 else None

    @property
    def first_point(self):
        """The first TimePoint on the timeline, or None if the
        timeline is empty.

        Returns
        -------
        :class:`TimePoint`

        """
        return self._points[0] if len(self._points) > 0 else None

    def note_array(self, **kwargs):
        """
        Create a structured array with note information
        from a `Part` object.

        Parameters
        ----------

        include_pitch_spelling : bool (optional)
            If `True`, includes pitch spelling information for each
            note. Default is False
        include_key_signature : bool (optional)
            If `True`, includes key signature information, i.e.,
            the key signature at the onset time of each note (all
            notes starting at the same time have the same key signature).
            Default is False
        include_time_signature : bool (optional)
            If `True`,  includes time signature information, i.e.,
            the time signature at the onset time of each note (all
            notes starting at the same time have the same time signature).
            Default is False
        include_metrical_position : bool (optional)
            If `True`,  includes metrical position information, i.e.,
            the position of the onset time of each note with respect to its
            measure (all notes starting at the same time have the same metrical
            position).
            Default is False
        include_grace_notes : bool (optional)
            If `True`,  includes grace note information, i.e. if a note is a
            grace note and the grace type "" for non grace notes).
            Default is False
        include_divs_per_quarter : bool (optional)
            If `True`,  includes the number of divs per quarter note.
            Default is False

        Returns:

        note_array : structured array
        """
        return note_array_from_part(self, **kwargs)

    def rest_array(
        self,
        include_pitch_spelling=False,
        include_key_signature=False,
        include_time_signature=False,
        include_metrical_position=False,
        include_grace_notes=False,
        include_staff=False,
        collapse=False,
    ):
        """
        Create a structured array with rest information
        from a `Part` object.

        Parameters
        ----------

        include_pitch_spelling : bool (optional)
            If `True`, includes pitch spelling information for each
            rest, i.e. all information is 0. Default is False
        include_key_signature : bool (optional)
            If `True`, includes key signature information, i.e.,
            the key signature at the onset time of each rest (all
            notes starting at the same time have the same key signature).
            Default is False
        include_time_signature : bool (optional)
            If `True`,  includes time signature information, i.e.,
            the time signature at the onset time of each note (all
            notes starting at the same time have the same time signature).
            Default is False
        include_metrical_position : bool (optional)
            If `True`,  includes metrical position information, i.e.,
            the position of the onset time of each rest with respect to its
            measure (all notes starting at the same time have the same metrical
            position).
            Default is False
        include_grace_notes : bool (optional)
            If `True`,  includes returns empty strings as type and false.
        feature_functions : list or str
            A list of feature functions. Elements of the list can be either
            the functions themselves or the names of a feature function as
            strings (or a mix). The feature functions specified by name are
            looked up in the `featuremixer.featurefunctions` module.

        Returns:

        rest_array : structured array
        """
        return rest_array_from_part(
            self,
            include_pitch_spelling=include_pitch_spelling,
            include_key_signature=include_key_signature,
            include_time_signature=include_time_signature,
            include_metrical_position=include_metrical_position,
            include_grace_notes=include_grace_notes,
            include_staff=include_staff,
            collapse=collapse,
        )

    def set_musical_beat_per_ts(self, mbeats_per_ts={}):
        """Set the number of musical beats for each time signature.
        If no musical beat is specified for a certain time signature,
        the default one is used, i.e. 2 for 6/X, 3 for 9/X, 4 for 12/X,
        and the number of beats for the others ts. Each musical beat
        has equal duration.

        Parameters
        ----------
        mbeats_per_ts : dict, optional
            A dict where the keys are time signature strings
            (e.g. "3/4") and the values are the number of musical beats.
            If a certain time signature is not specified, the defaults
            values are used.
            Defaults to an empty dict.

        """
        if not isinstance(mbeats_per_ts, dict):
            raise TypeError("mbeats_per_ts must be either a dictionary")

        # correctly set the musical beat for all time signatures
        for ts in self.iter_all(TimeSignature):
            ts_string = "{}/{}".format(ts.beats, ts.beat_type)
            if ts_string in mbeats_per_ts:
                ts.musical_beats = mbeats_per_ts[ts_string]
            else:  # set to default if not specified
                if ts.beats in MUSICAL_BEATS:
                    ts.musical_beats = MUSICAL_BEATS[ts.beats]
                else:
                    ts.musical_beats = ts.beats

    def use_musical_beat(self, mbeats_per_ts={}):
        """Consider the musical beat as the reference for all elements
        that concern the number and position of beats.
        An optional parameter can set the number of musical beats for
        specific time signatures, otherwise the default values are
        used.

        Parameters
        ----------
        mbeats_per_ts : dict, optional
            A dict where the keys are time signature strings
            (e.g. "3/4") and the values are the number of musical beats.
            If a certain time signature is not specified, the defaults
            values are used.
            Defaults to an empty dict.

        """
        if not self._use_musical_beat:
            self._use_musical_beat = True
            if mbeats_per_ts != {}:  # set the number of nbeats if specified
                self.set_musical_beat_per_ts(mbeats_per_ts)
        else:
            warnings.warn("Musical beats were already being used!")

    def use_notated_beat(self):
        """Consider the notated beat (numerator of time signature)
        as the reference for all elements that concern the number
        and position of beats.
        It also reset the number of musical beats for each time signature
        to default values.
        """
        if self._use_musical_beat:
            self._use_musical_beat = False
            # reset the number of musical beats to default values
            self.set_musical_beat_per_ts()
        else:
            warnings.warn("Notated beats were already being used!")

    # @property
    # def part_names(self):
    #     # get instrument name parts recursively
    #     chunks = []

    #     if self.part_name is not None:
    #         chunks.append(self.part_name)
    #         yield self.part_name

    #     pg = self.parent
    #     while pg is not None:
    #         if pg.group_name is not None:
    #             chunks.insert(0, pg.group_name)
    #             yield '  '.join(chunks)
    #         pg = pg.parent


class TimePoint(ComparableMixin):
    """A TimePoint represents a temporal position within a
    :class:`Part`.

    TimePoints are used to keep track of the starting and ending of
    musical elements in the part. They are created automatically when
    adding musical elements to a part using its :meth:`~Part.add`
    method, so there should be normally no reason to instantiate
    TimePoints manually.

    Parameters
    ----------
    t : int
        The time associated to this TimePoint. Should be a non-
        negative integer.
    quarter : int
        The duration of a quarter note at this TimePoint

    Attributes
    ----------
    t : int
        See parameters
    quarter : int
        See parameters
    starting_objects : dictionary
        A dictionary where the musical objects starting at this time
        are grouped by class.
    ending_objects : dictionary
        A dictionary where the musical objects ending at this time are
        grouped by class.
    prev : TimePoint
        The preceding TimePoint (or None if there is none)
    next : TimePoint
        The succeding TimePoint (or None if there is none)

    """

    def __init__(self, t, quarter=None):
        self.t = t
        self.quarter = quarter
        self.starting_objects = defaultdict(_OrderedSet)
        self.ending_objects = defaultdict(_OrderedSet)
        # prev and next are dynamically updated once the timepoint is part of a timeline
        self.next = None
        self.prev = None

    def __iadd__(self, value):
        assert isinstance(value, Number)
        self.t += value
        return self

    def __isub__(self, value):
        assert isinstance(value, Number)
        self.t -= value
        return self

    def __add__(self, value):
        assert isinstance(value, Number)
        new = copy(self)
        new += value
        return new

    def __sub__(self, value):
        assert isinstance(value, Number)
        new = copy(self)
        new -= value
        return new

    def __str__(self):
        return "TimePoint t={} quarter={}".format(self.t, self.quarter)

    def add_starting_object(self, obj):
        """Add object `obj` to the list of starting objects."""
        obj.start = self
        self.starting_objects[type(obj)].add(obj)

    def remove_starting_object(self, obj):
        """Remove object `obj` from the list of starting objects."""
        # TODO: check if object is stored under a superclass
        obj.start = None
        if type(obj) in self.starting_objects:
            try:
                self.starting_objects[type(obj)].remove(obj)
            except ValueError:
                # don't complain if the object isn't in starting_objects
                pass

    def remove_ending_object(self, obj):
        """Remove object `obj` from the list of ending objects."""
        # TODO: check if object is stored under a superclass
        obj.end = None
        if type(obj) in self.ending_objects:
            try:
                self.ending_objects[type(obj)].remove(obj)
            except ValueError:
                # don't complain if the object isn't in ending_objects
                pass

    def add_ending_object(self, obj):
        """Add object `obj` to the list of ending objects."""
        obj.end = self
        self.ending_objects[type(obj)].add(obj)

    def iter_starting(self, cls, include_subclasses=False):
        """Iterate over all objects of type `cls` that start at this
        time point.

        Parameters
        ----------
        cls : class
            The type of objects to iterate over
        include_subclasses : bool, optional
            When True, include all objects of all subclasses of `cls`
            in the iteration. Defaults to False.

        Yields
        -------
        cls
            Instance of type `cls`

        """
        yield from self.starting_objects[cls]
        if include_subclasses:
            for subcls in iter_subclasses(cls):
                yield from self.starting_objects[subcls]

    def iter_ending(self, cls, include_subclasses=False):
        """Iterate over all objects of type `cls` that end at this
        time point.

        Parameters
        ----------
        cls : class
            The type of objects to iterate over
        include_subclasses : bool, optional
            When True, include all objects of all subclasses of `cls`
            in the iteration. Defaults to False.

        Yields
        ------
        cls
            Instance of type `cls`

        """
        yield from self.ending_objects[cls]
        if include_subclasses:
            for subcls in iter_subclasses(cls):
                yield from self.ending_objects[subcls]

    def iter_prev(self, cls, eq=False, include_subclasses=False):
        """Iterate backwards in time from the current timepoint over
        starting object(s) of type `cls`.

        Parameters
        ----------
        cls : class
            Class of objects to iterate over
        eq : bool, optional
            If True start iterating at the current timepoint, rather
            than its predecessor. Defaults to False.
        include_subclasses : bool, optional
            If True include subclasses of `cls` in the iteration.
            Defaults to False.

        Yields
        ------
        cls
            Instances of `cls`

        """
        if eq:
            tp = self
        else:
            tp = self.prev

        while tp:
            yield from tp.iter_starting(cls, include_subclasses)
            tp = tp.prev

    def iter_next(self, cls, eq=False, include_subclasses=False):
        """Iterate forwards in time from the current timepoint over
        starting object(s) of type `cls`.

        Parameters
        ----------
        cls : class
            Class of objects to iterate over
        eq : bool, optional
            If True start iterating at the current timepoint, rather
            than its successor. Defaults to False.
        include_subclasses : bool, optional
            If True include subclasses of `cls` in the iteration.
            Defaults to False.

        Yields
        ------
        cls
            Instances of `cls`

        """
        if eq:
            tp = self
        else:
            tp = self.next

        while tp:
            yield from tp.iter_starting(cls, include_subclasses)
            tp = tp.next

    def _cmpkey(self):
        # This method returns the value to be compared (code for that is in
        # the ComparableMixin class)
        return self.t

    def _pp(self, tree):
        # pretty print the timepoint, including its starting and ending
        # objects
        result = ["{}{}".format(tree, self.__str__())]
        tree.push()

        ending_items_lists = sorted_dict_items(
            self.ending_objects.items(), key=lambda x: x[0].__name__
        )
        starting_items_lists = sorted_dict_items(
            self.starting_objects.items(), key=lambda x: x[0].__name__
        )

        ending_items = [
            o
            for _, oo in ending_items_lists
            for o in sorted(oo, key=lambda x: x.duration or -1, reverse=True)
        ]
        starting_items = [
            o
            for _, oo in starting_items_lists
            for o in sorted(oo, key=lambda x: x.duration or -1)
        ]

        if ending_items:
            result.append("{}".format(tree).rstrip())

            if starting_items:
                tree.next_item()
            else:
                tree.last_item()

            result.append("{}ending objects".format(tree))
            tree.push()
            result.append("{}".format(tree).rstrip())

            for i, item in enumerate(ending_items):
                if i == (len(ending_items) - 1):
                    tree.last_item()
                else:
                    tree.next_item()

                result.append("{}{}".format(tree, item))

            tree.pop()

        if starting_items:
            result.append("{}".format(tree).rstrip())
            tree.last_item()
            result.append("{}starting objects".format(tree))
            tree.push()
            result.append("{}".format(tree).rstrip())

            for i, item in enumerate(starting_items):
                if i == (len(starting_items) - 1):
                    tree.last_item()
                else:
                    tree.next_item()
                result.append("{}{}".format(tree, item))

            tree.pop()

        tree.pop()
        return result


class TimedObject(ReplaceRefMixin):
    """This is the base class of all classes that have a start and end
    point. The start and end attributes initialized to None, and are
    set/unset when the object is added to/removed from a Part, using
    its :meth:`~Part.add` and :meth:`~Part.remove` methods,
    respectively.

    Attributes
    ----------
    start : :class:`TimePoint`
        Start time of the object
    end : :class:`TimePoint`
        End time of the object

    """

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

    def __str__(self):
        start = "" if self.start is None else f"{self.start.t}"
        end = "" if self.end is None else f"{self.end.t}"
        return start + "--" + end + " " + type(self).__name__

    @property
    def duration(self):
        """The duration of the timed object in divisions. When either
        the start or the end property of the object are None, the
        duration is None.

        Returns
        -------
        int or None

        """
        if self.start is None or self.end is None:
            return None
        else:
            return self.end.t - self.start.t


class GenericNote(TimedObject):
    """Represents the common aspects of notes, rests, and unpitched
    notes.

    Parameters
    ----------
    id : str, optional (default: None)
        A string identifying the note. To be compatible with the
        MusicXML format, the id must be unique within a part and must
        not start with a number.
    voice : int, optional
        An integer representing the voice to which the note belongs.
        Defaults to None.
    staff : str, optional
        An integer representing the staff to which the note belongs.
        Defaults to None.
    doc_order : int, optional
        The document order index (zero-based), expressing the order of
        appearance of this note (with respect to other notes) in the
        document in case the Note belongs to a part that was imported
        from MusicXML. Defaults to None.
    stem_direction : str, optional
        The stem direction of the note. Can be 'up', 'down', or None.
        Defaults to None.
    technical: list, optional
        Technical notation elements.

    """

    def __init__(
        self,
        id=None,
        voice=None,
        staff=None,
        symbolic_duration=None,
        articulations=None,
        ornaments=None,
        doc_order=None,
        stem_direction=None,
        technical=None,
        **kwargs,
    ):
        self._sym_dur = None
        super().__init__(**kwargs)
        self.voice = voice
        self.id = id
        self.staff = staff
        self.symbolic_duration = symbolic_duration
        self.articulations = articulations
        self.ornaments = ornaments
        self.technical = technical
        self.doc_order = doc_order
        self.stem_direction = (
            stem_direction if stem_direction in ("up", "down") else None
        )
        # these attributes are set after the instance is constructed
        self.fermata = None
        self.tie_prev = None
        self.tie_next = None
        self.slur_stops = []
        self.slur_starts = []
        self.tuplet_stops = []
        self.tuplet_starts = []

        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(
            [
                "tie_prev",
                "tie_next",
                "slur_stops",
                "slur_starts",
                "tuplet_stops",
                "tuplet_starts",
            ]
        )

    @property
    def symbolic_duration(self):
        """The symbolic duration of the note.

        This property returns a dictionary specifying the symbolic
        duration of the note. The dictionary may have the following
        keys:

        * type : the note type as a string, e.g. 'quarter', 'half'

        * dots : an integer specifying the number of dots. When
          this key is missing it means there are no dots.

        * actual_notes : Specifies the number of actual notes in a
          rhythmical tuplet. Used in conjunction with `normal_notes`.

        * normal_notes : Specifies the normal number of notes in a
          rhythmical tuplet. For example a triplet of eights in the
          time of two eights would correspond to actual_notes=3,
          normal_notes=2.

        The symbolic duration dictionary of a note can either be
        set manually (for example by specifying the
        `symbolic_duration` constructor keyword argument), or left
        unspecified (i.e. None). In the latter case the symbolic
        duration is estimated dynamically based on the note start and
        end times. Note that this latter case is generally preferrable
        because it ensures that the symbolic duration is consistent
        with the numeric duration.

        If the symbolic duration cannot be estimated from the
        numeric duration None is returned.

        Returns
        -------
        dict or None
            A dictionary specifying the symbolic duration of the note, or
            None if the symbolic duration could not be estimated from the
            numeric duration.

        """
        if self._sym_dur is None:
            # compute value
            if not self.start or not self.end:
                warnings.warn(
                    "Cannot estimate symbolic duration for notes that "
                    "are not added to a Part"
                )
                return None
            if self.start.quarter is None:
                warnings.warn(
                    "Cannot estimate symbolic duration when not "
                    "quarter_duration has been set. "
                    "See Part.set_quarter_duration."
                )
                return None
            return estimate_symbolic_duration(self.duration, self.start.quarter)
        else:
            # return set value
            return self._sym_dur

    @symbolic_duration.setter
    def symbolic_duration(self, v):
        self._sym_dur = v

    @property
    def end_tied(self):
        """The `Timepoint` corresponding to the end of the note, or---
        when this note belongs to a group of tied notes---the end of
        the last note in the group.

        Returns
        -------
        TimePoint
            End of note

        """
        if self.tie_next is None:
            return self.end
        else:
            return self.tie_next.end_tied

    @property
    def duration_tied(self):
        """Time difference of the start of the note to the end of the
        note, or---when  this note belongs to a group of tied notes---
        the end of the last note in the group.

        Returns
        -------
        int
            Duration of note

        """
        if self.tie_next is None:
            return self.duration
        else:
            return self.duration + self.tie_next.duration_tied

    @property
    def duration_from_symbolic(self):
        """Return the numeric duration given the symbolic duration of
        the note and the quarter_duration in effect.

        Returns
        -------
        int or None

        """

        if self.symbolic_duration:
            # check for self.start, and self.start.quarter
            return symbolic_to_numeric_duration(
                self.symbolic_duration, self.start.quarter
            )
        else:
            return None

    @property
    def tie_prev_notes(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        if self.tie_prev:
            return self.tie_prev.tie_prev_notes + [self.tie_prev]
        else:
            return []

    @property
    def tie_next_notes(self):
        """TODO

        Parameters
        ----------

        Returns
        -------
        type
            Description of return value
        """

        if self.tie_next:
            return [self.tie_next] + self.tie_next.tie_next_notes
        else:
            return []

    # def iter_voice_prev(self):
    #     """TODO

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     type
    #         Description of return value
    #     """

    #     for n in self.start.iter_prev(GenericNote, include_subclasses=True):
    #         if n.voice == n.voice:
    #             yield n

    # def iter_voice_next(self):
    #     """TODO

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     type
    #         Description of return value
    #     """

    #     for n in self.start.iter_next(GenericNote, include_subclasses=True):
    #         if n.voice == n.voice:
    #             yield n

    def iter_chord(self, same_duration=True, same_voice=True):
        """Iterate over notes with coinciding start times.

        Parameters
        ----------
        same_duration : bool, optional
            When True limit the iteration to notes that have the same
            duration as the current note. Defaults to True.
        same_voice : bool, optional
            When True limit the iteration to notes that have the same
            voice as the current note. Defaults to True.

        Yields
        ------
        GenericNote

        """

        for n in self.start.iter_starting(GenericNote, include_subclasses=True):
            if ((not same_voice) or n.voice == self.voice) and (
                (not same_duration) or (n.duration == self.duration)
            ):
                yield n

    def __str__(self):
        s = "{} id={} voice={} staff={} type={}".format(
            super().__str__(),
            self.id,
            self.voice,
            self.staff,
            format_symbolic_duration(self.symbolic_duration),
        )
        if self.articulations:
            s += " articulations=({})".format(", ".join(self.articulations))
        if self.tie_prev or self.tie_next:
            all_tied = self.tie_prev_notes + [self] + self.tie_next_notes
            tied_id = "+".join(n.id or "None" for n in all_tied)
            return s + " tie_group={}".format(tied_id)
        else:
            return s


class Note(GenericNote):
    """Subclass of GenericNote representing pitched notes.

    Parameters
    ----------
    step : {'C', 'D', 'E', 'F', 'G', 'A', 'B'}
        The note name of the pitch (in upper case). If a lower case
        note name is given, it will be converted to upper case.
    octave : int
        An integer representing the octave of the pitch
    alter : int, optional
        An integer (or None) representing the alteration of the pitch as
        follows:

        -2
            double flat
        -1
            flat
        0 or None
            unaltered
        1
            sharp
        2
            double sharp

        Defaults to None.

    """

    def __init__(self, step, octave, alter=None, **kwargs):
        super().__init__(**kwargs)
        self.step = step.upper()
        self.octave = octave
        self.alter = alter
        self.beam = None

    def assign_beam(self, beam):
        self.beam = beam
        if self.beam is not None:
            self.beam.append(self)

    def __str__(self):
        return " ".join(
            (
                super().__str__(),
                "pitch={}{}{}".format(self.step, self.alter_sign, self.octave),
            )
        )

    @property
    def midi_pitch(self):
        """The midi pitch value of the note (MIDI note number). C4
        (middle C, in german: c') is note number 60.

        Returns
        -------
        integer
            The note's pitch as MIDI note number.

        """
        return pitch_spelling_to_midi_pitch(
            step=self.step, octave=self.octave, alter=self.alter
        )

    @property
    def alter_sign(self):
        """The alteration of the note

        Returns
        -------
        str

        """
        return ALTER_SIGNS[self.alter]


class UnpitchedNote(GenericNote):
    """Subclass of GenericNote representing unpitched notes.

    Parameters
    ----------
        Parameters
    ----------
    step : {'C', 'D', 'E', 'F', 'G', 'A', 'B'}
        The note name of the pitch (in upper case). If a lower case
        note name is given, it will be converted to upper case.
    octave : int
        An integer representing the octave of the pitch
    notehead : string
        A string representing the notehead.
        Defaults to None
    noteheadstyle : bool
        A boolean indicating whether the notehead is filled.
        Defaults to true

    """

    def __init__(
        self, step, octave, beam=None, notehead=None, noteheadstyle=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.step = step.upper()
        self.octave = octave
        self.beam = beam
        self.notehead = notehead
        self.noteheadstyle = noteheadstyle

        if self.beam is not None:
            self.beam.append(self)

    def __str__(self):
        return " ".join(
            (
                super().__str__(),
                "pitch={}{}{}".format(self.step, "", self.octave),
            )
        )

    @property
    def midi_pitch(self):
        """The midi pitch value of the note (MIDI note number).

        Returns
        -------
        integer
            The note's position as MIDI note number.

        """
        return pitch_spelling_to_midi_pitch(step=self.step, octave=self.octave, alter=0)


class Rest(GenericNote):
    """A subclass of GenericNote representing a rest."""

    def __init__(self, hidden=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden = hidden


class Beam(TimedObject):
    """Represent beams (for MEI)"""

    def __init__(self, id=None):
        super().__init__()
        self.id = id
        self.notes = []

    def append(self, note):
        note.beam = self
        self.notes.append(note)
        self.update_time()

    def update_time(self):
        start_idx = np.argmin([n.start.t for n in self.notes])
        end_idx = np.argmax([n.end.t for n in self.notes])

        self.start = self.notes[start_idx].start
        self.end = self.notes[end_idx].end


class GraceNote(Note):
    """A subclass of Note representing a grace note.

    Parameters
    ----------
    grace_type : {'grace', 'acciaccatura', 'appoggiatura'}
        The type of grace note. Use 'grace' for a unspecified grace
        note type.
    steal_proportion : float, optional
        The proportion of the previous (acciaccatura) or next
        (appoggiatura) note duration that is occupied by the grace
        note. Defaults to None.

    Attributes
    ----------
    main_note : :class:`Note`
        The (non-grace) note to which this grace note belongs.
    grace_seq_len : list
        The length of the sequence of grace notes to which this grace
        note belongs.

    """

    def __init__(self, grace_type, *args, steal_proportion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grace_type = grace_type
        self.steal_proportion = steal_proportion
        self.grace_next = None
        self.grace_prev = None
        self._ref_attrs.extend(["grace_next", "grace_prev"])

    @property
    def main_note(self):
        n = self.grace_next
        while isinstance(n, GraceNote):
            n = n.grace_next
        return n

    @property
    def grace_seq_len(self):
        return (
            sum(1 for _ in self.iter_grace_seq(backwards=True))
            + sum(1 for _ in self.iter_grace_seq())
            - 1
        )  # subtract one because self is counted twice

    @property
    def last_grace_note_in_seq(self):
        n = self
        while isinstance(n.grace_next, GraceNote):
            n = n.grace_next
        return n

    def iter_grace_seq(self, backwards=False):
        """Iterate over this and all subsequent/preceding grace notes,
        excluding the main note.

        Parameters
        ----------
        backwards : bool, optional
            When True, iterate over preceding grace notes. Otherwise
            iterate over subsequent grace notes. Defaults to False.

        Yields
        ------
        GraceNote

        """

        yield self
        if backwards:
            n = self.grace_prev
        else:
            n = self.grace_next
        while isinstance(n, GraceNote):
            yield n
            if backwards:
                n = n.grace_prev
            else:
                n = n.grace_next

    def __str__(self):
        return f"{super().__str__()} main_note={self.main_note}"


class Page(TimedObject):
    """A page in a musical score. Its start and end times describe the
    range of musical time that is spanned by the page.

    Parameters
    ----------
    number : int, optional
        The number of the system. Defaults to 0.

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number=0):
        super().__init__()
        self.number = number

    def __str__(self):
        return f"{super().__str__()} number={self.number}"


class System(TimedObject):
    """A system in a musical score. Its start and end times describe
    the range of musical time that is spanned by the system.

    Parameters
    ----------
    number : int, optional
        The number of the system. Defaults to 0.

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number=0):
        super().__init__()
        self.number = number

    def __str__(self):
        return f"{super().__str__()} number={self.number}"


class Clef(TimedObject):
    """Clefs associate the lines of a staff to musical pitches.

    Parameters
    ----------
    staff : int, optional
        The number of the staff to which this clef belongs.
    sign : {'G', 'F', 'C', 'percussion', 'TAB', 'jianpu',  'none'}
        The sign of the clef
    line : int
        The staff line at which the sign is positioned
    octave_change : int
        The number of octaves to shift the pitches up (postive) or
        down (negative)

    Attributes
    ----------
    staff : int
        See parameters
    sign : {'G', 'F', 'C', 'percussion', 'TAB', 'jianpu',  'none'}
        See parameters
    line : int
        See parameters
    octave_change : int
        See parameters

    """

    def __init__(self, staff, sign, line, octave_change):
        super().__init__()
        self.staff = staff
        self.sign = sign
        self.line = line
        self.octave_change = octave_change

    def __str__(self):
        return (
            f"{super().__str__()} sign={self.sign} "
            f"line={self.line} number={self.staff}"
        )


class Slur(TimedObject):
    """Slurs indicate musical grouping across notes.

    Parameters
    ----------
    start_note : :class:`Note`, optional
        The note at which this slur starts. Defaults to None.
    end_note : :class:`Note`, optional
        The note at which this slur ends. Defaults to None.

    Attributes
    ----------
    start_note : :class:`Note` or None
        See parameters
    end_note : :class:`Note` or None
        See parameters


    """

    def __init__(self, start_note=None, end_note=None):
        super().__init__()
        self._start_note = None
        self._end_note = None
        self.start_note = start_note
        self.end_note = end_note
        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(["start_note", "end_note"])

    @property
    def start_note(self):
        return self._start_note

    @start_note.setter
    def start_note(self, note):
        # make sure we received a note
        if note:
            if self.start:
                #  remove the slur from the current start time
                self.start.remove_starting_object(self)
            note.slur_starts.append(self)
        self._start_note = note

    @property
    def end_note(self):
        return self._end_note

    @end_note.setter
    def end_note(self, note):
        # make sure we received a note
        if note:
            if self.end:
                #  remove the slur from the current end time
                self.end.remove_ending_object(self)
            if note.end:
                # add it to the end time of the new end note
                note.end.add_ending_object(self)
            note.slur_stops.append(self)
        self._end_note = note

    def __str__(self):
        start = "" if self.start_note is None else "start={}".format(self.start_note.id)
        end = "" if self.end_note is None else "end={}".format(self.end_note.id)
        return " ".join((super().__str__(), start, end)).strip()


class Tuplet(TimedObject):
    """Tuplets indicate musical grouping across notes.

    Parameters
    ----------
    start_note : :class:`Note`, optional
        The note at which this tuplet starts. Defaults to None.
    end_note : :class:`Note`, optional
        The note at which this tuplet ends. Defaults to None.

    Attributes
    ----------
    start_note : :class:`Note` or None
        See parameters
    end_note : :class:`Note` or None
        See parameters

    """

    def __init__(
        self,
        start_note=None,
        end_note=None,
        actual_notes=None,
        normal_notes=None,
        actual_type=None,
        normal_type=None,
    ):
        super().__init__()
        self._start_note = None
        self._end_note = None
        self.start_note = start_note
        self.end_note = end_note
        self.actual_notes = actual_notes
        self.normal_notes = normal_notes
        self.actual_type = actual_type
        self.normal_type = normal_type
        # maintain a list of attributes to update when cloning this instance
        self._ref_attrs.extend(["start_note", "end_note"])

    @property
    def start_note(self):
        return self._start_note

    @start_note.setter
    def start_note(self, note):
        # make sure we received a note
        if note:
            if note.start:
                #  remove the tuplet from the current start time
                if self.start_note and self.start_note.start:
                    self.start_note.start.remove_starting_object(self)
            # else:
            #     warnings.warn('Note has no start time')
            note.tuplet_starts.append(self)
        self._start_note = note

    @property
    def end_note(self):
        return self._end_note

    @end_note.setter
    def end_note(self, note):
        # make sure we received a note
        if note:
            if note.end:
                if self.end_note and self.end_note.end:
                    #  remove the tuplet from the currentend time
                    self.end_note.end.remove_ending_object(self)
            # else:
            #     warnings.warn('Note has no end time')
            note.tuplet_stops.append(self)
        self._end_note = note

    @property
    def duration_multiplier(self) -> Fraction:
        """Ratio by which the durations are scaled with this tuplet, as a python Fraction object.
        This property is similar to `.tupletMultiplier` in music21:
        https://www.music21.org/music21docs/moduleReference/moduleDuration.html#music21.duration.Tuplet.tupletMultiplier

        For example, in a triplet of eighth notes, each eighth note would have a duration of
        duration_multiplier * normal_eighth_duration = 2/3 * normal_eighth_duration
        """
        if self.actual_type == self.normal_type:
            return Fraction(self.normal_notes, self.actual_notes)
        else:
            # In that case, we need to convert the normal_type into the actual_type, therefore
            # adapting normal_notes
            actual_dur = Fraction(LABEL_DURS[self.actual_type])
            normal_dur = Fraction(LABEL_DURS[self.normal_type])
            return (
                Fraction(self.normal_notes, self.actual_notes) * normal_dur / actual_dur
            )

    def __str__(self):
        n_actual = (
            ""
            if self.actual_notes is None
            else "actual_notes={}".format(self.actual_notes)
        )
        n_normal = (
            ""
            if self.normal_notes is None
            else "normal_notes={}".format(self.normal_notes)
        )
        t_actual = (
            ""
            if self.actual_type is None
            else "actual_type={}".format(self.actual_type)
        )
        t_normal = (
            ""
            if self.normal_type is None
            else "normal_type={}".format(self.normal_type)
        )
        start = "" if self.start_note is None else "start={}".format(self.start_note.id)
        end = "" if self.end_note is None else "end={}".format(self.end_note.id)
        return " ".join(
            (super().__str__(), start, end, n_actual, n_normal, t_actual, t_normal)
        ).strip()


class Repeat(TimedObject):
    """Repeats represent a repeated section in the score, designated
    by its start and end times.

    """

    def __init__(self):
        super().__init__()


class DaCapo(TimedObject):
    """A Da Capo sign."""


class Fine(TimedObject):
    """A Fine sign."""


class DalSegno(TimedObject):
    """A Dal Segno sign."""


class Segno(TimedObject):
    """A Segno sign."""


class ToCoda(TimedObject):
    """A To Coda sign."""


class Coda(TimedObject):
    """A Coda sign."""


class Fermata(TimedObject):
    """A Fermata sign.

    Parameters
    ----------
    ref : :class:`TimedObject` or None, optional
        An object to which this fermata applies. In practice this is a
        Note or a Barline. Defaults to None.

    Attributes
    ----------
    ref : :class:`TimedObject` or None
        See parameters

    """

    def __init__(self, ref=None):
        super().__init__()
        # ref(erent) can be a note or a barline
        self.ref = ref

    def __str__(self):
        return f"{super().__str__()} ref={self.ref}"


class Ending(TimedObject):
    """Class that represents one part of a 1---2--- type ending of a
    musical passage (a.k.a Volta brackets).

    Parameters
    ----------
    number : int
        The number associated to this ending

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number):
        super().__init__()
        self.number = number


class Barline(TimedObject):
    """Class that represents the style of a barline"""

    def __init__(self, style):
        super().__init__()
        self.style = style


class Measure(TimedObject):
    """A measure

    Parameters
    ----------
    number : int
        The running count independent of measure regularity/ volta endings, continuously counting up all measures in a musicxml score file and always starting from one.
    name : string, optional
        The ID of the measure in a given musicxml score file. Can be a non-number in case of volta endings, irregular measures (i.e., pickup measures in the middle of the piece). Defaults to None

    Attributes
    ----------
    number : int
        See parameters
    name : str
        See parameters

    """

    def __init__(self, number=None, name=None):
        super().__init__()
        self.number = number
        self.name = name

    def __str__(self):
        return f"{super().__str__()} number={self.number} name={self.name}"

    @property
    def page(self):
        """The page number on which this measure appears, or None if
        there is no associated page.

        Returns
        -------
        int or None

        """
        page = next(self.start.iter_prev(Page, eq=True), None)
        if page:
            return page.number
        else:
            return None

    @property
    def system(self):
        """The system number in which this measure appears, or None if
        there is no associated system.

        Returns
        -------
        int or None

        """
        system = next(self.start.iter_prev(System, eq=True), None)
        if system:
            return system.number
        else:
            return None

    # TODO: add `incomplete` or `anacrusis` property


class TimeSignature(TimedObject):
    """A time signature.

    Parameters
    ----------
    beats : int
        The number of beats in a measure (the numerator).
    beat_type : int
        The note type that defines the beat unit (the denominator).
        (4 for quarter notes, 2 for half notes, etc.)
    musical_beats : int
        The number of beats according to musicologial standards
        (2 if beats is 2 or 6; 3 if beats is 3 or 9; 4 if beats is 4 or 12;
        else beats)

    Attributes
    ----------
    beats : int
        See parameters
    beat_type : int
        See parameters
    musical_beat : int
        See parameters

    """

    def __init__(self, beats, beat_type):
        super().__init__()
        self.beats = beats
        self.beat_type = beat_type
        self.musical_beats = (  # if a value is provided, otherwise default to beats
            MUSICAL_BEATS[self.beats]
            if self.beats in MUSICAL_BEATS.keys()
            else self.beats
        )

    def __str__(self):
        return f"{super().__str__()} {self.beats}/{self.beat_type}"


class Tempo(TimedObject):
    """A tempo indication.

    Parameters
    ----------
    bpm : number
        The tempo indicated in rate per minute
    unit : str or None, optional
        The unit to which the specified rate correspnds. This is a
        string that expreses a duration category, such as "q" for
        quarter "h." for dotted half, and so on. When None, the unit
        is assumed to be quarters. Defaults to None.

    Attributes
    ----------
    bpm : number
        See parameters
    unit : str or None
        See parameters

    """

    def __init__(self, bpm, unit=None):
        super().__init__()
        self.bpm = bpm
        self.unit = unit

    @property
    def microseconds_per_quarter(self):
        """The number of microseconds per quarter under this tempo.

        This is useful for MIDI representations.

        Returns
        -------
        int

        """
        return int(
            np.round(60 * (10**6 / to_quarter_tempo(self.unit or "q", self.bpm)))
        )

    def __str__(self):
        if self.unit:
            return f"{super().__str__()} {self.unit}={self.bpm}"
        else:
            return f"{super().__str__()} bpm={self.bpm}"


class Staff(TimedObject):
    """A staff.

    Parameters
    ----------
    number : int
        The staff number
    lines : int, optional (default: 5)

    Attributes
    ----------
    number : int
        See parameters

    """

    def __init__(self, number, lines=5):
        super().__init__()
        self.number = number
        self.lines = lines

    def __str__(self):
        return f"{super().__str__()} number={self.number} lines={self.lines}"


class KeySignature(TimedObject):
    """Key signature.

    Parameters
    ----------
    fifths : number
        Number of sharps (positive) or flats (negative)
    mode : str
        Mode of the key, either 'major' or 'minor'

    Attributes
    ----------
    fifths : number
        See parameters
    mode : str
        See parameters

    """

    def __init__(self, fifths, mode):
        super().__init__()
        self.fifths = fifths
        self.mode = mode

    @property
    def name(self):
        """The key signature name, where the root is uppercase, and an
        trailing 'm' indicates minor modes (e.g. 'Am', 'G#').

        Returns
        -------
        str
            The key signature name

        """
        return fifths_mode_to_key_name(self.fifths, self.mode)

    def __str__(self):
        return (
            f"{super().__str__()} fifths={self.fifths}, mode={self.mode} ({self.name})"
        )


class Transposition(TimedObject):
    """Represents a <transpose> tag that tells how to change all
    (following) pitches of that part to put it to concert pitch (i.e.
    sounding pitch).

    Parameters
    ----------
    diatonic : int
        TODO
    chromatic : int
        The number of semi-tone steps to add or subtract to the pitch
        to get to the (sounding) concert pitch.

    Attributes
    ----------
    diatonic : int
        See parameters
    chromatic : int
        See parameters

    """

    def __init__(self, diatonic, chromatic):
        super().__init__()
        self.diatonic = diatonic
        self.chromatic = chromatic

    def __str__(self):
        return (
            f"{super().__str__()} diatonic={self.diatonic}, chromatic={self.chromatic}"
        )


class Words(TimedObject):
    """A textual element in the score.

    Parameters
    ----------
    text : str
        The text
    staff : int or None, optional
        The staff to which the text is associated. Defaults to None

    Attributes
    ----------
    text : str
        See parameters
    staff : int or None, optional
        See parameters

    """

    def __init__(self, text, staff=None):
        super().__init__()
        self.text = text
        self.staff = staff

    def __str__(self):
        return f'{super().__str__()} "{self.text}"'


class OctaveShiftDirection(TimedObject):
    """An octave shift direction.

    Parameters
    ----------

    """

    def __init__(self, shift_type, shift_size=8, staff=None):
        super().__init__()
        self.shift_type = shift_type
        self.shift_size = shift_size
        self.staff = staff
        self.applied = False

    def __str__(self):
        return f'{super().__str__()} "{self.shift_type}"'


class Cadence(TimedObject):
    """A cadence element in the score usually for Cadences."""

    def __init__(self, text, local_key=None):
        super().__init__()
        self.text = text
        self._filter_cadence_type()
        self.local_key = local_key

    def _filter_cadence_type(self):
        """Cadence should be one of PAC, IAC, HC, DC, EC, PC, or None"""
        # capitalize text
        self.text = self.text.upper()
        # Filter alphabet characters only.
        self.text = re.findall(r"[A-Z]+", self.text)[0]
        self.text = "IAC" if "IAC" in self.text else self.text
        if self.text not in ["PAC", "IAC", "HC", "DC", "EC", "PC"]:
            warnings.warn(f"Cadence type {self.text} not found. Setting to None")
            self.text = None

    def __str__(self):
        return f'{super().__str__()} "{self.text}"'


class Harmony(TimedObject):
    """A harmony element in the score not currently used.

    Parameters
    ----------
    text : str
        The harmony text

    Attributes
    ----------
    text : str
        See parameters
    """

    def __init__(self, text):
        super().__init__()
        self.text = text
        # assert issubclass(note, GenericNote)

    def __str__(self):
        return f'{super().__str__()} "{self.text}"'


class RomanNumeral(Harmony):
    """A harmony element in the score usually for Roman Numerals.

    Parameters
    ----------
    text : str
        The harmony text

    Attributes
    ----------
    text : str
        See parameters
    """

    def __init__(
        self,
        text,
        inversion=None,
        local_key=None,
        primary_degree=None,
        secondary_degree=None,
        quality=None,
    ):
        super().__init__(text)
        self.text = text
        self.accepted_qualities = (
            "7",
            "aug",
            "aug6",
            "aug7",
            "dim",
            "dim7",
            "hdim7",
            "maj",
            "maj7",
            "min",
            "min7",
        )
        # The key of an inversion is text from RN string, and the value is a tuple (has_seven,inversion)
        self.accepted_inversions = {
            "2": (3, True),
            "43": (2, True),
            "64": (2, False),
            "6": (1, False),
            "65": (1, True),
            "7": (0, True),
        }
        self.has_seven = "7" in text
        self.inversion = (
            inversion if inversion is not None else self._process_inversion()
        )
        self.local_key = (
            local_key if local_key is not None else self._process_local_key()
        )
        self.primary_degree = (
            primary_degree
            if primary_degree is not None
            else self._process_primary_degree()
        )
        self.secondary_degree = (
            secondary_degree
            if secondary_degree is not None
            else self._process_secondary_degree()
        )
        self.quality = (
            quality
            if quality is not None and quality in self.accepted_qualities
            else self._process_quality()
        )
        # only process the root note if the roman numeral is valid
        if (
            self.local_key
            and self.primary_degree
            and self.secondary_degree
            and self.quality
            and self.inversion
        ):
            self.root = self.find_root_note()
            self.bass_note = self.find_bass_note()

    def _process_inversion(self):
        """Find the inversion of the roman numeral from the text"""
        # The inversion should be right after the roman numeral.
        # If there is no inversion, return 0
        numeric_indications_in_text = re.findall(r"\d+", self.text)
        if len(numeric_indications_in_text) > 0:
            inversion, has_seven = self.accepted_inversions.get(
                numeric_indications_in_text[0], (0, False)
            )
            self.has_seven = has_seven
            return inversion
        return 0

    def _process_local_key(self):
        """Find the local key of the roman numeral from the text"""
        # The local key should be before the roman numeral.
        # If there is no local key, return None
        local_key = self.text.split(":")
        if len(local_key) > 1:
            return local_key[0]
        return None

    def _process_primary_degree(self):
        """Find the primary degree of the roman numeral from the text

        The primary degree should be a roman numeral between 1 and 7.
        """
        # The primary degree should be a roman numeral between 1 and 7.
        # If there is no primary degree, return None
        # Remove any key information
        roman_text = self.text.split(":")[-1]
        roman_text = roman_text.split(".")[-1] if "." in roman_text else roman_text
        primary_degree = re.search(r"[a-zA-Z+]+", roman_text)
        if primary_degree:
            prim_d = primary_degree.group(0)
            # if the primary degree is not in accepted values, return the closest one
            if prim_d in ACCEPTED_ROMANS:
                return prim_d
            else:
                matches = difflib.get_close_matches(
                    prim_d, ACCEPTED_ROMANS, n=1, cutoff=0.5
                )
                if matches:
                    return matches[0]
        return None

    def _process_secondary_degree(self):
        """Find the secondary degree of the roman numeral from the text

        The secondary degree should be a roman numeral between 1 and 7.
        If it is not specified in the text, return I (the tonic) when the primary degree is not none.
        """
        # The secondary degree should be a roman numeral between 1 and 7.
        # If it is not specified in the text, return I (the tonic) when the primary degree is not none.
        roman_text = self.text.split(":")[-1]
        split_pr_sec = roman_text.split("/")
        if len(split_pr_sec) > 1:
            secondary_degree = re.search(r"[a-zA-Z+]+", split_pr_sec[-1])
            return secondary_degree.group(0)
        elif self.primary_degree is not None and self.local_key is not None:
            secondary_degree = "I" if self.local_key.isupper() else "i"
            return secondary_degree
        return None

    def _process_quality(self):
        """Find the quality of the roman numeral from the text

        Accepted quality values are 7, aug, aug6, aug7, dim, dim7, hdim7, maj, maj7, min, min7.
        This format follows the standards from the latest version of the AugmentedNet model.
        Found out more here: github.com/napulen/AugmentedNet
        """
        # The quality should be M, m, +, o, or None.
        aug_cond = "aug" in self.text.lower() or "+" in self.text.lower()
        minor_cond = (
            self.primary_degree.islower() if self.primary_degree is not None else False
        )
        major_cond = (
            self.primary_degree.isupper() if self.primary_degree is not None else False
        )
        dim_cond = "dim" in self.text or "o" in self.text
        aug6_cond = (
            "ger" in self.text.lower()
            or "it" in self.text.lower()
            or "fr" in self.text.lower()
        )
        hdim_cond = "0" in self.text or "%" in self.text or "ø" in self.text
        if aug6_cond:
            quality = "aug6"
        elif "maj7" in self.text.lower():
            quality = "maj7"
        elif dim_cond and self.has_seven:
            quality = "dim7"
        elif dim_cond:
            quality = "dim"
        elif aug_cond and self.has_seven:
            quality = "aug7"
        elif aug_cond:
            quality = "aug"
        elif hdim_cond:
            quality = "hdim7"
        elif minor_cond and self.has_seven:
            quality = "min7"
        elif minor_cond:
            quality = "min"
        elif major_cond and self.has_seven:
            quality = "7"
        elif major_cond:
            quality = "maj"
        else:
            warnings.warn(
                f"Quality for {self.text} was not found, could be a special case. Setting to None."
            )
            quality = None
        return quality

    def find_root_note(self):
        """
        Find the root note of a chord.

        Returns
        -------
        number: int
            The number of the chord.
        """
        # Corrected step after degree2
        key_step = re.search(r"[a-gA-G]", self.local_key).group(0)
        key_alter = (
            re.search(r"[#b]", self.local_key).group(0)
            if re.search(r"[#b]", self.local_key)
            else ""
        )
        key_alter = ALT_TO_INT[key_alter]
        try:
            interval = (
                Roman2Interval_Min[self.secondary_degree]
                if self.local_key.islower()
                else Roman2Interval_Maj[self.secondary_degree]
            )
            step, alter = transpose_note(key_step, key_alter, interval)
        except KeyError:
            loc_k = self.secondary_degree
            glob_k = self.local_key
            step, alter = process_local_key(loc_k, glob_k, return_step_alter=True)
        # Corrected step after degree1
        # TODO add support for diminished and augmented chords
        try:
            interval = (
                Roman2Interval_Min[self.primary_degree]
                if self.secondary_degree.islower()
                else Roman2Interval_Maj[self.primary_degree]
            )
            step, alter = transpose_note(step, alter, interval)
            root = step + INT_TO_ALT[alter]
        except KeyError:
            loc_k = self.primary_degree
            glob_k = step.lower() if self.secondary_degree.islower() else step.upper()
            root = step + INT_TO_ALT[alter]
            root = process_local_key(loc_k, glob_k)

        return root

    def find_bass_note(self):
        # TODO add support for diminished and augmented chords
        step = re.search(r"[a-gA-G]", self.root).group(0)
        alter = re.search(r"[#b]", self.root)
        alter = ALT_TO_INT[alter.group(0)] if alter else 0

        if self.inversion == 1:
            if self.primary_degree.islower():
                step, alter = transpose_note(step, alter, Interval(3, "m"))
            else:
                step, alter = transpose_note(step, alter, Interval(3, "M"))
        elif self.inversion == 2:
            step, alter = transpose_note(step, alter, Interval(5, "P"))
        elif self.inversion == 3:
            step, alter = transpose_note(step, alter, Interval(7, "m"))

        bass_note_name = step + INT_TO_ALT[alter]
        return bass_note_name

    def __str__(self):
        return f'{super().__str__()} "{self.text}"'


class Cadence(TimedObject):
    """A cadence element in the score usually for Cadences."""

    def __init__(self, text, local_key=None):
        super().__init__()
        self.text = text
        self._filter_cadence_type()
        self.local_key = local_key

    def _filter_cadence_type(self):
        """Cadence should be one of PAC, IAC, HC, DC, EC, PC, or None"""
        # capitalize text
        self.text = self.text.upper()
        # Filter alphabet characters only.
        self.text = re.findall(r"[A-Z]+", self.text)[0]
        self.text = "IAC" if "IAC" in self.text else self.text
        if self.text not in ["PAC", "IAC", "HC", "DC", "EC", "PC"]:
            warnings.warn(f"Cadence type {self.text} not found. Setting to None")
            self.text = None

    def __str__(self):
        return f'{super().__str__()} "{self.text}"'


class Phrase(TimedObject):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"{super().__str__()}"


class ChordSymbol(Harmony):
    """A harmony element in the score usually for Chord Symbols."""

    def __init__(self, root, kind, bass=None):
        super().__init__(
            text=root + (f"/{kind}" if kind else "") + (f"/{bass}" if bass else "")
        )
        self.kind = kind
        self.root = root
        self.bass = bass

    def __str__(self):
        return f'{super().__str__()} "{self.root + self.kind}"'


class Interval(object):
    """
    An interval element usually used for transpositions

    Parameters
    ----------
    number : int
        The interval number (e.g. 1, 2, 3, 4, 5, 6, 7, ...)
    quality : str
        The interval quality (e.g. M, m, P, A, d, dd, AA)
    direction : str
        The interval direction (e.g. up, down)
    """

    def __init__(self, number, quality, direction="up"):
        self.number = number
        self.quality = quality
        self.direction = direction
        self.validate()

    def validate(self):
        number = self.number % 7
        number = 7 if number == 0 else number
        assert (
            self.quality + str(number) in INTERVALCLASSES
        ), f"Interval {number}{self.quality} not found"
        assert self.direction in [
            "up",
            "down",
        ], f"Interval direction {self.direction} not found"

    @property
    def semitones(self):
        return INTERVAL_TO_SEMITONES[self.quality + str(self.number)]

    def change_quality(self, num):
        """
        Change the quality of the interval by a given number of semitones.

        The Interval Number is not changed, only the quality.

        Examples:
        - M3 -> m3, num=-1
        - M3 -> A3, num=1
        - A4 -> d4, num=-2

        Parameters
        ----------
        num: int
            The number of semitones to change the quality by.

        Returns
        -------
        Interval
            The interval with the new quality, but the same number and direction.
        """
        change_direction_c = ["dd", "d", "P", "A", "AA"]
        change_direction_d = ["dd", "d", "m", "M", "A", "AA"]

        prev_quality = self.quality
        if num == 0:
            pass
        else:
            change_dir = (
                change_direction_c
                if self.number in [1, 4, 5, 8]
                else change_direction_d
            )
            cur_index = change_dir.index(prev_quality)
            new_index = cur_index + num
            if new_index >= len(change_dir) or new_index < 0:
                raise ValueError("Interval quality cannot be changed to that extent")
            self.quality = change_dir[new_index]
        return self

    def __str__(self):
        return f'{super().__str__()} "{self.number}{self.quality}"'


class Direction(TimedObject):
    """Base class for performance directions in the score."""

    def __init__(self, text=None, raw_text=None, staff=None):
        super().__init__()
        self.text = text if text is not None else ""
        self.raw_text = raw_text
        self.staff = staff

    def __str__(self):
        if self.raw_text is not None:
            return f'{super().__str__()} "{self.text}" raw_text="{self.raw_text}"'
        else:
            return f'{super().__str__()} "{self.text}"'


class LoudnessDirection(Direction):
    pass


class TempoDirection(Direction):
    pass


class ArticulationDirection(Direction):
    pass


class PedalDirection(Direction):
    pass


class ConstantDirection(Direction):
    pass


class DynamicDirection(Direction):
    pass


class ImpulsiveDirection(Direction):
    pass


class ConstantLoudnessDirection(ConstantDirection, LoudnessDirection):
    pass


class ConstantTempoDirection(ConstantDirection, TempoDirection):
    pass


class ConstantArticulationDirection(ConstantDirection, ArticulationDirection):
    pass


class DynamicLoudnessDirection(DynamicDirection, LoudnessDirection):
    def __init__(self, *args, wedge=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.wedge = wedge

    def __str__(self):
        if self.wedge:
            return f"{super().__str__()} wedge"
        else:
            return super().__str__()


class DynamicTempoDirection(DynamicDirection, TempoDirection):
    pass


class IncreasingLoudnessDirection(DynamicLoudnessDirection):
    pass


class DecreasingLoudnessDirection(DynamicLoudnessDirection):
    pass


class IncreasingTempoDirection(DynamicTempoDirection):
    pass


class DecreasingTempoDirection(DynamicTempoDirection):
    pass


class ImpulsiveLoudnessDirection(ImpulsiveDirection, LoudnessDirection):
    pass


class SustainPedalDirection(PedalDirection):
    """Represents a Sustain Pedal Direction"""

    def __init__(self, line=False, *args, **kwargs):
        super().__init__("sustain_pedal", *args, **kwargs)
        self.line = line


class ResetTempoDirection(ConstantTempoDirection):
    @property
    def reference_tempo(self):
        direction = None
        for d in self.start.iter_prev(ConstantTempoDirection):
            direction = d
        return direction


class NoteTechnicalNotation(object):
    """
    This object represents technical notations that
    are part of a GenericNote object (e.g., fingering,)
    These elements depend on a note, but can have their own properties
    """

    def __init__(self, type: str, info: Optional[Any] = None) -> None:
        self.type = type
        self.info = info


class Fingering(NoteTechnicalNotation):
    """
    This object represents fingering. For now, it supports attributes
    present in MusicXML:

    https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/fingering/

    Parameters
    ----------
    fingering : Optional[int]
        Fingering information. Can be None (usually the result of incorrect parsing of fingering).

    is_substitution: bool
        Whether this fingering is a substitution in the middle of a note. Default is False

    is_alternate: bool
        Whether this fingering is an alternative fingering. Default is False

    placement: str
        Placement of the fingering (above or below a note)
    """

    def __init__(
        self,
        fingering: Optional[int],
        is_substitution: bool = False,
        placement: Optional[str] = None,
        is_alternate: bool = False,
    ) -> None:
        super().__init__(
            type="fingering",
            info=fingering,
        )
        self.fingering = fingering
        self.is_alternate = is_alternate
        self.alternative_fingering = []
        self.is_substitution = is_substitution
        self.placement = placement


class PartGroup(object):
    """Represents a grouping of several instruments, usually named,
    and expressed in the score with a group symbol such as a brace or
    a bracket. In symphonic scores, bracketed part groups usually
    group families of instruments, such as woodwinds or brass, whereas
    braces are often used to group multiple instances of the same
    instrument. See the `MusicXML documentation
    <https://usermanuals.musicxml.com/MusicXML/Content/ST-MusicXML-
    group-symbol-value.htm>`_ for further information.

    Parameters
    ----------
    group_symbol : str or None, optional
        The symbol used for grouping instruments.

    Attributes
    ----------
    group_symbol : str or None

    name : str or None

    number : int

    parent : PartGroup or None

    children : list of Part or PartGroup objects

    """

    def __init__(self, group_symbol=None, group_name=None, number=None, id=None):
        self.group_symbol = group_symbol
        self.group_name = group_name
        self.number = number
        self.parent = None
        self.id = id
        self.children = []

    def _pp(self, tree):
        result = [
            '{}PartGroup: group_name="{}" group_symbol="{}"'.format(
                tree, self.group_name, self.group_symbol
            )
        ]
        tree.push()
        N = len(self.children)
        for i, child in enumerate(self.children):
            result.append("{}".format(tree).rstrip())
            if i == N - 1:
                tree.last_item()
            else:
                tree.next_item()
            result.extend(child._pp(tree))
        tree.pop()
        return result

    def pretty(self):
        """Return a pretty representation of this object.

        Returns
        -------
        str
            A pretty representation

        """
        return "\n".join(self._pp(PrettyPrintTree()))

    def note_array(self, *args, **kwargs):
        """A structured array containing pitch, onset, duration, voice
        and id for each note in each part of the PartGroup. The note
        ids in this array include the number of the part to which they
        belong.

        See Part.note_array()

        """
        return note_array_from_part_list(self.children, *args, **kwargs)

    def rest_array(self, *args, **kwargs):
        """A structured array containing pitch, onset, duration, voice
        and id for each note in each part of the PartGroup. The note
        ids in this array include the number of the part to which they
        belong.

        See Part.note_array()

        """
        return rest_array_from_part_list(self.children, *args, **kwargs)


class Score(object):
    """Main object for representing a score.

    The `Score` object is basically an iterable that provides access to all
    `Part` objects in a musical score.

    Parameters
    ----------
    id : str
        The identifier of the score. In order to be compatible with MusicXML
        the identifier should not start with a number.
    partlist : `Part`, `PartGroup` or list of `Part` or `PartGroup` instances.
        List of  `Part` or `PartGroup` objects.
    work_title: str, optional
        Work title of the score, if applicable.
    work_number: str, optional
        Work number of the score, if applicable.
    movement_title: str, optional
        Movement title of the score, if applicable.
    movement_number: str, optional
        Movement number of the score, if applicable.
    title : str, optional
        Title of the score, from <credit-words> tag
    subtitle: str, optional
        Subtitle of the score.
    composer: str, optional
        Composer of the score.
    lyricist: str, optional
        Lyricist of the score.
    copyright: str, optional.
        Copyright notice of the score.

    Attributes
    ----------
    id : str
        See parameters.
    parts : list of `Part` objects
        All `Part` objects.
    part_structure: list of `Part` or `PartGrop`
        List of all `Part` or `PartGroup` objects that specify the structure of
        the score.
    work_title: str
        See parameters.
    work_number: str
        See parameters.
    movement_title: str
        See parameters.
    movement_number: str
        See parameters.
    subtitle: str
        See parameters.
    composer: str
        See parameters.
    lyricist: str
        See parameters.
    copyright: str.
        See parameters.

    """

    id: Optional[str]
    work_title: Optional[str]
    work_number: Optional[str]
    movement_title: Optional[str]
    movement_number: Optional[str]
    title: Optional[str]
    subtitle: Optional[str]
    composer: Optional[str]
    lyricist: Optional[str]
    copyright: Optional[str]
    parts: List[Part]
    part_structure: List[Union[Part, PartGroup]]

    def __init__(
        self,
        partlist: Union[Part, PartGroup, Itertype[Union[Part, PartGroup]]],
        id: Optional[str] = None,
        work_title: Optional[str] = None,
        work_number: Optional[str] = None,
        movement_title: Optional[str] = None,
        movement_number: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        composer: Optional[str] = None,
        lyricist: Optional[str] = None,
        copyright: Optional[str] = None,
    ) -> None:
        self.id = id

        # Score Information (default from MuseScore/MusicXML)
        self.work_title = work_title
        self.work_number = work_number
        self.movement_title = movement_title
        self.movement_number = movement_number
        self.title = title
        self.subtitle = subtitle
        self.composer = composer
        self.lyricist = lyricist
        self.copyright = copyright

        # Flat list of parts
        self.parts = list(iter_parts(partlist))
        # List of Parts and PartGroups

        if isinstance(partlist, (Part, PartGroup)):
            self.part_structure = [partlist]
        elif isinstance(partlist, Iterable):
            self.part_structure = list(partlist)
        else:
            raise ValueError(
                "`partlist` should be a list, a `Part` or a `PartGrop` but"
                f" is {type(partlist)}."
            )

    def __getitem__(self, index: int) -> Part:
        """Get `Part in the score by index"""
        return self.parts[index]

    def __setitem__(self, index: int, part: Part) -> None:
        """Set `Part` in the score by index"""
        # TODO: How to update the score structure as well?
        self.parts[index] = part

    def __iter__(self) -> Iterator[Part]:
        self.iter_idx = 0
        return self

    def __next__(self) -> Part:
        if self.iter_idx == len(self.parts):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def __len__(self) -> int:
        """
        The lenght of the score is the number of part objects in `self.parts`
        """
        return len(self.parts)

    def note_array(
        self,
        unique_id_per_part=True,
        include_pitch_spelling=False,
        include_key_signature=False,
        include_time_signature=False,
        include_metrical_position=False,
        include_grace_notes=False,
        include_staff=False,
        include_divs_per_quarter=False,
        **kwargs,
    ) -> np.ndarray:
        """
        Get a note array that concatenates the note arrays of all Part/PartGroup
        objects in the score.
        """
        return note_array_from_part_list(
            part_list=self.parts,
            unique_id_per_part=unique_id_per_part,
            include_pitch_spelling=include_pitch_spelling,
            include_key_signature=include_key_signature,
            include_time_signature=include_time_signature,
            include_grace_notes=include_grace_notes,
            include_metrical_position=include_metrical_position,
            include_staff=include_staff,
            include_divs_per_quarter=include_divs_per_quarter,
            **kwargs,
        )


# Alias for typing score-like objects
ScoreLike = Union[List[Union[Part, PartGroup]], Part, PartGroup, Score]


class ScoreVariant(object):
    # non-public

    def __init__(self, part, start_time=0):
        self.t_unfold = start_time
        self.segments = []
        self.part = part

    def add_segment(self, start, end):
        self.segments.append((start, end, self.t_unfold))
        self.t_unfold += end.t - start.t

    @property
    def segment_times(self):
        """
        Return segment (start, end, offset) information for each of the segments in
        the score variant.
        """
        return [(s.t, e.t, o) for (s, e, o) in self.segments]

    def __str__(self):
        return f"{super().__str__()} {self.segment_times}"

    def clone(self):
        """
        Return a clone of the ScoreVariant
        """
        clone = ScoreVariant(self.part, self.t_unfold)
        clone.segments = self.segments[:]
        return clone

    def create_variant_part(self):
        part = Part(self.part.id, part_name=self.part.part_name)

        for start, end, offset in self.segments:
            delta = offset - start.t
            qd = self.part.quarter_durations(start.t, end.t)
            for t, quarter in qd:
                part.set_quarter_duration(t + delta, quarter)
            # After creating the new part we need to replace references to
            # objects in the old part to references in the new part
            # (e.g. t.next, t.prev, note.tie_next). For this we keep track of
            # correspondences between objects (timepoints, notes, measures,
            # etc), in o_map
            o_map = {}
            o_new = set()
            tp = start
            while tp != end:
                # make a new timepoint, corresponding to tp
                tp_new = part.get_or_add_point(tp.t + delta)
                o_gen = (o for oo in tp.starting_objects.values() for o in oo)
                for o in o_gen:
                    # special cases:

                    # don't include some TimedObjects in the unfolded part
                    if isinstance(
                        o,
                        (
                            Repeat,
                            Ending,
                            ToCoda,
                            DaCapo,
                            DalSegno,
                            Segment,
                            System,
                            Page,
                        ),
                    ):
                        continue

                    # don't repeat time sig if it hasn't changed
                    elif isinstance(o, TimeSignature):
                        prev = next(tp_new.iter_prev(TimeSignature), None)
                        if (prev is not None) and (
                            (o.beats, o.beat_type) == (prev.beats, prev.beat_type)
                        ):
                            continue
                    # don't repeat key sig if it hasn't changed
                    elif isinstance(o, KeySignature):
                        prev = next(tp_new.iter_prev(KeySignature), None)
                        if (prev is not None) and (
                            (o.fifths, o.mode) == (prev.fifths, prev.mode)
                        ):
                            continue

                    # don't repeat clef if it hasn't changed
                    elif isinstance(o, Clef):
                        prev = next(tp_new.iter_prev(Clef), None)
                        if (prev is not None) and (
                            (o.sign, o.line, o.staff)
                            == (prev.sign, prev.line, prev.staff)
                        ):
                            continue

                    # make a copy of the object
                    o_copy = copy(o)
                    # add it to the set of new objects (for which the refs will
                    # be replaced)
                    o_new.add(o_copy)
                    # keep track of the correspondence between o and o_copy
                    o_map[o] = o_copy
                    # add the start of the new object to the part
                    tp_new.add_starting_object(o_copy)
                    if o.end is not None:
                        # add the end of the object to the part
                        tp_end = part.get_or_add_point(o.end.t + delta)
                        tp_end.add_ending_object(o_copy)

                tp = tp.next
                if tp is None:
                    raise Exception(
                        "segment end not a successor of segment start, "
                        "invalid score variant"
                    )

            # special case: fermata starting at end of segment should be
            # included if it does not belong to a note, and comes at the end of
            # a measure (o.ref == 'right')
            for o in end.starting_objects[Fermata]:
                if o.ref in (None, "right"):
                    o_copy = copy(o)
                    tp_new = part.get_or_add_point(end.t + delta)
                    tp_new.add_starting_object(o_copy)

            # for each of the new objects, replace the references to the old
            # objects to their corresponding new objects
            for o in o_new:
                o.replace_refs(o_map)

        # replace prev/next references in timepoints
        for tp, tp_next in iter_current_next(part._points):
            tp.next = tp_next
            tp_next.prev = tp

        return part


def add_measures(part):
    """Add measures to a part.

    This function adds Measure objects to the part according to any
    time signatures present in the part. Any existing measures will be
    untouched, and added measures will be delimited by the existing
    measures.

    The Part object will be modified in place.

    Parameters
    ----------
    part : :class:`Part`
        Part instance

    """

    timesigs = np.array(
        [(ts.start.t, ts.beats) for ts in part.iter_all(TimeSignature)], dtype=int
    )

    if len(timesigs) == 0:
        warnings.warn("No time signatures found, not adding measures")
        return

    start = part.first_point.t
    end = part.last_point.t

    if start == end:
        return

    # make sure we cover time from the start of the timeline
    if len(timesigs) == 0 or timesigs[0, 0] > start:
        timesigs = np.vstack(([[start, 4]], timesigs))

    # in unlikely case of timesig at last point, remove it
    if timesigs[-1, 0] >= end:
        timesigs = timesigs[:-1]

    ts_start_times = timesigs[:, 0]
    beats_per_measure = timesigs[:, 1]
    ts_end_times = ts_start_times[1:]

    # make sure we cover time until the end of the timeline
    if len(ts_end_times) == 0 or ts_end_times[-1] < end:
        ts_end_times = np.r_[ts_end_times, end]

    assert len(ts_start_times) == len(ts_end_times)

    beat_map = part.beat_map
    inv_beat_map = part.inv_beat_map
    mcounter = 1

    for ts_start, ts_end, measure_dur in zip(
        ts_start_times, ts_end_times, beats_per_measure
    ):
        pos = ts_start

        while pos < ts_end:
            measure_start = pos
            measure_end_beats = min(beat_map(pos) + measure_dur, beat_map(end))
            measure_end = min(ts_end, inv_beat_map(measure_end_beats))
            # any existing measures between measure_start and measure_end
            existing_measure = next(
                part.iter_all(Measure, measure_start, measure_end), None
            )
            if existing_measure:
                if existing_measure.start.t == measure_start:
                    assert existing_measure.end.t > pos
                    pos = existing_measure.end.t
                    existing_measure.number = mcounter
                    mcounter += 1
                    continue

                else:
                    measure_end = existing_measure.start.t

            part.add(
                Measure(number=mcounter, name=str(mcounter)),
                int(measure_start),
                int(measure_end),
            )

            # if measure exists but was not at measure_start,
            # a filler measure is added with number mcounter
            if existing_measure:
                pos = existing_measure.end.t
                existing_measure.number = mcounter + 1
                mcounter = mcounter + 2
            else:
                pos = measure_end
                mcounter += 1


def remove_grace_notes(part):
    """Remove all grace notes from a timeline.

    The specified timeline object will be modified in place.

    Parameters
    ----------
    timeline : Timeline
        The timeline from which to remove the grace notes

    """
    for gn in list(part.iter_all(GraceNote)):
        part.remove(gn)


def expand_grace_notes(part):
    """Expand grace note durations in a part.

    The specified part object will be modified in place.

    Parameters
    ----------
    part : :class:`Part`
        The part on which to expand the grace notes

    """
    for gn in part.iter_all(GraceNote):
        dur = symbolic_to_numeric_duration(gn.symbolic_duration, gn.start.quarter)
        part.remove(gn, "end")
        part.add(gn, end=gn.start.t + int(np.round(dur)))


def iter_parts(partlist):
    """Iterate over all Part instances in partlist, which is a list of
    either Part or PartGroup instances. PartGroup instances contain
    one or more parts or further partgroups, and are traversed in a
    depth-first fashion.

    This function is designed to take the result of
    :func:`partitura.load_score_midi` and :func:`partitura.load_musicxml` as
    input.

    Parameters
    ----------
    partlist : Score, list, Part, or PartGroup
        A :class:`partitura.score.Part` object,
        :class:`partitura.score.PartGroup` or a list of these

    Yields
    -------
        :class:`Part` instances in `partlist`

    """

    if not isinstance(partlist, (list, tuple, set)):
        _partlist = [partlist]

    elif isinstance(partlist, Score):
        _partlist = partlist.parts

    else:
        _partlist = partlist

    for el in _partlist:
        if isinstance(el, Part):
            yield el
        else:
            for eel in iter_parts(el.children):
                yield eel


def repeats_to_start_end(repeats, first, last):
    # non-public, deprecated, unused
    """Return pairs of (start, end) TimePoints corresponding to the start and
    end times of each Repeat object. If any of the start or end attributes
    are None, replace it with the end/start of the preceding/succeeding
    Repeat, respectively, or `first` or `last`.

    Parameters
    ----------
    repeats : list
        list of Repeat instances, possibly with None-valued start/end
        attributes
    first : TimePoint
        The first TimePoint in the timeline
    last : TimePoint
        The last TimePoint in the timeline

    Returns
    -------
    list
        list of (start, end) TimePoints corresponding to each Repeat in
        `repeats`

    """
    t = first
    starts = []
    ends = []
    for repeat in repeats:
        starts.append(t if repeat.start is None else repeat.start)
        if repeat.end is not None:
            t = repeat.end

    t = last
    for repeat in reversed(repeats):
        ends.append(t if repeat.end is None else repeat.end)
        if repeat.start is not None:
            t = repeat.start
    ends.reverse()
    return list(zip(starts, ends))


def _make_tied_note_id(prev_id):
    # non-public
    """Create a derived note ID for newly created notes, by appending
    letters to the ID. If the original ID has the form X-Y (e.g.
    n1-1), then the letter will be appended to the X part.

    Parameters
    ----------
    prev_id : str
        Original note ID

    Returns
    -------
    str
        Derived note ID

    Examples
    --------
    >>> _make_tied_note_id('n0')
    'n0a'
    >>> _make_tied_note_id('n0a')
    'n0b'
    >>> _make_tied_note_id('n0-1')
    'n0a-1'

    """
    prev_id_parts = prev_id.split("-", 1)
    prev_id_p1 = prev_id_parts[0]
    if prev_id_p1:
        if ord(prev_id_p1[-1]) < ord("a") - 1:
            return "-".join(["{}a".format(prev_id_p1)] + prev_id_parts[1:])
        else:
            return "-".join(
                ["{}{}".format(prev_id_p1[:-1], chr(ord(prev_id[-1]) + 1))]
                + prev_id_parts[1:]
            )
    else:
        return None


def tie_notes(part):
    """Find notes that span measure boundaries and notes with composite
    durations, and split them adding ties.

    Parameters
    ----------
    part : :class:`Part`
        Description of `part`

    """
    # split and tie notes at measure boundaries
    for note in list(part.iter_all(Note)):
        next_measure = next(note.start.iter_next(Measure), None)
        cur_note = note
        note_end = cur_note.end

        # keep the list of stopping slurs, we need to transfer them to the last
        # tied note
        slur_stops = cur_note.slur_stops

        while next_measure and cur_note.end > next_measure.start:
            part.remove(cur_note, "end")
            cur_note.slur_stops = []
            part.add(cur_note, None, next_measure.start.t)
            cur_note.symbolic_duration = estimate_symbolic_duration(
                next_measure.start.t - cur_note.start.t, cur_note.start.quarter
            )
            sym_dur = estimate_symbolic_duration(
                note_end.t - next_measure.start.t, next_measure.start.quarter
            )
            if cur_note.id is not None:
                note_id = _make_tied_note_id(cur_note.id)
            else:
                note_id = None
            if isinstance(cur_note, UnpitchedNote):
                next_note = UnpitchedNote(
                    cur_note.step,
                    cur_note.octave,
                    id=note_id,
                    voice=cur_note.voice,
                    staff=cur_note.staff,
                    symbolic_duration=sym_dur,
                )
            else:
                next_note = Note(
                    note.step,
                    note.octave,
                    note.alter,
                    id=note_id,
                    voice=note.voice,
                    staff=note.staff,
                    symbolic_duration=sym_dur,
                )
            part.add(next_note, next_measure.start.t, note_end.t)

            cur_note.tie_next = next_note
            next_note.tie_prev = cur_note

            cur_note = next_note

            next_measure = next(cur_note.start.iter_next(Measure), None)

        if cur_note != note:
            for slur in slur_stops:
                slur.end_note = cur_note

    # then split/tie any notes that do not have a fractional/dot duration
    divs_map = part.quarter_duration_map
    max_splits = 3
    failed = 0
    succeeded = 0
    for i, note in enumerate(list(part.iter_all(Note))):
        if note.symbolic_duration is None:
            splits = find_tie_split(
                note.start.t, note.end.t, int(divs_map(note.start.t)), max_splits
            )

            if splits:
                succeeded += 1
                split_note(part, note, splits)
            else:
                failed += 1


def set_end_times(parts):
    # non-public
    """Set missing end times of musical elements in a part to equal
    the start times of the subsequent element of the same class. This
    is useful for some classes

    This function modifies the parts in place.

    Parameters
    ----------
    part : Part or PartGroup, or list of these
        Parts to be processed

    """
    for part in iter_parts(parts):
        # page, system, loudnessdirection, tempodirection
        _set_end_times(part, Page)
        _set_end_times(part, System)
        _set_end_times(part, ConstantLoudnessDirection)
        _set_end_times(part, ConstantTempoDirection)
        _set_end_times(part, ConstantArticulationDirection)


def _set_end_times(part, cls):
    acc = []
    t = None

    for obj in part.iter_all(cls, include_subclasses=True):
        if obj.start == t:
            if obj.end is None:
                acc.append(obj)

        else:
            for o in acc:
                part.add(o, end=obj.start.t)

            acc = []

            if obj.end is None:
                acc.append(obj)

            t = obj.start

    for o in acc:
        part.add(o, end=part.last_point.t)


def split_note(part, note, splits):
    # non-public

    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert len(splits) > 0
    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert note.symbolic_duration is None
    part.remove(note)
    orig_tie_next = note.tie_next
    slur_stops = note.slur_stops
    cur_note = note
    start, end, sym_dur = splits.pop(0)
    cur_note.symbolic_duration = sym_dur
    part.add(cur_note, start, end)
    while splits:
        note.slur_stops = []

        if cur_note.id is not None:
            note_id = _make_tied_note_id(cur_note.id)
        else:
            note_id = None

        next_note = Note(
            note.step,
            note.octave,
            note.alter,
            voice=note.voice,
            id=note_id,
            staff=note.staff,
        )
        cur_note.tie_next = next_note
        next_note.tie_prev = cur_note

        cur_note = next_note
        start, end, sym_dur = splits.pop(0)
        cur_note.symbolic_duration = sym_dur

        part.add(cur_note, start, end)

    cur_note.tie_next = orig_tie_next

    if cur_note != note:
        for slur in slur_stops:
            slur.end_note = cur_note


def find_tuplets(part):
    """Identify tuplets in `part` and set their symbolic durations
    explicitly.

    This function adds `actual_notes` and `normal_notes` keys to
    the symbolic duration of tuplet notes.

    This function modifies the part in place.

    Parameters
    ----------
    part : :class:`Part`
        Part instance

    """

    # quick shot at finding tuplets intended to cover some common cases.

    # are tuplets always in the same voice?

    # quite arbitrary:
    search_for_tuplets = [9, 7, 5, 3]
    # only look for x:2 tuplets
    normal_notes = 2

    candidates = []
    prev_end = None

    # 1. group consecutive notes without symbolic_duration
    for note in part.iter_all(GenericNote, include_subclasses=True):
        if note.symbolic_duration is None:
            if note.start.t == prev_end:
                candidates[-1].append(note)
            else:
                candidates.append([note])
            prev_end = note.end.t

    # 2. within each group
    for group in candidates:
        # 3. search for the predefined list of tuplets
        for actual_notes in search_for_tuplets:
            if actual_notes > len(group):
                # tuplet requires more notes than we have
                continue

            tup_start = 0

            while tup_start <= (len(group) - actual_notes):
                note_tuplet = group[tup_start : tup_start + actual_notes]
                # durs = set(n.duration for n in group[:tuplet-1])
                durs = set(n.duration for n in note_tuplet)

                if len(durs) > 1:
                    # notes have different durations (possibly valid but not
                    # supported here)
                    # continue
                    tup_start += 1
                else:
                    start = note_tuplet[0].start.t
                    end = note_tuplet[-1].end.t
                    total_dur = end - start

                    # total duration of tuplet notes must be integer-divisble by
                    # normal_notes
                    if total_dur % normal_notes > 0:
                        tup_start += 1
                    else:
                        # estimate duration type
                        dur_type = estimate_symbolic_duration(
                            total_dur // normal_notes, note_tuplet[0].start.quarter
                        )

                        if dur_type and dur_type.get("dots", 0) == 0:
                            # recognized duration without dots
                            dur_type["actual_notes"] = actual_notes
                            dur_type["normal_notes"] = normal_notes
                            for note in note_tuplet:
                                note.symbolic_duration = dur_type.copy()
                            start_note = note_tuplet[0]
                            stop_note = note_tuplet[-1]
                            tuplet = Tuplet(start_note, stop_note)
                            assert (
                                start_note.start.t <= stop_note.start.t
                            ), "The start note of a Tuplet should be before the stop note"
                            part.add(tuplet, start_note.start.t, stop_note.end.t)
                            tup_start += actual_notes

                        else:
                            tup_start += 1


def sanitize_part(part, tie_tolerance=0):
    """Find and remove incomplete structures in a part such as Tuplets
    and Slurs without start or end and grace notes without a main
    note.

    This function modifies the part in place.

    Parameters
    ----------
    part : :class:`Part`
        Part instance
    tie_tolerange: int, optional
        The maximum number of divs that separates notes that are tied together.
        Ideally, it is 0, but not so nice scores happen.

    """
    remove_grace_counter = 0
    elements_to_remove = []
    for gn in part.iter_all(GraceNote):
        if gn.main_note is None:
            for no in part.iter_all(
                Note, include_subclasses=False, start=gn.start.t, end=gn.start.t + 1
            ):
                if no.voice == gn.voice:
                    gn.last_grace_note_in_seq.grace_next = no

        if gn.main_note is None:
            elements_to_remove.append(gn)
            remove_grace_counter += 1

    remove_tuplet_counter = 0
    for tp in part.iter_all(Tuplet):
        if tp.end_note is None or tp.start_note is None:
            elements_to_remove.append(tp)
            remove_tuplet_counter += 1

    remove_slur_counter = 0
    for sl in part.iter_all(Slur):
        if sl.end_note is None or sl.start_note is None:
            elements_to_remove.append(sl)
            remove_slur_counter += 1

    for el in elements_to_remove:
        part.remove(el)

    remove_tie_counter = 0
    for n in part.notes_tied:
        if n.tie_next != None:
            d = n.duration_tied
            s = n.start.t
            e = n.end_tied.t
            if abs((e - s) - d) > tie_tolerance:
                remove_tie_counter += 1
                all_tied = n.tie_prev_notes + [n] + n.tie_next_notes
                for tn in all_tied:
                    tn.tie_next = None
                    tn.tie_prev = None

    warnings.warn(
        "part_sanitize removed {} incomplete tuplets, "
        "{} incomplete slurs, {} incomplete grace, "
        "and {} wrong ties."
        "notes".format(
            remove_tuplet_counter,
            remove_slur_counter,
            remove_grace_counter,
            remove_tie_counter,
        ),
        stacklevel=2,
    )


def assign_note_ids(parts, keep=False):
    """
    Assigns new note IDs mainly used for loaders.

    parts : list or score.PartGroup or score.Part
        Some Partitura parts
    keep : bool
        Keep or given note IDs or assign new ones.
    """
    if keep:
        # Keep existing note id's
        for p, part in enumerate(iter_parts(parts)):
            for ni, n in enumerate(part.iter_all(GenericNote, include_subclasses=True)):
                if isinstance(n, Rest):
                    n.id = "p{0}r{1}".format(p, ni) if n.id is None else n.id
                else:
                    n.id = "p{0}n{1}".format(p, ni) if n.id is None else n.id

    else:
        # assign note ids to ensure uniqueness across all parts, discarding any
        # existing note ids
        ni = 0
        ri = 0
        for part in iter_parts(parts):
            for n in part.iter_all(GenericNote, include_subclasses=True):
                if isinstance(n, Rest):
                    n.id = "r{}".format(ri)
                    ri += 1
                else:
                    n.id = "n{}".format(ni)
                    ni += 1


class Segment(TimedObject):
    """
    Class that represents any segment between two navigation markers such as repetitions,
    Volta brackets, or capo/fine/coda/segno directions.

    Parameters
    ----------
    id: string
        unique, ordererd identifier string
    to: list
        list of ids of possible destinations
    await_to:
        list of ids of possible destinations after a jump
    type : string, optional
        String for the type of the segment (either "default" or "leap_start" and "leap_end"). A "leap" tuple has the effect of forcing the fastest (shortest) repetition unfolding after this segment, as is commonly expected after capo/fine/coda/segno directions.
    info: string, optional
        String to describe the segment, used only for printing (pretty_segments)

    """

    def __init__(self, id, to, await_to, force_seq=False, type="default", info=""):
        self.id = id
        self.to = to
        self.await_to = await_to
        self.force_full_sequence = force_seq
        self.type = type
        self.info = info


def add_segments(part, force_new=False):
    """
    Add segment objects to a part based on repetition and capo/fine/coda/segno directions.

    Parameters
    ----------
    part: part
        A score part
    force_new: all_repeats : bool, optional
        Flag to delete and add segments from scratch,
        used for creating new unfoldings after segment attributes have been changed.
        defaults to false.
    """
    existing_segments = [seg for seg in part.iter_all(Segment)]
    if len(existing_segments) > 0:
        if force_new:
            for seg in existing_segments:
                part.remove(seg)
        else:
            return

    boundaries = defaultdict(dict)
    destinations = defaultdict(list)

    valid_repeats = [
        r for r in part.iter_all(Repeat) if r.start is not None and r.end is not None
    ]
    valid_endings = [
        r for r in part.iter_all(Ending) if r.start is not None and r.end is not None
    ]

    for r in valid_repeats:
        boundaries[r.start.t]["repeat_start"] = r
        boundaries[r.end.t]["repeat_end"] = r
    for v in valid_endings:
        boundaries[v.start.t]["volta_start"] = v
        boundaries[v.end.t]["volta_end"] = v
    for c in part.iter_all(Coda):
        boundaries[c.start.t]["coda"] = c
        destinations["coda"].append(c.start.t)
    for c in part.iter_all(ToCoda):
        boundaries[c.start.t]["tocoda"] = c
    for c in part.iter_all(DaCapo):
        boundaries[c.start.t]["dacapo"] = c
    for c in part.iter_all(Fine):
        boundaries[c.start.t]["fine"] = c
    for c in part.iter_all(Segno):
        boundaries[c.start.t]["segno"] = c
        destinations["segno"].append(c.start.t)
    for c in part.iter_all(DalSegno):
        boundaries[c.start.t]["dalsegno"] = c

    boundaries[part.last_point.t]["end"] = None
    boundaries[part.first_point.t]["start"] = None

    boundary_times = list(boundaries.keys())
    boundary_times.sort()

    # for every segment get an id, its jump destinations and properties
    init_character = 65
    segment_info = dict()
    for i, (s, e) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
        segment_info[s] = {
            "ID": chr(init_character + i),
            "start": s,
            "end": e,
            "to": [],
            "force_full_sequence": False,
            "type": "default",
            "info": list(),
            "volta_numbers": list(),
        }
    segment_info[boundary_times[-1]] = {"ID": "END"}

    current_volta_repeat_start = 0
    current_volta_end = 0
    current_volta_total_number = 0

    for ss in boundary_times[:-1]:
        se = segment_info[ss]["end"]

        # loop through the boundaries at the end of current segment
        for boundary_type in boundaries[se].keys():
            # REPEATS
            if boundary_type == "repeat_start":
                segment_info[ss]["to"].append(segment_info[se]["ID"])
            if boundary_type == "repeat_end":
                if "volta_end" not in list(boundaries[se].keys()):
                    segment_info[ss]["to"].append(segment_info[se]["ID"])
                    repeat_start = boundaries[se][boundary_type].start.t
                    segment_info[ss]["to"].append(segment_info[repeat_start]["ID"])
                segment_info[ss]["info"].append("repeat_end")

            # VOLTA BRACKETS
            if boundary_type == "volta_start":
                if "volta_end" not in list(boundaries[se].keys()):
                    current_volta_total_number = 0
                    current_volta_end = se
                    for volta_number in range(
                        10
                    ):  # maximal expected number of volta brackets 10
                        if "volta_start" in list(boundaries[current_volta_end].keys()):
                            # add the beginning to the jump destinations
                            numbers = boundaries[current_volta_end][
                                "volta_start"
                            ].number.split(",")
                            numbers = [str(int(n)) for n in numbers]
                            current_volta_total_number += len(numbers)
                            for no in numbers:
                                segment_info[ss]["to"].append(
                                    no
                                    + "_Volta_"
                                    + segment_info[current_volta_end]["ID"]
                                )
                            segment_info[current_volta_end]["info"].append(
                                "volta " + ",".join(numbers)
                            )
                            segment_info[current_volta_end]["volta_numbers"] += numbers
                            # segment_info[bracket_end]["info"].append(str(len(numbers)))
                            # update the search time to the end of the ext bracket
                            current_volta_end = boundaries[current_volta_end][
                                "volta_start"
                            ].end.t

            if boundary_type == "volta_end":
                current_volta_numbers = segment_info[ss]["volta_numbers"]
                for vn in current_volta_numbers:
                    if vn != str(current_volta_total_number):
                        # if repeating volta bracket, jump back to start
                        # check if repeat exists (might not be for 3+ volta brackets)
                        if "repeat_end" in list(boundaries[se].keys()):
                            current_volta_repeat_start = max(
                                boundaries[se]["repeat_end"].start.t,
                                current_volta_repeat_start,
                            )
                        repeat_start = current_volta_repeat_start
                        segment_info[ss]["to"].append(
                            "Z_Volta_" + segment_info[repeat_start]["ID"]
                        )

                if str(current_volta_total_number) in current_volta_numbers:
                    # else just go to the segment after the last
                    segment_info[ss]["to"].append(segment_info[current_volta_end]["ID"])

            # NAVIGATION SYMBOLS

            # Navigation1_ = destinations that should only be used after all others
            # Navigation2_ = destinations that are used *after* a jump
            if boundary_type == "coda":
                # if a coda symbol is passed just continue
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                segment_info[se]["type"] = "leap_end"
                segment_info[se]["info"].append("Coda")

            if boundary_type == "tocoda":
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                # find the coda and jump there
                coda_time = destinations["coda"][0]
                segment_info[ss]["to"].append(
                    "Navigation2_" + segment_info[coda_time]["ID"]
                )
                segment_info[ss]["type"] = "leap_start"
                segment_info[ss]["info"].append("al coda")

            if boundary_type == "segno":
                # if a segno symbol is passed just continue
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                segment_info[se]["type"] = "leap_end"
                segment_info[se]["info"].append("segno")

            if boundary_type == "dalsegno":
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                # find the segno and jump there
                segno_time = destinations["segno"][0]
                segment_info[ss]["to"].append(
                    "Navigation1_" + segment_info[segno_time]["ID"]
                )
                segment_info[ss]["to"].append("Navigation2_" + segment_info[se]["ID"])
                segment_info[ss]["type"] = "leap_start"
                segment_info[ss]["info"].append("dal segno")

            if boundary_type == "dacapo":
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                # jump to the start
                segment_info[ss]["to"].append(
                    "Navigation1_" + segment_info[part.first_point.t]["ID"]
                )
                segment_info[ss]["to"].append("Navigation2_" + segment_info[se]["ID"])
                segment_info[ss]["type"] = "leap_start"
                segment_info[ss]["info"].append("da capo")

            if boundary_type == "fine":
                segment_info[ss]["to"].append(segment_info[se]["ID"])
                # jump to the start
                segment_info[ss]["to"].append(
                    "Navigation2_" + segment_info[part.last_point.t]["ID"]
                )
                segment_info[ss]["info"].append("fine")

            # GENERIC
            if boundary_type == "end":
                segment_info[ss]["to"].append(segment_info[se]["ID"])

            # first segments is always a leap destination (da capo)
            if ss == 0:
                segment_info[ss]["type"] = "leap_end"

    # clean up and ORDER all the jump destination information
    for start_time in boundary_times[:-1]:
        destinations = segment_info[start_time]["to"]
        destinations_no_volta = [
            dest
            for dest in destinations
            if "Volta_" not in dest and "Navigation" not in dest
        ]
        destinations_volta = [dest for dest in destinations if "Volta_" in dest]
        # dal segno and da capo
        destinations_navigation1 = [
            dest[12:] for dest in destinations if "Navigation1_" in dest
        ]
        # al coda and fine
        destinations_navigation2 = [
            dest[12:] for dest in destinations if "Navigation2_" in dest
        ]

        # sort the repeats by ascending segment ID
        destinations_no_volta = list(set(destinations_no_volta))
        # make sure the "END" destination is the last
        destinations_except_await = (
            destinations_volta + destinations_no_volta + destinations_navigation1
        )
        if "END" in destinations_except_await:
            while "END" in destinations_no_volta:
                destinations_no_volta.remove("END")
            destinations_navigation1.append("END")

        # sort repeat destinations by ascending ID
        destinations_no_volta.sort()
        # sort destinations by volta number
        destinations_volta.sort()
        # keep only the segment IDs
        destinations_volta = [d[8:] for d in destinations_volta]
        # don't jump to volta brackets w/t number
        destinations_no_volta = [
            d for d in destinations_no_volta if d not in destinations_volta
        ]
        destinations_cleaned = (
            destinations_volta + destinations_no_volta + destinations_navigation1
        )

        part.add(
            Segment(
                id=segment_info[start_time]["ID"],
                to=destinations_cleaned,
                await_to=destinations_navigation2,  # await_to,
                force_seq=segment_info[start_time]["force_full_sequence"],
                type=segment_info[start_time]["type"],
                info=", ".join(segment_info[start_time]["info"]),
            ),
            segment_info[start_time]["start"],
            segment_info[start_time]["end"],
        )


def get_segments(part):
    """
    Get dictionary of segment objects of a part.

    Parameters
    ----------
    part: part
        A score part

    Returns
    -------
    segments: dict
        A dictionary of Segment objects indexed by segment IDs.

    """
    return {seg.id: seg for seg in part.iter_all(Segment)}


def pretty_segments(part):
    """
    Get a pretty string of all the segments in a part.
    """
    add_segments(part)
    segments = get_segments(part)
    string_list = [
        str(segments[p].id)
        + " -> (choice) "
        + "{:<8}".format(",".join(segments[p].to))
        + "\t segment "
        + "{:<20}".format(
            str(part.beat_map(segments[p].start.t))
            + " - "
            + str(part.beat_map(segments[p].end.t))
        )
        + "\t duration: "
        + "{:<6}".format(
            str(part.beat_map(segments[p].end.t) - part.beat_map(segments[p].start.t))
        )
        + "\t info: "
        + str(segments[p].info)
        for p in segments.keys()
    ]
    return "\n".join(string_list)


class Path:
    """
    Path that represents a sequence of segments.

    Parameters
    ----------
    path : list
        The string of segment IDs
    segments : dict
        A dictionary of available segments by segment ID
    used_segment_jumps : defaultdict(list), optional
        dictionary of used jumps per segment in this path
    no_repeats : bool, optional
        Flag to generate no repeating jump destinations with list_of_destinations_from_last_segment
    all_repeats : bool, optional
        Flag to generate all repeating jump destinations with list_of_destinations_from_last_segment
        (lower prority than no_repeats)
    jumped: bool
        indicates the presence of a da capo, dal segno, or al coda jump in this path
    """

    def __init__(
        self,
        path_list,
        segments,
        used_segment_jumps=None,
        no_repeats=False,
        all_repeats=False,
        jumped=False,
    ):
        self.path = path_list
        self.segments = segments
        if used_segment_jumps is None:
            self.used_segment_jumps = defaultdict(list)
        else:
            self.used_segment_jumps = used_segment_jumps
        self.ended = False
        self.jumped = False
        self.no_repeats = no_repeats
        self.all_repeats = all_repeats
        self.jumped = jumped

    def __str__(self):
        """
        return a string of segment IDs.
        """
        return "-".join(self.path)

    def __len__(self):
        return len(self.path)

    def pretty(self, part=None):
        """
        create a pretty string describing this path instance.
        If a corresponding part is given, the string will give
        segment times in beats, else in divs.
        """
        if part is None:
            string_list = [
                str(self.segments[p].id)
                + " -> (choice) "
                + ",".join(self.segments[p].to)
                + "  \t segment "
                + str(self.segments[p].start.t)
                + " - "
                + str(self.segments[p].end.t)
                + "\t duration: "
                + str(self.segments[p].duration)
                + "  \t type: "
                + str(self.segments[p].type)
                for p in self.path
            ]
        else:
            string_list = [
                str(self.segments[p].id)
                + " -> (choice) "
                + ",".join(self.segments[p].to)
                + "  \t segment "
                + str(part.beat_map(self.segments[p].start.t))
                + " - "
                + str(part.beat_map(self.segments[p].end.t))
                + "\t duration: "
                + str(part.beat_map(self.segments[p].duration))
                + "  \t type: "
                + str(self.segments[p].type)
                for p in self.path
            ]
        return "\n".join(string_list)

    def copy(self):
        """
        create a copy of this path instance.
        """
        new_path = Path(
            copy(self.path),
            copy(self.segments),
            no_repeats=self.no_repeats,
            all_repeats=self.all_repeats,
            jumped=self.jumped,
        )
        for key in self.used_segment_jumps:
            for used_dest in self.used_segment_jumps[key]:
                new_path.used_segment_jumps[key].append(used_dest)

        return new_path

    def make_copy_with_jump_to(self, destination, ignore_leap_info=True):
        """
        create a copy of this path instance with an added jump.
        If the jump is a leap (dal segno, da capo, al coda)
        and leap information is not ignored,
        set the new Path to subsequently follow the the shortest version.
        """
        new_path = self.copy()
        new_path.used_segment_jumps[new_path.path[-1]].append(destination)
        new_path.path.append(destination)

        if (
            new_path.segments[destination].type == "leap_end"
            and new_path.segments[new_path.path[-2]].type == "leap_start"
        ):
            if not new_path.jumped:
                new_path.jumped = True
                for segid in new_path.segments.keys():
                    seg = new_path.segments[segid]
                    # if destinations await the second round, add them
                    if len(seg.await_to) > 0:
                        # keep only jumps to the past
                        to = [idx for idx in seg.to if idx <= seg.id]
                        # add the waiting destinations
                        to += seg.await_to
                        # replace destinations
                        seg.to = to
                    # delete used destinations
                    new_path.used_segment_jumps[segid] = list()
                # add the jump destination to the used ones
                new_path.used_segment_jumps[new_path.path[-2]].append(destination)

            if not ignore_leap_info:
                new_path.no_repeats = True
        return new_path

    @property
    def list_of_destinations_from_last_segment(self):
        destinations = list(self.segments[self.path[-1]].to)
        previously_used_destinations = self.used_segment_jumps[self.path[-1]]
        # only continue in order of the sequence, after full consumption, start at zero
        # if the full or minimal sequence is forced,
        # return only the single possible jump destination, else return possibly many.

        if len(previously_used_destinations) != 0:
            last_destination = previously_used_destinations[-1]
            last_destination_count = previously_used_destinations.count(
                last_destination
            )
            last_destination_index = [
                i for i, n in enumerate(destinations * 100) if n == last_destination
            ][last_destination_count - 1]
            last_destination_index %= len(destinations)

        if self.no_repeats:
            # currently this is in higher priority than the full sequence
            return [destinations[-1]]

        elif self.segments[self.path[-1]].force_full_sequence or self.all_repeats:
            # if the full sequence should be used in general
            if len(previously_used_destinations) == 0:
                return [destinations[0]]
            else:
                # last_destination = previously_used_destinations[-1]
                # last_destination_index = destinations.index(last_destination)
                if last_destination_index < (len(destinations) - 1):
                    return [destinations[last_destination_index + 1]]
                else:
                    return [destinations[0]]

        else:
            if len(previously_used_destinations) == 0:
                return copy(destinations)
            else:
                # last_destination = previously_used_destinations[-1]
                # last_destination_index = destinations.index(last_destination)
                if last_destination_index < (len(destinations) - 1):
                    return copy(destinations[last_destination_index + 1 :])
                else:
                    return copy(destinations)


def unfold_paths(path, paths, ignore_leap_info=True):
    """
    Given a starting Path (at least one segment) recursively unfold into all possible
    Paths with its segments. Ended Paths are stored in a list.

    Parameters
    ----------
    path : Path
        a starting Path with at least one segment to be unfolded
    paths : list
        empty list to accumulate paths that are fully unfolded until an "end" keyword was found
    """
    destinations = path.list_of_destinations_from_last_segment
    for destination_id in destinations:
        if destination_id == "END":
            path.ended = True
            paths.append(path)
        else:
            new_path = path.make_copy_with_jump_to(
                destination_id, ignore_leap_info=ignore_leap_info
            )
            unfold_paths(new_path, paths, ignore_leap_info=ignore_leap_info)


def get_paths(part, no_repeats=False, all_repeats=False, ignore_leap_info=True):
    """
    Get a list of paths and and a dictionary of segment objects of a part.

    Common settings to get specific paths:
    - default: all possible paths
        (no_repeats = False, all_repeats = False, ignore_leap_info = True)
    - default: all possible paths but without repetitions after leap
        (no_repeats = False, all_repeats = False, ignore_leap_info = False)
    - The longest possible path
        (no_repeats = False, all_repeats = True, ignore_leap_info = True)
    - The longest possible path but without repetitions after leap
        (no_repeats = False, all_repeats = True, ignore_leap_info = False)
    - The shortest possible path.
        (no_repeats = True)
        Note this might not be musically valid, e.g. a passing a "fine"
        even a first time will stop this unfolding.

    Parameters
    ----------
    part: part
        A score part
    no_repeats : bool, optional
        Flag to choose no repeating segments, i.e. the shortest path.
    all_repeats : bool, optional
        Flag to choose all repeating segments, i.e. the longest path.
        (lower priority than the previous flag)
    ignore_leap_info : bool, optional
        If not ignored, Path changes to no_repeats = True if a leap is encountered.
        (A leap is a used dal segno, da capo, or al coda marking)

    Returns
    -------
    paths: list
        A list of path objects

    """
    add_segments(part)
    segments = get_segments(part)
    paths = list()
    unfold_paths(
        Path([chr(65)], segments, no_repeats=no_repeats, all_repeats=all_repeats),
        paths,
        ignore_leap_info=ignore_leap_info,
    )

    return paths


def new_part_from_path(path, part, update_ids=True):
    """
    create a new Part from a Path and an underlying Part

    Parameters
    ----------
    path: Path
        A Path object
    part: part
        A score part
    update_ids : bool (optional)
        Update note ids to reflect the repetitions. Note IDs will have
        a '-<repetition number>', e.g., 'n132-1' and 'n132-2'
        represent the first and second repetition of 'n132' in the
        input `part`. Defaults to False.

    Returns
    -------
    new_part: part
        A score part corresponding to the Path

    """
    scorevariant = ScoreVariant(part)
    for segment_id in path.path:
        scorevariant.add_segment(
            path.segments[segment_id].start, path.segments[segment_id].end
        )

    new_part = scorevariant.create_variant_part()
    if update_ids:
        update_note_ids_after_unfolding(new_part)
    return new_part


def new_scorevariant_from_path(path, part):
    """
    create a new Part from a Path and an underlying Part

    Parameters
    ----------
    path: Path
        A Path object
    part: part
        A score part

    Returns
    -------
    scorevariant: ScoreVariant
        A ScoreVariant object with segments corresponding to the part

    """
    scorevariant = ScoreVariant(part)
    for segment_id in path.path:
        scorevariant.add_segment(
            path.segments[segment_id].start, path.segments[segment_id].end
        )
    return scorevariant


# UPDATED VERSION
def iter_unfolded_parts(part, update_ids=True):
    """Iterate over unfolded clones of `part`.

    For each repeat construct in `part` the iterator produces two
    clones, one with the repeat included and another without the
    repeat. That means the number of items returned is two to the
    power of the number of repeat constructs in the part.

    The first item returned by the iterator is the version of the part
    without any repeated sections, the last item is the version of the
    part with all repeat constructs expanded.

    Parameters
    ----------
    part : :class:`Part`
        Part to unfold
    update_ids : bool (optional)
        Update note ids to reflect the repetitions. Note IDs will have
        a '-<repetition number>', e.g., 'n132-1' and 'n132-2'
        represent the first and second repetition of 'n132' in the
        input `part`. Defaults to False.

    Yields
    ------

    """
    paths = get_paths(part, no_repeats=False, all_repeats=False, ignore_leap_info=True)

    for p in paths:
        yield new_part_from_path(p, part, update_ids=update_ids)


# UPDATED VERSION
def unfold_part_maximal(score: ScoreLike, update_ids=True, ignore_leaps=True):
    """Return the "maximally" unfolded part/score, that is, a copy of the
    part where all segments marked with repeat signs are included
    twice.

    Parameters
    ----------
    score : ScoreLike
        The Part/Score to unfold.
    update_ids : bool (optional)
        Update note ids to reflect the repetitions. Note IDs will have
        a '-<repetition number>', e.g., 'n132-1' and 'n132-2'
        represent the first and second repetition of 'n132' in the
        input `part`. Defaults to False.
    ignore_leaps : bool (optional)
        If ignored, repetitions after a leap are unfolded fully.
        A leap is a used dal segno, da capo, or al coda marking.
        Defaults to True.

    Returns
    -------
    unfolded_part : ScoreLike
        The unfolded Part/Score

    """
    if isinstance(score, Score):
        # Copy needs to be deep, otherwise the recursion limit will be exceeded
        old_recursion_depth = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        # Deep copy of score
        new_score = deepcopy(score)
        # Reset recursion limit to previous value to avoid side effects
        sys.setrecursionlimit(old_recursion_depth)
        new_partlist = list()
        for score in new_score.parts:
            unfolded_part = unfold_part_maximal(
                score, update_ids=update_ids, ignore_leaps=ignore_leaps
            )
            new_partlist.append(unfolded_part)
        new_score.parts = new_partlist
        return new_score

    paths = get_paths(
        score, no_repeats=False, all_repeats=True, ignore_leap_info=ignore_leaps
    )

    unfolded_part = new_part_from_path(paths[0], score, update_ids=update_ids)
    return unfolded_part


def unfold_part_minimal(score: ScoreLike):
    """Return the "minimally" unfolded score/part, that is, a copy of the
    part where all segments marked with repeat are included only once.
    For voltas only the last volta segment is included.
    Note this might not be musically valid, e.g. a passing a "fine" even a first time will stop this unfolding.
    Warning: The unfolding of repeats is computed part-wise, inconsistent repeat markings of parts of a single result
    in inconsistent unfoldings.

    Parameters
    ----------
    score: ScoreLike
        The score/part to unfold.

    Returns
    -------
    unfolded_score : ScoreLike
        The unfolded Part

    """
    if isinstance(score, Score):
        # Copy needs to be deep, otherwise the recursion limit will be exceeded
        old_recursion_depth = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        # Deep copy of score
        unfolded_score = deepcopy(score)
        # Reset recursion limit to previous value to avoid side effects
        sys.setrecursionlimit(old_recursion_depth)
        new_partlist = list()
        for part in unfolded_score.parts:
            unfolded_part = unfold_part_minimal(part)
            new_partlist.append(unfolded_part)
        unfolded_score.parts = new_partlist
        return unfolded_score

    paths = get_paths(score, no_repeats=True, all_repeats=False, ignore_leap_info=True)

    unfolded_score = new_part_from_path(paths[0], score, update_ids=False)
    return unfolded_score


# UPDATED / UNCHANGED VERSION
def unfold_part_alignment(part, alignment):
    """Return the unfolded part given an alignment, that is, a copy
    of the part where the segments are repeated according to the
    repetitions in a performance.

    Parameters
    ----------
    part : :class:`Part`
        The Part to unfold.
    alignment : list of dictionaries
        List of dictionaries containing an alignment (like the ones
        obtained from a MatchFile (see `alignment_from_matchfile`).

    Returns
    -------
    unfolded_part : :class:`Part`
        The unfolded Part

    """

    unfolded_parts = []

    alignment_ids = []

    for n in alignment:
        if n["label"] == "match" or n["label"] == "deletion":
            alignment_ids.append(n["score_id"])

    score_variants = make_score_variants(part)

    alignment_score_ids = np.zeros((len(alignment_ids), len(score_variants)))
    unfolded_part_length = np.zeros(len(score_variants))
    for j, sv in enumerate(score_variants):
        u_part = sv.create_variant_part()
        update_note_ids_after_unfolding(u_part)
        unfolded_parts.append(u_part)
        u_part_ids = [n.id for n in u_part.notes_tied]
        unfolded_part_length[j] = len(u_part_ids)
        for i, aid in enumerate(alignment_ids):
            alignment_score_ids[i, j] = aid in u_part_ids

    coverage = np.mean(alignment_score_ids, 0)

    best_idx = np.where(coverage == coverage.max())[0]

    if len(best_idx) > 1:
        best_idx = best_idx[unfolded_part_length[best_idx].argmin()]

    # append "-1" to alignment if the score_id's in alignment
    if not any(["-1" in al.get("score_id", "") for al in alignment]):
        for n in alignment:
            if "score_id" in n:
                n["score_id"] = f"{n['score_id']}-1"

    return unfolded_parts[int(best_idx)]


# UPDATED
def make_score_variants(part):
    # non-public (use unfold_part_maximal, or iter_unfolded_parts)

    """
    Create a list of ScoreVariant objects, each representing a
    distinct way to unfold the score, based on the repeat structure.

    Parameters
    ----------
    part : :class:`Part`
        A part for which to make the score variants

    Returns
    -------
    list
        List of ScoreVariant objects

    """
    paths = get_paths(part, no_repeats=False, all_repeats=False, ignore_leap_info=True)

    svs = list()
    for path in paths:
        svs.append(new_scorevariant_from_path(path, part))

    return svs


def _fill_rests_within_measure(measure: Measure, part: Part) -> None:
    start_time = measure.start.t
    end_time = measure.end.t
    notes = np.array(list(part.iter_all(GenericNote, start_time, end_time)))
    voc_staff = np.array([[n.voice, n.staff] for n in notes])
    un_voc_staff, inverse_map = np.unique(voc_staff, axis=0, return_inverse=True)
    for i in range(un_voc_staff):
        note_mask = inverse_map == i
        notes_per_vocstaff = notes[note_mask]
        # get note with min start.t
        min_start_note = notes_per_vocstaff[np.argmin(notes_per_vocstaff.start.t)]
        if min_start_note.start.t > start_time:
            sym_dur = estimate_symbolic_duration(
                min_start_note.start.t - start_time, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_start_note.staff,
                voice=min_start_note.voice,
            )
            part.add(rest, start_time, min_start_note.start.t)

        min_end_note = notes_per_vocstaff[
            np.argmin(np.vectorize(lambda x: x.end.t)(notes_per_vocstaff))
        ]
        if min_end_note.end.t < end_time:
            sym_dur = estimate_symbolic_duration(
                end_time - min_end_note.end.t, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_end_note.staff,
                voice=min_end_note.voice,
            )
            part.add(rest, min_end_note.end.t, end_time)


def _fill_rests_global(
    measure: Measure, part: Part, unique_voc_staff: np.ndarray
) -> None:
    start_time = measure.start.t
    end_time = measure.end.t
    if end_time - start_time == 0:
        return
    notes = np.array(
        list(part.iter_all(GenericNote, start_time, end_time, include_subclasses=True))
    )
    voc_staff = np.array([[n.voice, n.staff] for n in notes])
    un_voc_staff, inverse_map = np.unique(voc_staff, axis=0, return_inverse=True)
    for i in range(un_voc_staff.shape[0]):
        note_mask = inverse_map == i
        notes_per_vocstaff = notes[note_mask]
        # get note with min start.t
        min_start_note = notes_per_vocstaff[
            np.argmin(np.vectorize(lambda x: x.start.t)(notes_per_vocstaff))
        ]
        if min_start_note.start.t > start_time:
            sym_dur = estimate_symbolic_duration(
                min_start_note.start.t - start_time, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_start_note.staff,
                voice=min_start_note.voice,
            )
            part.add(rest, start_time, min_start_note.start.t)

        min_end_note = notes_per_vocstaff[
            np.argmax(np.vectorize(lambda x: x.end.t)(notes_per_vocstaff))
        ]
        if min_end_note.end.t < end_time:
            sym_dur = estimate_symbolic_duration(
                end_time - min_end_note.end.t, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_end_note.staff,
                voice=min_end_note.voice,
            )
            part.add(rest, min_end_note.end.t, end_time)

    if un_voc_staff.shape[0] != unique_voc_staff.shape[0]:
        if un_voc_staff.shape[0] == 0:
            diff = unique_voc_staff
        else:
            # View `un_voc_staff` and `unique_voc_staff` as 1-D structured arrays
            x_sa = un_voc_staff.view([("", un_voc_staff.dtype)] * un_voc_staff.shape[1])
            y_sa = unique_voc_staff.view(
                [("", unique_voc_staff.dtype)] * unique_voc_staff.shape[1]
            )
            # Find rows in `unique_voc_staff` that are not in `un_voc_staff`
            diff = np.setdiff1d(y_sa, x_sa)
        for voice, staff in diff:
            sym_dur = estimate_symbolic_duration(
                end_time - start_time, part._quarter_durations[0]
            )
            rest = Rest(symbolic_duration=sym_dur, staff=staff, voice=voice)
            part.add(rest, start_time, end_time)


def fill_rests(score_data: ScoreLike, measurewise=True) -> None:
    """
    Fill rests in a score when a voice starts in a middle of a measure and no rest precedes.

    When measurewise is True, the voices are searched within a measure.
    When measurewise is False, the rests are filled globally in the score for all voices and staffs.

    Parameters
    ----------
    score_data: ScoreLike
        The score to fill rests
    measurewise: bool
        If True, fill rests within a measure. If False, fill rests globally in the score.
    """
    if isinstance(score_data, Score):
        partlist = score_data.parts
    else:
        partlist = [score_data]
    for part in partlist:
        measures = part.measures
        if measurewise:
            for measure in measures:
                _fill_rests_within_measure(measure, part)
        else:
            note_array = part.note_array(include_staff=True)
            unique_vocstaff = np.unique(
                np.array([note_array["voice"], note_array["staff"]], dtype=np.int64),
                axis=1,
            )
            for measure in measures:
                _fill_rests_global(measure, part, unique_vocstaff.T)


def merge_parts(parts, reassign="voice"):
    """Merge list of parts or PartGroup into a single part.
     All parts are expected to have the same time signature
    and quarter division.

    All elements are merged, except elements with class:Barline,
    Page, System, Clef, Measure, TimeSignature, KeySignature
    that are only taken from the first part.

    WARNING: this modifies the elements in the input, so the
    original input should not be used anymore.

    Parameters
    ----------
    parts : PartGroup, list of parts and partGroups
        The parts to merge
    reassign: string (optional)
        If "staff" the new part have as many staves as the sum
        of the staves in parts, and the staff numbers get reassigned.
        If "voice", the new part have only one staff, and as manually
        voices as the sum of the voices in parts; the voice number
        get reassigned.
        If "both", we reassign all the staves and voices to have unique staff
        and voice numbers. According to musicxml standards, we consider 4 voices
        per staff. So for example staff 1 will have voices 1,2,3,4, staff 2 will
        have voices 5,6,7,8, and so on.

    Returns
    -------
    Part
        A new part that contains the elements of the old parts

    """
    # check if reassign has valid values
    if reassign not in ["staff", "voice", "auto"]:
        raise ValueError(
            "Only 'staff' and 'voice' are supported ressign values. Found", reassign
        )

    # unfold grouppart and list of parts in a list of parts
    if isinstance(parts, Score):
        parts = parts.parts
    else:
        parts = list(iter_parts(parts))

    # if there is only one part (it could be a list with one part or a partGroup with one part)
    if len(parts) == 1:
        return parts[0]

    # check if there is only one division for all parts
    parts_quarter_times = [p._quarter_times for p in parts]
    parts_quarter_durations = [p._quarter_durations for p in parts]
    if not all([len(qd) == 1 for qd in parts_quarter_durations]):
        raise Exception(
            "Merging parts with multiple divisions is not supported. Found divisions",
            parts_quarter_durations,
            "at times",
            parts_quarter_times,
        )

    # pass from an array of array with one elements, to array of elements
    parts_quarter_durations = [durs[0] for durs in parts_quarter_durations]

    lcm = np.lcm.reduce(parts_quarter_durations)
    time_multiplier_per_part = [int(lcm / d) for d in parts_quarter_durations]

    # create a new part and fill it with all objects in other parts
    new_part = Part(parts[0].id)
    new_part._quarter_times = [0]
    new_part._quarter_durations = [lcm]

    note_arrays = [part.note_array(include_staff=True) for part in parts]
    # find the unique number of voices for each part (voice numbers start from 1)
    unique_voices = [np.unique(note_array["voice"]) for note_array in note_arrays]
    # find the unique number of staves for each part
    unique_staves = [np.unique(note_array["staff"]) for note_array in note_arrays]
    # find the maximum number of voices for each part (voice numbers start from 1)
    maximum_voices = [max(unique_voice, default=1) for unique_voice in unique_voices]
    # find the maximum number of staves for each part
    maximum_staves = [max(unique_staff, default=1) for unique_staff in unique_staves]

    if reassign in ["staff", "auto"]:
        el_to_discard = (
            Barline,
            Page,
            System,
            Measure,
            TimeSignature,
            KeySignature,
            DaCapo,
            Fine,
            Fermata,
            Ending,
            Tempo,
        )
    elif reassign == "voice":
        el_to_discard = (
            Barline,
            Page,
            System,
            Clef,
            Measure,
            TimeSignature,
            KeySignature,
            DaCapo,
            Fine,
            Fermata,
            Ending,
            Tempo,
        )

    for p_ind, p in enumerate(parts):
        if reassign == "auto":
            # find how many staves this part has
            n_staves = len(unique_staves[p_ind])
            # find total number of staves in previous parts
            if p_ind == 0:
                n_previous_staves = 0
            else:
                n_previous_staves = sum(
                    [len(unique_staff) for unique_staff in unique_staves[:p_ind]]
                )
            # build a mapping between the old staff numbers and the new staff numbers
            staff_mapping = dict(
                zip(
                    unique_staves[p_ind], n_previous_staves + np.arange(1, n_staves + 1)
                )
            )
            # find how many voices this part has
            n_voices = len(unique_voices[p_ind])
            # build a mapping between the old and new voices
            voice_mapping = dict(
                zip(
                    unique_voices[p_ind],
                    n_previous_staves * 4 + np.arange(1, n_voices + 1),
                )
            )
        for e in p.iter_all():
            # full copy the first part and partially copy the others
            # we don't copy elements like duplicate barlines, clefs or
            # time signatures for others
            # TODO : check  DaCapo, Fine, Fermata, Ending, Tempo
            if p_ind == 0 or not isinstance(
                e,
                el_to_discard,
            ):  # a time multiplier is used to account for different divisions
                new_start = e.start.t * time_multiplier_per_part[p_ind]
                new_end = (
                    e.end.t * time_multiplier_per_part[p_ind]
                    if e.end is not None
                    else None
                )
                if reassign == "voice":
                    if isinstance(e, GenericNote):
                        e.voice = e.voice + sum(maximum_voices[:p_ind])
                elif reassign == "staff":
                    if isinstance(e, (GenericNote, Words, Direction, Clef)):
                        e.staff = (e.staff if e.staff is not None else 1) + sum(
                            maximum_staves[:p_ind]
                        )
                elif reassign == "auto":
                    # assign based on the voice and staff mappings
                    if isinstance(e, GenericNote):
                        # new voice is computed as the sum of voices in staves in previous parts, plus the current
                        e.voice = voice_mapping[e.voice]
                    if isinstance(e, (GenericNote, Words, Direction, Clef)):
                        e.staff = staff_mapping[e.staff]
                new_part.add(e, start=new_start, end=new_end)

                # new_part.add(copy.deepcopy(e), start=new_start, end=new_end)
    return new_part


def _fill_rests_within_measure(measure: Measure, part: Part) -> None:
    start_time = measure.start.t
    end_time = measure.end.t
    notes = np.array(
        list(part.iter_all(GenericNote, start_time, end_time, include_subclasses=True))
    )

    # voc_staff is now transformed to only voice
    voc_staff = np.array([[n.voice, n.staff] for n in notes])
    un_voice, inverse_map = np.unique(voc_staff[:, 0], axis=0, return_inverse=True)
    # Check if a staff is empty and fill it with rests
    unique_staff = np.unique(voc_staff[:, 1])
    if len(unique_staff) < part.number_of_staves:
        for staff in range(1, part.number_of_staves + 1):
            if staff not in unique_staff:
                # solution when estimation returns composite durations.
                sym_dur = estimate_symbolic_duration(
                    end_time - start_time,
                    part._quarter_durations[0],
                    return_com_durations=True,
                )
                if isinstance(sym_dur, tuple):
                    st = start_time
                    for i, sd in enumerate(sym_dur):
                        et = start_time + symbolic_to_numeric_duration(
                            sd, part._quarter_durations[0]
                        )
                        rest = Rest(
                            symbolic_duration=sd, staff=staff, voice=un_voice.max() + 1
                        )
                        part.add(rest, st, et)
                        st = et
                else:
                    rest = Rest(
                        symbolic_duration=sym_dur, staff=staff, voice=un_voice.max() + 1
                    )
                    part.add(rest, start_time, end_time)
    # Now we fill the rests for each voice
    for i in range(len(un_voice)):
        note_mask = inverse_map == i
        notes_per_vocstaff = notes[note_mask]
        sort_note_start = np.argsort(
            np.vectorize(lambda x: x.start.t)(notes_per_vocstaff)
        )
        sort_note_end = np.argsort(np.vectorize(lambda x: x.end.t)(notes_per_vocstaff))
        # get note with min start.t and fill the rest before it if needed
        min_start_note = notes_per_vocstaff[sort_note_start[0]]
        if min_start_note.start.t > start_time:
            sym_dur = estimate_symbolic_duration(
                min_start_note.start.t - start_time,
                part._quarter_durations[0],
                return_com_durations=True,
            )
            # solution when estimation returns composite durations.
            if isinstance(sym_dur, tuple):
                st = start_time
                for i, sd in enumerate(sym_dur):
                    et = st + symbolic_to_numeric_duration(
                        sd, part._quarter_durations[0]
                    )
                    rest = Rest(
                        symbolic_duration=sd,
                        staff=min_start_note.staff,
                        voice=min_start_note.voice,
                    )
                    part.add(rest, st, et)
                    st = et
            else:
                rest = Rest(
                    symbolic_duration=sym_dur,
                    staff=min_start_note.staff,
                    voice=min_start_note.voice,
                )
                part.add(rest, start_time, min_start_note.start.t)

        # get note with max end.t and fill the rest after it if needed
        min_end_note = notes_per_vocstaff[sort_note_end[-1]]
        if min_end_note.end.t < end_time:
            sym_dur = estimate_symbolic_duration(
                end_time - min_end_note.end.t,
                part._quarter_durations[0],
                return_com_durations=True,
            )
            # solution when estimation returns composite durations.
            if isinstance(sym_dur, tuple):
                st = min_end_note.end.t
                for i, sd in enumerate(sym_dur):
                    et = st + symbolic_to_numeric_duration(
                        sd, part._quarter_durations[0]
                    )
                    rest = Rest(
                        symbolic_duration=sd,
                        staff=min_end_note.staff,
                        voice=min_end_note.voice,
                    )
                    part.add(rest, st, et)
                    st = et
            else:
                rest = Rest(
                    symbolic_duration=sym_dur,
                    staff=min_end_note.staff,
                    voice=min_end_note.voice,
                )
                part.add(rest, min_end_note.end.t, end_time)

        if len(sort_note_start) <= 1:
            continue
        # fill the rests between notes if needed (i.e. if there is a gap between notes)
        for i in range(1, len(sort_note_start)):
            if (
                notes_per_vocstaff[sort_note_start[i]].start.t
                > notes_per_vocstaff[sort_note_end[i - 1]].end.t
            ):
                sym_dur = estimate_symbolic_duration(
                    notes_per_vocstaff[sort_note_start[i]].start.t
                    - notes_per_vocstaff[sort_note_end[i - 1]].end.t,
                    part._quarter_durations[0],
                    return_com_durations=True,
                )
                if isinstance(sym_dur, tuple):
                    st = notes_per_vocstaff[sort_note_end[i - 1]].end.t
                    for i, sd in enumerate(sym_dur):
                        et = st + symbolic_to_numeric_duration(
                            sd, part._quarter_durations[0]
                        )
                        rest = Rest(
                            symbolic_duration=sd,
                            staff=notes_per_vocstaff[sort_note_end[i - 1]].staff,
                            voice=notes_per_vocstaff[sort_note_end[i - 1]].voice,
                        )
                        part.add(rest, st, et)
                        st = et
                else:
                    rest = Rest(
                        symbolic_duration=sym_dur,
                        staff=notes_per_vocstaff[sort_note_end[i - 1]].staff,
                        voice=notes_per_vocstaff[sort_note_end[i - 1]].voice,
                    )
                    part.add(
                        rest,
                        notes_per_vocstaff[sort_note_end[i - 1]].end.t,
                        notes_per_vocstaff[sort_note_start[i]].start.t,
                    )


def _fill_rests_global(
    measure: Measure, part: Part, unique_voc_staff: np.ndarray
) -> None:
    start_time = measure.start.t
    end_time = measure.end.t
    if end_time - start_time == 0:
        return
    notes = np.array(
        list(part.iter_all(GenericNote, start_time, end_time, include_subclasses=True))
    )
    voc_staff = np.array([[n.voice, n.staff] for n in notes])
    un_voc_staff, inverse_map = np.unique(voc_staff, axis=0, return_inverse=True)
    for i in range(un_voc_staff.shape[0]):
        note_mask = inverse_map == i
        notes_per_vocstaff = notes[note_mask]
        # get note with min start.t
        min_start_note = notes_per_vocstaff[
            np.argmin(np.vectorize(lambda x: x.start.t)(notes_per_vocstaff))
        ]
        if min_start_note.start.t > start_time:
            sym_dur = estimate_symbolic_duration(
                min_start_note.start.t - start_time, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_start_note.staff,
                voice=min_start_note.voice,
            )
            part.add(rest, start_time, min_start_note.start.t)

        min_end_note = notes_per_vocstaff[
            np.argmax(np.vectorize(lambda x: x.end.t)(notes_per_vocstaff))
        ]
        if min_end_note.end.t < end_time:
            sym_dur = estimate_symbolic_duration(
                end_time - min_end_note.end.t, part._quarter_durations[0]
            )
            rest = Rest(
                symbolic_duration=sym_dur,
                staff=min_end_note.staff,
                voice=min_end_note.voice,
            )
            part.add(rest, min_end_note.end.t, end_time)

    if un_voc_staff.shape[0] != unique_voc_staff.shape[0]:
        if un_voc_staff.shape[0] == 0:
            diff = unique_voc_staff
        else:
            # View `un_voc_staff` and `unique_voc_staff` as 1-D structured arrays
            x_sa = un_voc_staff.view([("", un_voc_staff.dtype)] * un_voc_staff.shape[1])
            y_sa = unique_voc_staff.view(
                [("", unique_voc_staff.dtype)] * unique_voc_staff.shape[1]
            )
            # Find rows in `unique_voc_staff` that are not in `un_voc_staff`
            diff = np.setdiff1d(y_sa, x_sa)
        for voice, staff in diff:
            sym_dur = estimate_symbolic_duration(
                end_time - start_time, part._quarter_durations[0]
            )
            rest = Rest(symbolic_duration=sym_dur, staff=staff, voice=voice)
            part.add(rest, start_time, end_time)


def fill_rests(score_data: ScoreLike, measurewise=True) -> None:
    """
    Fill rests in a score when a voice starts in a middle of a measure and no rest precedes.

    When measurewise is True, the voices are searched within a measure.
    When measurewise is False, the rests are filled globally in the score for all voices and staffs.

    Parameters
    ----------
    score_data: ScoreLike
        The score to fill rests
    measurewise: bool
        If True, fill rests within a measure. If False, fill rests globally in the score.
    """
    if isinstance(score_data, Score):
        partlist = score_data.parts
    else:
        partlist = [score_data]
    for part in partlist:
        measures = part.measures
        if measurewise:
            for measure in measures:
                _fill_rests_within_measure(measure, part)
        else:
            note_array = part.note_array(include_staff=True)
            unique_vocstaff = np.unique(
                np.array([note_array["voice"], note_array["staff"]], dtype=np.int64),
                axis=1,
            )
            for measure in measures:
                _fill_rests_global(measure, part, unique_vocstaff.T)


def infer_beaming(part: ScoreLike):
    """
    Infer beaming from the metrical position of notes in a part.

    This function infers the beaming based on the time signature for all notes.
    It separates the notes into groups based on their staff and voice.

    Parameters
    ----------
    part: ScoreLike
        The part to infer beaming for. This can be a part or a score.
        If a score is given, the function will infer beaming for all parts in the score.

    """

    if isinstance(part, Score):
        for p in part.parts:
            infer_beaming(p)
    else:
        note_array = part.note_array(
            include_metrical_position=True,
            include_staff=True,
            include_time_signature=True,
        )
        beat_ends = note_array["onset_beat"] + note_array["duration_beat"]
        # split note_array into groups based on staff and voice
        # unique_vocstaff = np.unique(note_array[['voice', 'staff']], axis=0)
        # for v, s in unique_vocstaff:
        #     mask = (note_array['voice'] == v) & (note_array['staff'] == s)
        unique_vocstaff = np.unique(note_array["voice"])
        for v in unique_vocstaff:
            mask = note_array["voice"] == v
            # get the metrical position of the notes
            na_vocstaff = note_array[mask]
            # get the beat ends of the notes
            beat_end = beat_ends[mask]
            # get notes
            beat_multiplier = 4 / na_vocstaff["ts_beat_type"]
            mus_beats = (
                na_vocstaff["ts_beats"]
                / na_vocstaff["ts_mus_beats"]
                * (na_vocstaff["ts_beat_type"] > 4)
            )
            mus_beats = np.where(mus_beats == 0, 1, mus_beats)
            max_mus_beat = mus_beats.max()
            beam_start_mask = np.isclose(
                np.mod(na_vocstaff["onset_beat"], mus_beats), 0.0
            ) & (na_vocstaff["duration_beat"] * beat_multiplier <= 0.5)
            beam_end_mask = np.isclose(np.mod(beat_end, mus_beats), 0.0) & (
                na_vocstaff["duration_beat"] * beat_multiplier <= 0.5
            )
            beam_between = (
                (na_vocstaff["duration_beat"] * beat_multiplier <= 0.5)
                & ~beam_start_mask
                & ~beam_end_mask
            )
            id_beam_start = na_vocstaff["id"][beam_start_mask]
            id_beam_end = na_vocstaff["id"][beam_end_mask]
            id_beam_between = na_vocstaff["id"][beam_between]
            start_time = na_vocstaff["onset_div"].min()
            end_time = na_vocstaff["onset_div"].max() + 1
            prev_beam = None
            notes_in_beam = []
            notes_in_vs = list(part.iter_all(Note, start_time, end_time))
            notes_in_vs.sort(key=lambda x: x.start.t)
            prev_start = 0
            for note in notes_in_vs:
                # if note.voice != v or note.staff != s:
                if note.voice != v:
                    continue
                if note.beam is not None:
                    continue

                if (
                    part.beat_map(note.start.t) - part.beat_map(prev_start)
                    > max_mus_beat
                ):
                    prev_beam = None
                    notes_in_beam = []
                    prev_start = note.start.t
                if note.id in id_beam_start:
                    notes_in_beam = []
                    prev_beam = Beam()
                    prev_start = note.start.t
                    notes_in_beam.append(note)
                elif note.id in id_beam_end:
                    if prev_beam is not None:
                        notes_in_beam.append(note)
                        if len(notes_in_beam) > 1:
                            for n in notes_in_beam:
                                n.assign_beam(prev_beam)
                            part.add(prev_beam, prev_beam.start.t, prev_beam.end.t)
                    notes_in_beam = []
                    prev_beam = None
                elif note.id in id_beam_between:
                    if prev_beam is None:
                        prev_start = note.start.t
                        notes_in_beam = []
                        prev_beam = Beam()
                    notes_in_beam.append(note)


def is_a_within_b(a, b, wholly=False):
    """
    Returns a boolean indicating whether a is (wholly) within b.

    Parameters
    ----------
    a: TimePoint, TimedObject, int
        Query object
    b: TimedObject
        Container object
    wholly: bool
        True = a needs to wholly contained in b
    """
    contained = None
    if not isinstance(b, TimedObject):
        warnings.warn("b needs to be TimedObject")
    if isinstance(a, TimePoint):
        contained = a.t <= b.end.t and a.t >= b.start.t
    elif isinstance(a, int):
        contained = a <= b.end.t and a >= b.start.t
    elif isinstance(a, TimedObject):
        contained_start = a.start.t <= b.end.t and a.start.t >= b.start.t
        contained_end = a.end.t <= b.end.t and a.end.t >= b.start.t
        if wholly:
            contained = contained_start and contained_end
        else:
            contained = contained_start or contained_end
    else:
        warnings.warn("a needs to be TimePoint, TimedObject, or int.")
    return contained


def process_local_key(loc_k_text, glob_k_text, return_step_alter=False):
    local_key_sharps = loc_k_text.count("#")
    local_key_flats = loc_k_text.count("b")
    local_key = loc_k_text.replace("#", "").replace("b", "")
    local_key_is_minor = local_key.islower()
    local_key = local_key.lower()
    global_key_is_minor = glob_k_text.islower()
    if (
        local_key_is_minor == global_key_is_minor
        and local_key == "i"
        and local_key_sharps - local_key_flats == 0
        and (not return_step_alter)
    ):
        return glob_k_text
    g_key = "minor" if glob_k_text.islower() else "major"
    num, qual = LOCAL_KEY_TRASPOSITIONS_DCML[g_key][local_key]
    transposition_interval = Interval(num, qual)
    transposition_interval = transposition_interval.change_quality(
        local_key_sharps - local_key_flats
    )
    key_step = re.search(r"[a-gA-G]", glob_k_text).group(0)
    key_alter = (
        re.search(r"[#b]", glob_k_text).group(0)
        if re.search(r"[#b]", glob_k_text)
        else ""
    )
    key_alter = key_alter.replace("b", "-")
    key_alter = ALT_TO_INT[key_alter]
    key_step, key_alter = transpose_note(key_step, key_alter, transposition_interval)
    if return_step_alter:
        return key_step, key_alter
    local_key = (
        key_step.lower() if local_key_is_minor else key_step.upper()
    ) + INT_TO_ALT[key_alter]
    return local_key


Roman2Interval_Maj = {
    "I": Interval(1, "P"),
    "II": Interval(2, "M"),
    "III": Interval(3, "M"),
    "III+": Interval(3, "M"),
    "IV": Interval(4, "P"),
    "V": Interval(5, "P"),
    "VI": Interval(6, "M"),
    "VII": Interval(7, "M"),
    "i": Interval(1, "P"),
    "ii": Interval(2, "M"),
    "iii": Interval(3, "m"),
    "iv": Interval(4, "P"),
    "v": Interval(5, "P"),
    "vi": Interval(6, "M"),
    "vii": Interval(7, "M"),
    "viio": Interval(7, "M"),
    "N": Interval(2, "m"),
    "iio": Interval(2, "M"),
    "Ger7": Interval(4, "A"),
    "Fr7": Interval(4, "A"),
    "It": Interval(4, "A"),
}

Roman2Interval_Min = {
    "I": Interval(1, "P"),
    "II": Interval(2, "M"),
    "III": Interval(3, "m"),
    "III+": Interval(3, "m"),
    "IV": Interval(4, "P"),
    "V": Interval(5, "P"),
    "VI": Interval(6, "m"),
    "VII": Interval(7, "m"),
    "i": Interval(1, "P"),
    "ii": Interval(2, "M"),
    "iii": Interval(3, "m"),
    "iv": Interval(4, "P"),
    "v": Interval(5, "P"),
    "vi": Interval(6, "M"),
    "vii": Interval(7, "M"),
    "viio": Interval(7, "M"),
    "N": Interval(2, "m"),
    "iio": Interval(2, "M"),
    "Ger7": Interval(4, "A"),
    "Fr7": Interval(4, "A"),
    "It": Interval(4, "A"),
}


def process_local_key(loc_k, glob_k, return_step_alter=False):
    local_key_sharps = loc_k.count("#")
    local_key_flats = loc_k.count("b")
    local_key = loc_k.replace("#", "").replace("b", "")
    local_key_is_minor = local_key.islower()
    local_key = local_key.lower()
    global_key_is_minor = glob_k.islower()
    if (
        local_key_is_minor == global_key_is_minor
        and local_key == "i"
        and local_key_sharps - local_key_flats == 0
        and (not return_step_alter)
    ):
        return glob_k
    g_key = "minor" if glob_k.islower() else "major"
    # keep only letters in local_key
    local_key = re.sub(r"[^a-zA-Z]", "", local_key)
    num, qual = LOCAL_KEY_TRASPOSITIONS_DCML[g_key][local_key]
    transposition_interval = Interval(num, qual)
    transposition_interval = transposition_interval.change_quality(
        local_key_sharps - local_key_flats
    )
    key_step = re.search(r"[a-gA-G]", glob_k).group(0)
    key_alter = (
        re.search(r"[#b]", glob_k).group(0) if re.search(r"[#b]", glob_k) else ""
    )
    key_alter = key_alter.replace("b", "-")
    key_alter = ALT_TO_INT[key_alter]
    key_step, key_alter = transpose_note(key_step, key_alter, transposition_interval)
    if return_step_alter:
        return key_step, key_alter
    local_key = (
        key_step.lower() if local_key_is_minor else key_step.upper()
    ) + INT_TO_ALT[key_alter]
    return local_key


class InvalidTimePointException(Exception):
    """Raised when a time point is instantiated with an invalid number."""

    def __init__(self, message=None):
        super().__init__(message)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
