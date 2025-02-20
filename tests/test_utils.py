#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utility methods.
"""
import unittest
import partitura
import numpy as np

from partitura.utils import music
from partitura.musicanalysis import performance_codec
from tests import (
    MATCH_IMPORT_EXPORT_TESTFILES,
    VOSA_TESTFILES,
    MOZART_VARIATION_FILES,
    TOKENIZER_TESTFILES,
)

from scipy.interpolate import interp1d as scinterp1d
from partitura.utils.generic import interp1d as pinterp1d
from partitura.utils.music import tokenize

try:
    import miditok
    import miditoolkit

    HAS_MIDITOK = True
except ImportError:
    HAS_MIDITOK = False

RNG = np.random.RandomState(1984)


class TestGetMatchedNotes(unittest.TestCase):
    def test_get_matched_notes(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            perf, alignment, scr = partitura.load_match(
                filename=fn,
                create_score=True,
            )
            perf_note_array = perf.note_array()
            scr_note_array = scr.note_array()
            matched_idxs = performance_codec.get_matched_notes(
                spart_note_array=scr_note_array,
                ppart_note_array=perf_note_array,
                alignment=alignment,
            )
            scr_pitch = scr_note_array["pitch"][matched_idxs[:, 0]]
            perf_pitch = perf_note_array["pitch"][matched_idxs[:, 1]]

            self.assertTrue(np.all(scr_pitch == perf_pitch))


class TestGetTimeMapsFromAlignment(unittest.TestCase):
    def test_get_time_maps_from_alignment(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            note_ids = scr.note_array()["id"]
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)

            for bpm in beats_per_minute:
                ppart = music.performance_from_part(part=scr[0], bpm=bpm)
                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in note_ids
                ]

                (
                    ptime_to_stime_map,
                    stime_to_ptime_map,
                ) = performance_codec.get_time_maps_from_alignment(
                    spart_or_note_array=scr[0],
                    ppart_or_note_array=ppart,
                    alignment=alignment,
                    remove_ornaments=True,
                )

                score_onsets = np.arange(4, 0.5)
                performed_onsets = 60 / bpm * score_onsets

                self.assertTrue(
                    np.all(score_onsets == ptime_to_stime_map(performed_onsets))
                )
                self.assertTrue(
                    np.all(performed_onsets == stime_to_ptime_map(score_onsets))
                )


class TestPerformanceFromPart(unittest.TestCase):
    def test_performance_from_part(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)
            midi_velocity = RNG.randint(30, 127, size=2)
            for bpm, vel in zip(beats_per_minute, midi_velocity):
                ppart = music.performance_from_part(part=scr[0], bpm=bpm, velocity=vel)

                # assert that both objects have the same number of notes
                self.assertEqual(len(scr[0].notes_tied), len(ppart.notes))

                snote_array = scr[0].note_array()
                pnote_array = ppart.note_array()

                # check MIDI velocities
                self.assertTrue(np.all(pnote_array["velocity"] == vel))

                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in snote_array["id"]
                ]

                matched_idxs = performance_codec.get_matched_notes(
                    spart_note_array=snote_array,
                    ppart_note_array=pnote_array,
                    alignment=alignment,
                )

                # check pitch
                self.assertTrue(
                    np.all(
                        pnote_array["pitch"][matched_idxs[:, 1]]
                        == snote_array["pitch"][matched_idxs[:, 0]]
                    )
                )
                # check durations
                self.assertTrue(
                    np.allclose(
                        pnote_array["duration_sec"],
                        snote_array["duration_beat"] * (60 / bpm),
                    )
                )

                pnote_array = pnote_array[matched_idxs[:, 1]]
                snote_array = snote_array[matched_idxs[:, 0]]

                unique_onsets = np.unique(snote_array["onset_beat"])
                unique_onset_idxs = [
                    np.where(snote_array["onset_beat"] == uo)[0] for uo in unique_onsets
                ]

                # check performed onsets
                perf_onsets = np.array(
                    [
                        np.mean(pnote_array["onset_sec"][uix])
                        for uix in unique_onset_idxs
                    ]
                )

                beat_period = np.diff(perf_onsets) / np.diff(unique_onsets)

                # check that that the performance corresponds to the expected tempo
                self.assertTrue(np.allclose(60 / beat_period, bpm))

    def get_tempo_curve(self, score_onsets, performance_onsets):
        """
        Get tempo curve
        """
        unique_sonsets = np.unique(score_onsets)
        # Ensure that everything is sorted (I'm just paranoid ;)
        unique_sonsets.sort()
        unique_ponsets = np.unique(performance_onsets)
        # Ensure that everything is sorted
        unique_ponsets.sort()

        bp = np.diff(unique_ponsets) / np.diff(unique_sonsets)

        # Beats per minute for each of the unique onsets
        # the last bpm is just assuming that the tempo remains
        # constant after the last onset.
        bpm = np.r_[60 / bp, 60 / bp[-1]]

        return bpm

    def test_performance_notearray_from_score_notearray_bpm(self):
        """
        Test possibilities for bpm argument in
        utils.music.performance_notearray_from_score_notearray
        """
        score = partitura.load_score(MOZART_VARIATION_FILES["musicxml"])

        score_note_array = score.note_array()

        unique_onsets = np.unique(score_note_array["onset_beat"])
        unique_onsets.sort()
        # Test constant tempo
        bpm = 30
        velocity = 65
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=velocity,
        )

        self.assertTrue(
            np.allclose(
                self.get_tempo_curve(
                    score_note_array["onset_beat"],
                    perf_note_array["onset_sec"],
                ),
                bpm,
            )
        )

        # Test callable tempo
        def bpm_fun(onset):
            """
            Test function the first half of the piece will be played
            twice as fast
            """
            if isinstance(onset, (int, float)):
                onset = np.array([onset])

            bpm = np.zeros(len(onset), dtype=float)

            midpoint = (unique_onsets.max() - unique_onsets.min()) / 2
            bpm[np.where(onset <= midpoint)[0]] = 120
            bpm[np.where(onset > midpoint)[0]] = 60

            return bpm

        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm_fun,
            velocity=velocity,
        )

        bpm = self.get_tempo_curve(
            score_note_array["onset_beat"],
            perf_note_array["onset_sec"],
        )

        midpoint = (unique_onsets.max() - unique_onsets.min()) / 2

        self.assertTrue(
            np.allclose(
                bpm[np.where(unique_onsets <= midpoint)[0]],
                120,
            )
        )

        self.assertTrue(np.allclose(bpm[np.where(unique_onsets > midpoint)[0]], 60))

        # Test tempo as an array
        bpm_expected = 40 * RNG.rand(len(unique_onsets)) + 30

        # Test using 1d array
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=np.column_stack((unique_onsets, bpm_expected)),
            velocity=velocity,
        )

        bpm_predicted = self.get_tempo_curve(
            score_note_array["onset_beat"],
            perf_note_array["onset_sec"],
        )

        # do not consider the last element, since get_tempo_curve only computes
        # the tempo up to the last onset (otherwise offsets need to be considered)
        self.assertTrue(np.allclose(bpm_expected[:-1], bpm_predicted[:-1], atol=1e-3))

        try:
            # This should trigger an error because bpm_expected is a 1D array
            perf_note_array = music.performance_notearray_from_score_notearray(
                snote_array=score_note_array,
                bpm=bpm_expected,
                velocity=velocity,
            )
            self.assertTrue(False)

        except ValueError:
            # We are expecting the previous code to trigger an error
            self.assertTrue(True)

    def get_velocity_curves(self, velocity, score_onsets):
        """
        Get velocity curve by aggregating MIDI velocity values for
        each onset
        """
        unique_onsets = np.unique(score_onsets)
        # Ensure that everything is sorted (I'm just paranoid ;)
        unique_onsets.sort()

        unique_onset_idxs = [np.where(score_onsets == uo)[0] for uo in unique_onsets]
        velocity_curve = np.array([velocity[ui].mean() for ui in unique_onset_idxs])

        return velocity_curve

    def test_performance_notearray_from_score_notearray_velocity(self):
        """
        Test velocity arguments in
        utils.music.performance_notearray_from_score_notearray
        """
        score = partitura.load_score(MOZART_VARIATION_FILES["musicxml"])

        score_note_array = score.note_array()

        unique_onsets = np.unique(score_note_array["onset_beat"])
        unique_onsets.sort()

        # Test constant velocity
        bpm = 120
        velocity = 65
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=velocity,
        )

        self.assertTrue(all(perf_note_array["velocity"] == velocity))

        # Test callable velocity
        def vel_fun(onset):
            """
            Test function the first half of the piece will be played
            twice as loud
            """
            if isinstance(onset, (int, float)):
                onset = np.array([onset])

            vel = np.zeros(len(onset), dtype=float)

            midpoint = (unique_onsets.max() - unique_onsets.min()) / 2
            vel[np.where(onset <= midpoint)[0]] = 120
            vel[np.where(onset > midpoint)[0]] = 60

            return vel

        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            bpm=bpm,
            velocity=vel_fun,
        )

        vel = self.get_velocity_curves(
            perf_note_array["velocity"], score_note_array["onset_beat"]
        )

        midpoint = (unique_onsets.max() - unique_onsets.min()) / 2

        self.assertTrue(
            np.allclose(
                vel[np.where(unique_onsets <= midpoint)[0]],
                120,
            )
        )

        self.assertTrue(np.allclose(vel[np.where(unique_onsets > midpoint)[0]], 60))

        # Test tempo as an array
        vel_expected = np.round(40 * RNG.rand(len(unique_onsets)) + 30)

        # Test using 1d array
        perf_note_array = music.performance_notearray_from_score_notearray(
            snote_array=score_note_array,
            velocity=np.column_stack((unique_onsets, vel_expected)),
            bpm=bpm,
        )

        vel_predicted = self.get_velocity_curves(
            perf_note_array["velocity"],
            score_note_array["onset_beat"],
        )

        self.assertTrue(np.allclose(vel_expected, vel_predicted, atol=1e-3))

        try:
            # This should trigger an error because vel_expected is a 1D array
            perf_note_array = music.performance_notearray_from_score_notearray(
                snote_array=score_note_array,
                bpm=bpm,
                velocity=vel_expected,
            )
            self.assertTrue(False)

        except ValueError:
            # We are expecting the previous code to trigger an error
            self.assertTrue(True)

    def test_generate_random_performance_note_array(self):
        """
        Test `generate_random_performance_note_array` method
        """
        n_notes = 100
        duration = 15
        max_note_duration = 9
        min_note_duration = 1
        max_velocity = 75
        min_velocity = 30
        random_note_array = music.generate_random_performance_note_array(
            num_notes=n_notes,
            rng=1234,
            duration=duration,
            max_note_duration=max_note_duration,
            min_note_duration=min_note_duration,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            return_performance=False,
        )

        # Assert that the output is a numpy array
        self.assertTrue(isinstance(random_note_array, np.ndarray))
        # Test that the generated array has the specified number of notes
        self.assertTrue(len(random_note_array) == n_notes)

        offsets = random_note_array["onset_sec"] + random_note_array["duration_sec"]

        # Test that the note array has the specified duration
        self.assertTrue(np.isclose(offsets.max(), duration))

        # Test that the generated durations and velocities are within the
        # specified bounds
        self.assertTrue(np.all(random_note_array["duration_sec"] <= max_note_duration))
        self.assertTrue(np.all(random_note_array["duration_sec"] >= min_note_duration))
        self.assertTrue(np.all(random_note_array["velocity"] >= min_velocity))
        self.assertTrue(np.all(random_note_array["velocity"] <= max_velocity))

        # Test that the output is a Performance instance
        random_performance = music.generate_random_performance_note_array(
            num_notes=n_notes,
            duration=duration,
            max_note_duration=max_note_duration,
            min_note_duration=min_note_duration,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            return_performance=True,
        )

        self.assertTrue(
            isinstance(random_performance, partitura.performance.Performance)
        )

    def test_sliceperf(self):

        perf = partitura.load_performance_midi(MOZART_VARIATION_FILES["midi"])
        ppart = perf[0]
        ppart.sustain_pedal_threshold = 127

        note_array = ppart.note_array()

        start_time = 10
        end_time = 20

        idx = np.where(
            np.logical_and(
                note_array["onset_sec"] >= start_time,
                note_array["onset_sec"] <= end_time,
            )
        )

        target_note_array = note_array[idx]

        def test_arrays(clip_note_off, reindex_notes):

            # Test without clipping note offs
            ppart_slice = music.slice_ppart_by_time(
                ppart=ppart,
                start_time=start_time,
                end_time=end_time,
                clip_note_off=clip_note_off,
                reindex_notes=reindex_notes,
            )
            slice_note_array = ppart_slice.note_array()

            self.assertTrue(len(target_note_array) == len(slice_note_array))
            self.assertTrue(
                slice_note_array["onset_sec"].max() <= (end_time - start_time)
            )

            if clip_note_off:
                self.assertTrue(
                    (
                        slice_note_array["onset_sec"] + slice_note_array["duration_sec"]
                    ).max()
                    <= (end_time - start_time)
                )
            else:
                self.assertTrue(
                    (
                        slice_note_array["onset_sec"] + slice_note_array["duration_sec"]
                    ).max()
                    >= (end_time - start_time)
                )

            self.assertTrue(
                np.isclose(
                    target_note_array["onset_sec"].min() - start_time,
                    slice_note_array["onset_sec"].min(),
                )
            )
            self.assertTrue(
                np.isclose(
                    target_note_array["onset_sec"].max() - start_time,
                    slice_note_array["onset_sec"].max(),
                )
            )

            nidx = slice_note_array["id"]
            nidx.sort()

            if reindex_notes:
                tidx = np.array([f"n{idx}" for idx in range(len(nidx))])
            else:
                tidx = target_note_array["id"]

            tidx.sort()
            self.assertTrue(np.all(nidx == tidx))

        for cno in (True, False):
            for rin in (True, False):
                test_arrays(cno, rin)


class TestGenericUtils(unittest.TestCase):
    def test_interp1d(self):
        """
        Test `interp1d`
        """

        # Test that the we get the same results as with
        # scipy
        rng = np.random.RandomState(1984)

        x = rng.randn(100)
        y = 3 * x + 1

        sinterp = scinterp1d(x=x, y=y)

        pinterp = pinterp1d(x=x, y=y)

        y_scipy = sinterp(x)
        y_partitura = pinterp(x)

        self.assertTrue(np.all(y_scipy == y_partitura))

        # Test that we don't get an error with inputs
        # with length 1

        x = rng.randn(1)
        y = rng.randn(1)

        pinterp = pinterp1d(x=x, y=y)

        x_test = rng.randn(1000)

        y_partitura = pinterp(x_test)

        self.assertTrue(y_partitura.shape == x_test.shape)
        self.assertTrue(np.all(y_partitura == y))

        # setting the axis when the input has length 1
        x = rng.randn(1)
        y = rng.randn(1, 5)
        pinterp = pinterp1d(x=x, y=y)

        y_partitura = pinterp(x_test)

        self.assertTrue(y_partitura.shape == (len(x_test), y.shape[1]))
        self.assertTrue(np.all(y_partitura == y))

        # Test setting dtype of the output

        dtypes = (
            float,
            int,
            np.int8,
            np.int16,
            np.float32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
            # np.float128,
        )

        for dtype in dtypes:

            x = rng.randn(100)
            y = rng.randn(100)

            pinterp = pinterp1d(x=x, y=y, dtype=dtype)

            y_partitura = pinterp(x)
            # assert that the dtype of the array is correct
            self.assertTrue(y_partitura.dtype == dtype)
            # assert that the result is the same as casting the expected
            # output as the specified dtype
            self.assertTrue(np.allclose(y_partitura, y.astype(dtype)))

        # Test setting outputs of sizes larger than 1

        x = rng.randn(100)
        y = rng.randn(100, 2)

        sinterp = scinterp1d(
            x,
            y,
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

        pinterp = pinterp1d(
            x,
            y,
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.assertTrue(np.all(sinterp(x) == pinterp(x)))


if HAS_MIDITOK:
    # Only run these tests if miditok is installed
    class TestTokenizer(unittest.TestCase):
        def test_tokenize1(self):
            """Test the partitura tokenizer"""
            tokenizer = miditok.MIDILike()
            # produce tokens from the score with partitura
            pt_score = partitura.load_score(TOKENIZER_TESTFILES[0]["score"])
            pt_tokens = tokenize(pt_score, tokenizer)[0].tokens
            # produce tokens from the manually created MIDI file
            mtok_midi = miditoolkit.MidiFile(TOKENIZER_TESTFILES[0]["midi"])
            mtok_tokens = tokenizer(mtok_midi)[0].tokens
            # filter out velocity tokens
            pt_tokens = [tok for tok in pt_tokens if not tok.startswith("Velocity")]
            mtok_tokens = [tok for tok in mtok_tokens if not tok.startswith("Velocity")]
            self.assertTrue(pt_tokens == mtok_tokens)

        def test_tokenize2(self):
            """Test the partitura tokenizer"""
            tokenizer = miditok.REMI()
            # produce tokens from the score with partitura
            pt_score = partitura.load_score(TOKENIZER_TESTFILES[0]["score"])
            pt_tokens = tokenize(pt_score, tokenizer)[0].tokens
            # produce tokens from the manually created MIDI file
            mtok_midi = miditoolkit.MidiFile(TOKENIZER_TESTFILES[0]["midi"])
            mtok_tokens = tokenizer(mtok_midi)[0].tokens
            # filter out velocity tokens
            pt_tokens = [tok for tok in pt_tokens if not tok.startswith("Velocity")]
            mtok_tokens = [tok for tok in mtok_tokens if not tok.startswith("Velocity")]
            self.assertTrue(pt_tokens == mtok_tokens)


    class TestSymbolicDurationEstimator(unittest.TestCase):
        def test_estimate_symbolic_duration(self):
            """
            Test `estimate_symbolic_duration`
            """
            divs = 12
            quarters = 4
            expected = [
                {},
                {'type': '32nd', 'actual_notes': 3, 'normal_notes': 2},
                {'type': '16th', 'actual_notes': 3, 'normal_notes': 2},
                {'type': '16th', 'dots': 0},
                {'type': 'eighth', 'actual_notes': 3, 'normal_notes': 2},
                ({'type': '16th', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'eighth', 'dots': 0},
                ({'type': 'eighth', 'dots': 0}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'quarter', 'actual_notes': 3, 'normal_notes': 2},
                {'type': 'eighth', 'dots': 1},
                ({'type': 'eighth', 'dots': 0}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'eighth', 'dots': 1}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'quarter', 'dots': 0},
                ({'type': 'quarter', 'dots': 0}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'quarter', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'quarter', 'dots': 0}, {'type': '16th', 'dots': 0}),
                {'type': 'half', 'actual_notes': 3, 'normal_notes': 2},
                ({'type': 'quarter', 'dots': 0}, {'type': '16th', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'quarter', 'dots': 1},
                ({'type': 'quarter', 'dots': 1}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'quarter', 'dots': 1}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'quarter', 'dots': 2},
                ({'type': 'quarter', 'dots': 1}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'quarter', 'dots': 2}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'half', 'dots': 0},
                ({'type': 'half', 'dots': 0}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': '16th', 'dots': 0}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': '16th', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 0}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 0}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': 'quarter', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 1}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 0}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 0}, {'type': 'eighth', 'dots': 1}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'half', 'dots': 1},
                ({'type': 'half', 'dots': 1}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 1}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 1}, {'type': '16th', 'dots': 0}),
                ({'type': 'half', 'dots': 1}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 1}, {'type': '16th', 'dots': 0}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'half', 'dots': 2},
                ({'type': 'half', 'dots': 2}, {'type': '32nd', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 1}, {'type': 'quarter', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                {'type': 'half', 'dots': 3},
                ({'type': 'half', 'dots': 2}, {'type': 'eighth', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
                ({'type': 'half', 'dots': 3}, {'type': '16th', 'dots': 0, 'actual_notes': 3, 'normal_notes': 2}),
            ]
            predicted = []
            for k in range(divs * 0, divs * quarters):
                a = partitura.utils.estimate_symbolic_duration(
                    k, divs, return_com_durations=True)
                predicted.append(a)
            self.assertTrue(all([expected[i] == predicted[i] for i in range(len(expected))]))
            