#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Top level of the utilities module.
"""

from partitura.utils.generic import (
    ComparableMixin,
    ReplaceRefMixin,
    partition,
    iter_subclasses,
    iter_current_next,
    sorted_dict_items,
    PrettyPrintTree,
    find_nearest,
    add_field,
    show_diff,
    search,
    _OrderedSet,
)
from partitura.utils.music import (
    MIDI_BASE_CLASS,
    ALTER_SIGNS,
    find_tie_split,
    fifths_mode_to_key_name,
    key_name_to_fifths_mode,
    estimate_symbolic_duration,
    format_symbolic_duration,
    symbolic_to_numeric_duration,
    to_quarter_tempo,
    pitch_spelling_to_midi_pitch,
    estimate_clef_properties,
    ensure_pitch_spelling_format,
    ensure_notearray,
    compute_pianoroll,
    compute_pitch_class_pianoroll,
    pianoroll_to_notearray,
    match_note_arrays,
    key_mode_to_int,
    clef_sign_to_int,
    clef_int_to_sign,
    remove_silence_from_performed_part,
    note_array_from_part_list,
    slice_notearray_by_time,
    note_array_from_part,
    note_array_from_note_list,
    get_time_units_from_note_array,
    update_note_ids_after_unfolding,
    note_name_to_pitch_spelling,
    note_name_to_midi_pitch,
    pitch_spelling_to_note_name,
    rest_array_from_part,
    rest_array_from_rest_list,
    ensure_rest_array,
    rest_array_from_part_list,
)
from partitura.utils.synth import synthesize

from partitura.utils.misc import (
    PathLike,
    get_document_name,
    deprecated_alias,
    deprecated_parameter,
)

from partitura.utils.normalize import normalize

__all__ = [
    "ensure_notearray",
    "note_array_from_part",
    "ensure_rest_array",
    "compute_pianoroll",
    "compute_pitch_class_pianoroll",
    "pianoroll_to_notearray",
    "slice_notearray_by_time",
    "key_name_to_fifths_mode",
    "fifths_mode_to_key_name",
    "key_mode_to_int",
    "clef_sign_to_int",
    "clef_int_to_sign",
    "pitch_spelling_to_midi_pitch",
    "pitch_spelling_to_note_name",
    "show_diff",
    "PrettyPrintTree",
    "synthesize",
    "normalize",
]
