#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing MusicXML files.
"""

import os
import warnings
import zipfile

from typing import Union, Optional, List
import numpy as np
from lxml import etree


# lxml does XSD validation too but has problems with the MusicXML 3.1 XSD, so we use
# the xmlschema package for validating MusicXML against the definition
import xmlschema
import pkg_resources
from partitura.directions import parse_direction
import partitura.score as score
from partitura.score import assign_note_ids
from partitura.utils import ensure_notearray
from partitura.utils.misc import (
    deprecated_alias,
    deprecated_parameter,
    PathLike,
    parse_ints,
)

__all__ = ["load_musicxml", "musicxml_to_notearray"]

_MUSICXML_SCHEMA = pkg_resources.resource_filename("partitura", "assets/musicxml.xsd")
_XML_VALIDATOR = None
DYN_DIRECTIONS = {
    "f": score.ConstantLoudnessDirection,
    "ff": score.ConstantLoudnessDirection,
    "fff": score.ConstantLoudnessDirection,
    "ffff": score.ConstantLoudnessDirection,
    "fffff": score.ConstantLoudnessDirection,
    "ffffff": score.ConstantLoudnessDirection,
    "n": score.ConstantLoudnessDirection,
    "mf": score.ConstantLoudnessDirection,
    "mp": score.ConstantLoudnessDirection,
    "p": score.ConstantLoudnessDirection,
    "pp": score.ConstantLoudnessDirection,
    "ppp": score.ConstantLoudnessDirection,
    "pppp": score.ConstantLoudnessDirection,
    "ppppp": score.ConstantLoudnessDirection,
    "pppppp": score.ConstantLoudnessDirection,
    "fp": score.ImpulsiveLoudnessDirection,
    "pf": score.ImpulsiveLoudnessDirection,
    "rf": score.ImpulsiveLoudnessDirection,
    "rfz": score.ImpulsiveLoudnessDirection,
    "fz": score.ImpulsiveLoudnessDirection,
    "sf": score.ImpulsiveLoudnessDirection,
    "sffz": score.ImpulsiveLoudnessDirection,
    "sfp": score.ImpulsiveLoudnessDirection,
    "sfzp": score.ImpulsiveLoudnessDirection,
    "sfpp": score.ImpulsiveLoudnessDirection,
    "sfz": score.ImpulsiveLoudnessDirection,
}

PEDAL_DIRECTIONS = {
    "sustain_pedal": score.SustainPedalDirection,
}

TEMPO_DIRECTIONS = {
    "Adagio": score.ConstantTempoDirection,
    "Andante": score.ConstantTempoDirection,
    "Andante amoroso": score.ConstantTempoDirection,
    "Andante cantabile": score.ConstantTempoDirection,
    "Andante grazioso": score.ConstantTempoDirection,
    "Menuetto": score.ConstantTempoDirection,
    "Allegretto grazioso": score.ConstantTempoDirection,
    "Allegro moderato": score.ConstantTempoDirection,
    "Allegro assai": score.ConstantTempoDirection,
    "Allegro": score.ConstantTempoDirection,
    "Allegretto": score.ConstantTempoDirection,
    "Molto allegro": score.ConstantTempoDirection,
    "Presto": score.ConstantTempoDirection,
}

OCTAVE_SHIFTS = {8: 1, 15: 2, 22: 3}

ACCIDENTAL_MAP = {
    "sharp": 1,
    "natural": 0,
    "flat": -1,
    "double-sharp": 2,
    "double-flat": -2,
}


def validate_musicxml(xml, debug=False):
    """
    Validate an XML file against an XSD.

    Parameters
    ----------
    xml: str
        Path to XML file
    debug: bool, optional
        If True, raise an exception when the xml is invalid, and print out the
        cause. Otherwise just return True when the XML is valid and False otherwise

    Returns
    -------
    bool or None
        None if debug=True, True or False otherwise, signalling validity

    """
    global _XML_VALIDATOR
    if not _XML_VALIDATOR:
        _XML_VALIDATOR = xmlschema.XMLSchema(_MUSICXML_SCHEMA)
    if debug:
        return _XML_VALIDATOR.validate(xml)
    else:
        return _XML_VALIDATOR.is_valid(xml)


def _parse_partlist(partlist):
    """
    This parses the <part-list> ... </part-list> element in the beginning
    of the MusicXML file where each instrument is declared, instruments and
    their staves are grouped (braces, brackets), etc.

    Parameters
    ----------
    partlist : etree element
        The part-list etree element

    Returns
    -------
    list:
        list of PartGroup objects
    dict:
        Dictionary of pairs (partid, Part) where the Part objects are
        instantiated with part-name and part-abbreviation if these are specified in
        the part list definition.
    """

    structure = []
    current_group = None
    part_dict = {}
    for e in partlist:
        if e.tag == "part-group":
            if e.get("type") == "start":
                gr_name = get_value_from_tag(e, "group-name", str)
                gr_symbol = get_value_from_tag(e, "group-symbol", str)
                gr_number = get_value_from_attribute(e, "number", int)
                new_group = score.PartGroup(gr_symbol, gr_name, gr_number)

                if current_group is None:
                    current_group = new_group
                else:
                    current_group.children.append(new_group)
                    new_group.parent = current_group
                    current_group = new_group

            elif e.get("type") == "stop":
                if current_group.parent is None:
                    structure.append(current_group)
                    current_group = None
                else:
                    current_group = current_group.parent

        elif e.tag == "score-part":
            part_id = e.get("id")
            part = score.Part(part_id)

            # set part name and abbreviation if available
            part.part_name = next(iter(e.xpath("part-name/text()")), None)
            part.part_abbreviation = next(
                iter(e.xpath("part-abbreviation/text()")), None
            )
            part_dict[part_id] = part

            if current_group is None:
                structure.append(part)
            else:
                current_group.children.append(part)
                part.parent = current_group

    if current_group is not None:
        warnings.warn("part-group {0} was not ended".format(current_group.number))
        structure.append(current_group)

    return structure, part_dict


@deprecated_alias(xml="filename")
@deprecated_parameter("ensure_list")
def load_musicxml(
    filename: PathLike,
    validate: bool = False,
    force_note_ids: Optional[Union[bool, str]] = None,
    ignore_invisible_objects: bool = False,
) -> score.Score:
    """Parse a MusicXML file and build a composite score ontology
    structure from it (see also scoreontology.py).

    Parameters
    ----------
    xml : str or file-like  object
        Path to the MusicXML file to be parsed, or a file-like object
    validate : bool, optional
        When True the validity of the MusicXML is checked against the
        MusicXML 3.1 specification before loading the file. An
        exception will be raised when the MusicXML is invalid.
        Defaults to False.
    force_note_ids : (bool, 'keep') optional.
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the MusicXML will be discarded. If 'keep', only notes without
        a note id will be assigned one.
    ignore_invisible_objects : bool, optional
        When True, objects that with the attribute `print-object="no"`
        will be ignored. Defaults to False.

    Returns
    -------
    scr: Score
        A `Score` instance.

    """
    # NOTE: raising warning for ignore_invisible_objects is not ideal and it should be changed in the future
    if ignore_invisible_objects:
        warnings.warn(
            "Be advised that the 'ignore_invisible_objects' option might sometimes lead "
            "to parsing errors for unusual musicxml files. \n"
            "Note that when ignore_invisible_objects is False (the default), the parsing works as expected as before."
        )

    xml = None
    if isinstance(filename, str):
        if zipfile.is_zipfile(filename):
            with zipfile.ZipFile(filename) as zipped_xml:
                contained_xml_name = zipped_xml.namelist()[-1]
                xml = zipped_xml.open(contained_xml_name)

    if xml is None:
        xml = filename

    if validate:
        validate_musicxml(xml, debug=True)
        # if xml is a file-like object we need to set the read pointer to the
        # start of the file for parsing
        if hasattr(xml, "seek"):
            xml.seek(0)

    parser = etree.XMLParser(
        resolve_entities=False,
        huge_tree=False,
        remove_comments=True,
        remove_blank_text=True,
    )
    document = etree.parse(xml, parser)

    if document.getroot().tag != "score-partwise":
        raise Exception("Currently only score-partwise structure is supported")

    partlist_el = document.find("part-list")

    if partlist_el is not None:
        # parse the (hierarchical) structure of score parts
        # (instruments) that are listed in the part-list element
        partlist, part_dict = _parse_partlist(partlist_el)
        # Go through each <part> to obtain the content of the parts.
        # The Part instances will be modified in place
        _parse_parts(
            document, part_dict, ignore_invisible_objects=ignore_invisible_objects
        )
    else:
        partlist = []

    if force_note_ids is True or force_note_ids == "keep":
        assign_note_ids(partlist, force_note_ids == "keep")

    composer = None
    scid = None
    work_title = None
    work_number = None
    movement_title = None
    movement_number = None
    title = None
    subtitle = None
    lyricist = None
    copyright = None

    # The work tag is preferred for the title of the score, otherwise
    # this method will search in the credit tags
    work_info_el = document.find("work")

    if work_info_el is not None:
        scid = get_value_from_tag(
            e=work_info_el,
            tag="work-title",
            as_type=str,
        )
        scidn = get_value_from_tag(
            e=work_info_el,
            tag="work-number",
            as_type=str,
        )
        work_title = scid
        work_number = scidn

    movement_title_el = document.find(".//movement-title")
    movement_number_el = document.find(".//movement-number")
    if movement_title_el is not None:
        movement_title = movement_title_el.text
    if movement_number_el is not None:
        movement_number = movement_number_el.text

    score_identification_el = document.find("identification")

    # The identification tag has preference over credit
    if score_identification_el is not None:
        copyright = get_value_from_tag(score_identification_el, "rights", str)

        for cel in score_identification_el.findall("creator"):
            if get_value_from_attribute(cel, "type", str) == "composer":
                composer = str(cel.text)
            if get_value_from_attribute(cel, "type", str) == "lyricist":
                lyricist = str(cel.text)

    for cel in document.findall("credit"):
        credit_type = get_value_from_tag(cel, "credit-type", str)
        if credit_type == "title" and title is None:
            title = get_value_from_tag(cel, "credit-words", str)

        elif credit_type == "subtitle" and subtitle is None:
            subtitle = get_value_from_tag(cel, "credit-words", str)

        elif credit_type == "composer" and composer is None:
            composer = get_value_from_tag(cel, "credit-words", str)

        elif credit_type == "lyricist" and lyricist is None:
            lyricist = get_value_from_tag(cel, "credit-words", str)

        elif credit_type == "rights" and copyright is None:
            copyright = get_value_from_tag(cel, "credit-words", str)

    scr = score.Score(
        id=scid,
        partlist=partlist,
        work_number=work_number,
        work_title=work_title,
        movement_number=movement_number,
        movement_title=movement_title,
        title=title,
        subtitle=subtitle,
        composer=composer,
        lyricist=lyricist,
        copyright=copyright,
    )

    return scr


def _parse_parts(document, part_dict, ignore_invisible_objects=False):
    """
    Populate the Part instances that are the values of `part_dict` with the
    musical content in document.

    Parameters
    ----------
    document : lxml.etree.ElementTree
        The ElementTree representation of the MusicXML document
    part_dict : dict
        A dictionary with key--value pairs (part_id, Part instance), as returned
        by the _parse_partlist() function.
    ignore_invisible_objects : bool, optional
        When True, objects that with the attribute `print-object="no"` will be ignored.
    """

    for part_el in document.findall("part"):
        part_id = part_el.get("id", "P1")
        part = part_dict.get(part_id, score.Part(part_id))

        position = 0
        ongoing = {}
        doc_order = 0
        # add new page and system at start of part
        _handle_new_page(position, part, ongoing)
        _handle_new_system(position, part, ongoing)

        for mc, measure_el in enumerate(part_el.xpath("measure")):
            position, doc_order = _handle_measure(
                measure_el,
                position,
                part,
                ongoing,
                doc_order,
                mc + 1,
                ignore_invisible_objects,
            )

        # complete unfinished endings
        for o in part.iter_all(score.Ending, mode="ending"):
            if o.start is None:
                part.add(o, part.measure_map(o.end.t - 1)[0], None)
                warnings.warn(
                    "Found ending[stop] without a preceding ending[start]\n"
                    "Single measure bracket is assumed"
                )

        for o in part.iter_all(score.Ending, mode="starting"):
            if o.end is None:
                part.add(o, None, part.measure_map(o.start.t)[1])
                warnings.warn(
                    "Found ending[start] without a following ending[stop]\n"
                    "Single measure bracket is assumed"
                )

        # Complete repeats without end.
        volta_repeats = list()
        for o in part.iter_all(score.Repeat, mode="starting"):
            if o.end is None:
                # if len(o.start.starting_objects[score.Repeat]) > 0:
                #     starting = list(o.start.starting_objects[score.Repeat].keys())[0]
                #     # if unstarted repeat from volta, continue for now
                #     if len(starting.end.ending_objects[score.Repeat]) > 0:
                #         # if repeat from volta, continue for now
                #         volta_repeats.append(o)
                #         continue

                starting_repeats = [
                    r for r in part.iter_all(score.Repeat) if r.start is not None
                ]
                end_times = [r.start.t for r in starting_repeats] + [part._points[-1].t]
                end_time_id = np.searchsorted(end_times, o.start.t + 1)
                part.add(o, None, end_times[end_time_id])
                warnings.warn(
                    "Found repeat without end\n"
                    "Ending point {} is assumed".format(end_times[end_time_id])
                )

        # complete unstarted repeats
        volta_repeats = list()
        for o in part.iter_all(score.Repeat, mode="ending"):
            if o.start is None:
                if len(o.end.ending_objects[score.Ending]) > 0:
                    ending = list(o.end.ending_objects[score.Ending].keys())[0]
                    # if unstarted repeat from volta, continue for now
                    if len(ending.start.ending_objects[score.Repeat]) > 0:
                        volta_repeats.append(o)
                        continue

                # go back to the end of the last repeat
                start_times = [0] + [r.end.t for r in part.iter_all(score.Repeat)]
                start_time_id = np.searchsorted(start_times, o.end.t) - 1
                part.add(o, start_times[start_time_id], None)
                warnings.warn(
                    "Found repeat without start\n"
                    "Starting point {} is assumed".format(start_times[start_time_id])
                )

        # complete unstarted repeats in volta with start time of first repeat
        for o in volta_repeats:
            start_times = [0] + [r.start.t for r in part.iter_all(score.Repeat)]
            start_time_id = np.searchsorted(start_times, o.end.t) - 1
            part.add(o, start_times[start_time_id], None)
            warnings.warn(
                "Found repeat without start\n"
                "Starting point {} is assumed".format(start_times[start_time_id])
            )

        # remove unfinished elements from the timeline
        for k, o in ongoing.items():
            if k not in ("page", "system", "repeat") and k[0] not in ("tie", "ending"):
                if isinstance(o, list):
                    for o_i in o:
                        part.remove(o_i)
                else:
                    part.remove(o)

        # check whether all grace notes have a main note
        for gn in part.iter_all(score.GraceNote):
            if gn.main_note is None:
                for no in part.iter_all(
                    score.Note,
                    include_subclasses=False,
                    start=gn.start.t,
                    end=gn.start.t + 1,
                ):
                    if no.voice == gn.voice:
                        gn.last_grace_note_in_seq.grace_next = no

            if gn.main_note is None:
                warnings.warn(
                    "grace note without recoverable same voice main note: {}".format(gn)
                )
                warnings.warn("might be cadenza notation")

        # set end times for various musical elements that only have a start time
        # when constructed from MusicXML
        score.set_end_times(part)

        # Octave Shifts are only visual doesn't to change the octave of notes.
        # Apply octave shifts directly to notes
        # for shift in part.iter_all(score.OctaveShiftDirection):
        #     # Shifts normal notes
        #     for note in part.iter_all(score.Note, start=shift.start.t, end=shift.end.t):
        #         if note.staff == shift.staff:
        #             if shift.shift_type == "up":
        #                 note.octave -= OCTAVE_SHIFTS[shift.shift_size] if shift.shift_size in OCTAVE_SHIFTS.keys() else 0
        #             elif shift.shift_type == "down":
        #                 note.octave += OCTAVE_SHIFTS[shift.shift_size] if shift.shift_size in OCTAVE_SHIFTS.keys() else 0
        #     # Shifts grace notes
        #     for note in part.iter_all(score.GraceNote, start=shift.start.t, end=shift.end.t):
        #         if note.staff == shift.staff:
        #             if shift.shift_type == "up":
        #                 note.octave -= OCTAVE_SHIFTS[shift.shift_size] if shift.shift_size in OCTAVE_SHIFTS.keys() else 0
        #             elif shift.shift_type == "down":
        #                 note.octave += OCTAVE_SHIFTS[shift.shift_size] if shift.shift_size in OCTAVE_SHIFTS.keys() else 0
        #     shift.applied = True


def _handle_measure(
    measure_el,
    position,
    part,
    ongoing,
    doc_order,
    measure_counter,
    ignore_invisible_objects=False,
):
    """Parse a <measure>...</measure> element, adding it and its contents to the part.

    Parameters
    ----------
    measure_el : etree.Element
        An etree Element instance with a <measure> tag
    position : int
        The starting time of the measure
    part : score.Part
        The part object to which the measure belongs.
    ongoing : dict
        A dict specifying the score.Page and score.System of the measure
    doc_order : int
        The index of the first note element in the current measure in the xml file.
    measure_counter : int
        The index of the <measure> tag in the xml file, starting from 1
    ignore_invisible_objects : bool, optional
        When True, objects that with the attribute `print-object="no"` will be ignored.

    Returns
    -------
    measure_maxtime : int
        The ending time of the measure
    doc_order : int
        The index of the first note element in the next measure in the xml file.

    """

    # make a measure object
    measure = make_measure(measure_el, measure_counter)

    # add the start of the measure to the time line
    part.add(measure, position)

    # Initialize Beams in Measure
    prev_beam = None
    # keep track of the position within the measure
    # measure_pos = 0
    measure_start = position
    # keep track of the previous note (in case of <chord>)
    prev_note = None
    # used to keep track of the duration of the measure
    measure_maxtime = measure_start
    trailing_children = []
    for i, e in enumerate(measure_el):
        # If the object is invisible and the user wants it, skip the object
        # Will probably not skip everything, but works at least for notes and rests
        if ignore_invisible_objects:
            print_obj = get_value_from_attribute(e, "print-object", str)
            notehead = e.find("notehead")  # Musescore mask notes with notehead="none"
            if print_obj == "no" or (notehead is not None and notehead.text == "none"):
                # Still update position for invisible notes (to avoid problems with backups)
                if e.tag == "note":
                    duration = get_value_from_tag(e, "duration", int) or 0
                    position += duration
                # Skip the object
                continue

        if e.tag == "backup":
            # <xs:documentation>The backup and forward elements are required
            # to coordinate multiple voices in one part, including music on
            # multiple staves. The backup type is generally used to move
            # between voices and staves. Thus the backup element does not
            # include voice or staff elements. Duration values should always
            # be positive, and should not cross measure boundaries or
            # mid-measure changes in the divisions value.</xs:documentation>

            duration = get_value_from_tag(e, "duration", int) or 0
            position -= duration

            if position < measure.start.t:
                warnings.warn(
                    (
                        "<backup> crosses measure boundary, adjusting "
                        "position from {} to {} in Measure {}"
                    ).format(position, measure.start.t, measure.number)
                )
                position = measure.start.t

            # <backup> tags trigger an update of the measure
            # duration up to the measure position (after the
            # <backup> has been processed); This has been found to
            # account for implicit measure durations in
            # Baerenreiter MusicXML files.
            measure_maxtime = max(measure_maxtime, position)

        elif e.tag == "forward":
            duration = get_value_from_tag(e, "duration", int) or 0
            position += duration
            measure_maxtime = max(measure_maxtime, position)

        elif e.tag == "attributes":
            _handle_attributes(e, position, part)

        elif e.tag == "direction":
            _handle_direction(e, position, part, ongoing)

        elif e.tag == "print":
            # new-page/new-system occurring anywhere in the measure take effect
            # at the start of the measure, so we pass measure_start rather than
            # position
            _handle_print(e, measure_start, part, ongoing)

        elif e.tag == "sound":
            _handle_sound(e, position, part)

        elif e.tag == "note":
            (position, prev_note, prev_beam) = _handle_note(
                e, position, part, ongoing, prev_note, doc_order, prev_beam
            )
            doc_order += 1
            measure_maxtime = max(measure_maxtime, position)

        elif e.tag == "harmony":
            _handle_harmony(e, position, part)

        elif e.tag == "barline":
            location = get_value_from_attribute(e, "location", str)
            if (location is None) or (location == "right"):
                position_barline = measure_maxtime
            elif location == "left":
                position_barline = measure_start
            else:
                position_barline = position

            repeat = e.find("repeat")
            if repeat is not None:
                _handle_repeat(repeat, position_barline, part, ongoing)

            ending = e.find("ending")
            if ending is not None:
                _handle_ending(ending, position_barline, part, ongoing)

            bar_style_e = e.find("bar-style")
            if bar_style_e is not None:
                bar_style = score.Barline(bar_style_e.text)
                part.add(bar_style, position)

            # <!ELEMENT barline (bar-style?, %editorial;, wavy-line?,
            #     segno?, coda?, (fermata, fermata?)?, ending?, repeat?)>
            # <!ATTLIST barline
            #     location (right | left | middle) "right"
            #     segno CDATA #IMPLIED
            #     coda CDATA #IMPLIED
            #     divisions CDATA #IMPLIED

            fermata_e = e.find("fermata")
            if fermata_e is not None:
                location = e.get("location")
                fermata = score.Fermata(location)
                if location is None:
                    # missing location attribute on barline defaults to
                    # "right". In this case the barline should occur as the last
                    # element in the measure
                    trailing_children.append(fermata)
                else:
                    part.add(fermata, position)

            # TODO: handle segno/fine/dacapo

        else:
            warnings.warn("ignoring tag {0}".format(e.tag), stacklevel=2)

    for obj in trailing_children:
        part.add(obj, measure_maxtime)

    # add end time of measure
    part.add(measure, None, measure_maxtime)

    return measure_maxtime, doc_order


def _handle_harmony(e, position, part):
    """Handle a <harmony> element."""
    if e.find("function") is not None:
        text = e.find("function").text
        if text is not None:
            if "|" in text:
                text = text.split("|")
                if len(text) > 2:
                    warnings.warn(
                        f"Ignoring multiple cadence annotations {text[2:]}",
                        stacklevel=2,
                    )
                text, cadence_annotation = text[0], text[1]
                part.add(score.Cadence(cadence_annotation), position)
            part.add(score.RomanNumeral(text), position)
    elif e.find("kind") is not None and e.find("root") is not None:
        # TODO: handle kind text which is other kind of annotation also root
        kind = e.find("kind").get("text")
        root = e.find("root").find("root-step").text
        part.add(score.ChordSymbol(root=root, kind=kind), position)
        text = None
    else:
        text = None

    if text is None:
        warnings.warn("ignoring empty <harmony> tag", stacklevel=2)


def _handle_repeat(e, position, part, ongoing):
    key = "repeat"

    if e.get("direction") == "forward":
        o = score.Repeat()
        ongoing[key] = o
        part.add(o, position)

    elif e.get("direction") == "backward":
        o = ongoing.pop(key, None)

        if o is None:
            # implicit repeat start: create Repeat
            # object and add it at the beginning of
            # the self retroactively
            o = score.Repeat()
            # no, clean up later with shortest possible repeat start
            # part.add(o, 0)

        part.add(o, None, position)


def _handle_ending(e, position, part, ongoing):
    # key = "ending"
    key = ("ending", getattr(e, "number", "0"))

    if e.get("type") == "start":
        o = score.Ending(e.get("number"))
        ongoing[key] = o
        part.add(o, position)

    elif e.get("type") in ("stop", "discontinue"):
        o = ongoing.pop(key, None)

        if o is None:
            warnings.warn(
                "Found ending[stop] without a preceding ending[start]\n"
                + "Single measure bracket is assumed"
            )
            o = score.Ending(e.get("number"))
            part.add(o, None, position)

        else:
            part.add(o, None, position)


def _handle_new_page(position, part, ongoing):
    if "page" in ongoing:
        if position == 0:
            # ignore non-informative new-page at start of score
            return

        part.add(ongoing["page"], None, position)
        page_nr = ongoing["page"].number + 1
    else:
        page_nr = 1

    page = score.Page(page_nr)
    part.add(page, position)
    ongoing["page"] = page


def _handle_new_system(position, part, ongoing):
    if "system" in ongoing:
        if position == 0:
            # ignore non-informative new-system at start of score
            return

        # end current page
        part.add(ongoing["system"], None, position)
        system_nr = ongoing["system"].number + 1
    else:
        system_nr = 1

    system = score.System(system_nr)
    part.add(system, position)
    ongoing["system"] = system


def make_measure(xml_measure, measure_counter):
    """
    Parameters
    ----------
    xml_measure : etree.Element
        An etree Element instance with a <measure> tag
    measure_counter : int
        The index of the <measure> tag in the xml file, starting from 1

    Returns
    -------
    measure : score.Measure
        A measure object with a number and optional name attribute
    """

    measure = score.Measure()

    measure.number = measure_counter
    measure.name = get_value_from_attribute(xml_measure, "number", str)

    return measure


def _handle_attributes(e, position, part):
    """"""

    ts_num = get_value_from_tag(e, "time/beats", int)
    ts_den = get_value_from_tag(e, "time/beat-type", int)
    if ts_num and ts_den:
        part.add(score.TimeSignature(ts_num, ts_den), position)

    fifths = get_value_from_tag(e, "key/fifths", int)
    mode = get_value_from_tag(e, "key/mode", str)
    if fifths is not None or mode is not None:
        part.add(score.KeySignature(fifths, mode), position)

    diat = get_value_from_tag(e, "transpose/diatonic", int)
    chrom = get_value_from_tag(e, "transpose/chromatic", int)
    if diat is not None or chrom is not None:
        part.add(score.Transposition(diat, chrom), position)

    divs = get_value_from_tag(e, "divisions", int)
    if divs:
        # part.add(score.Divisions(divs), position)
        part.set_quarter_duration(position, divs)

    clefs = get_clefs(e)
    for clef in clefs:
        part.add(score.Clef(**clef), position)


def get_offset(e):
    offset = e.find("offset")

    if offset is None:
        return None

    else:
        sounding = offset.attrib.get("sound", "no")
        return int(offset.text) if sounding == "yes" else 0


def _handle_direction(e, position, part, ongoing):
    # <!--
    #     A direction is a musical indication that is not attached
    #     to a specific note. Two or more may be combined to
    #     indicate starts and stops of wedges, dashes, etc.

    #     By default, a series of direction-type elements and a
    #     series of child elements of a direction-type within a
    #     single direction element follow one another in sequence
    #     visually. For a series of direction-type children, non-
    #     positional formatting attributes are carried over from
    #     the previous element by default.
    # -->
    # <!ELEMENT direction (direction-type+, offset?,
    #     %editorial-voice;, staff?, sound?)>
    # <!ATTLIST direction
    #     %placement;
    #     %direction;
    # >

    staff = get_value_from_tag(e, "staff", int) or None

    sound_directions = e.findall("sound")
    for sd in sound_directions:
        # TODO: figure out which values occur in the wild for each attribute
        if get_value_from_attribute(sd, "fine", str) in ("yes", "1"):
            part.add(score.Fine(), position)
        if get_value_from_attribute(sd, "dacapo", str) == "yes":
            part.add(score.DaCapo(), position)
        if get_value_from_attribute(sd, "tocoda", str) == "coda":
            part.add(score.ToCoda(), position)
        if get_value_from_attribute(sd, "coda", str) == "coda":
            part.add(score.Coda(), position)
        if get_value_from_attribute(sd, "dalsegno", str) == "segno":
            part.add(score.DalSegno(), position)
        if get_value_from_attribute(sd, "segno", str) == "segno":
            part.add(score.Segno(), position)

    # <direction-type> ... </...>
    direction_types = e.findall("direction-type")
    # <!ELEMENT direction-type (rehearsal+ | segno+ | words+ |
    #     coda+ | wedge | dynamics+ | dashes | bracket | pedal |
    #     metronome | octave-shift | harp-pedals | damp | damp-all |
    #     eyeglasses | string-mute | scordatura | image |
    #     principal-voice | accordion-registration | percussion+ |
    #     other-direction)>

    # direction-types supported here:
    # * words
    # * wedge
    # * dynamics
    # * dashes
    # * coda
    # * TODO: pedal
    # * TODO: damp

    # here we gather all starting and ending directions,
    # to be added to the part afterwards
    starting_directions = []
    ending_directions = []

    # keep track of starting and stopping dashes
    dashes_keys = {}

    # keep track of starting and stopping pedals
    # pedal_keys = {}

    for direction_type in direction_types:
        # direction_type
        dt = next(iter(direction_type))

        if dt.tag == "dynamics":
            # first child of direction-type is dynamics, there may be subsequent
            # dynamics items, so we loop:
            for child in direction_type:
                # check if child has no children, in which case continue
                # interpret as score.Direction, fall back to score.Words
                dyn_el = next(iter(child), None)
                if dyn_el is not None:
                    direction = DYN_DIRECTIONS.get(dyn_el.tag, score.Words)(
                        dyn_el.tag, staff=staff
                    )
                    starting_directions.append(direction)

        elif dt.tag == "words":
            # first child of direction-type is words, there may be subsequent
            # words items, so we loop:
            for child in direction_type:
                # try to make a direction out of words
                parse_result = parse_direction(child.text)
                starting_directions.extend(parse_result)

        elif dt.tag == "wedge":
            number = get_value_from_attribute(dt, "number", int) or 1
            key = ("wedge", number)
            wedge_type = get_value_from_attribute(dt, "type", str)

            if wedge_type in ("crescendo", "diminuendo"):
                if wedge_type == "crescendo":
                    o = score.IncreasingLoudnessDirection(wedge_type, wedge=True)
                else:
                    o = score.DecreasingLoudnessDirection(wedge_type, wedge=True)
                starting_directions.append(o)
                ongoing[key] = o

            elif wedge_type == "stop":
                o = ongoing.get(key)
                if o is not None:
                    ending_directions.append(o)
                    del ongoing[key]
                else:
                    warnings.warn("Did not find a wedge start element for wedge stop!")

        elif dt.tag == "dashes":
            # start/stop/continue
            dashes_type = get_value_from_attribute(dt, "type", str)
            number = get_value_from_attribute(dt, "number", int) or 1
            dashes_keys[("dashes", number)] = dashes_type
            # TODO: for now we ignore dashes_type == continue, because it exists
            # only as a function of the visual appearance. However, if dashes
            # that are continued over a system are stopped at the end of the
            # system before they are continued at the start of the next, this
            # will not be treated correctly. I'm not sure how dashes spanning
            # systems are encoded in practice (need examples).
        elif dt.tag == "octave-shift":
            # start/stop
            octave_shift_type = get_value_from_attribute(dt, "type", str)
            octave_shift_size = get_value_from_attribute(dt, "size", int) or 8
            octave_shift_number = get_value_from_attribute(dt, "number", int) or 1
            key = ("octave-shift", octave_shift_number)
            if octave_shift_type in ["up", "down"]:
                if key in ongoing:
                    eo = ongoing.pop(key)
                    ending_directions.append(eo)

                o = score.OctaveShiftDirection(
                    octave_shift_type, octave_shift_size, staff=staff
                )

                starting_directions.append(o)

                ongoing[key] = o

            elif octave_shift_type == "stop":
                o = ongoing.get(key)
                if o is not None:
                    ending_directions.append(o)
                    del ongoing[key]

                else:
                    warnings.warn(
                        "Did not find a octave_shift start element for octave_shift stop!"
                    )

        elif dt.tag == "pedal":
            # start/stop
            pedal_type = get_value_from_attribute(dt, "type", str)
            pedal_line = get_value_from_attribute(dt, "line", str) == "yes"

            number = get_value_from_attribute(dt, "number", int) or 1
            key = ("pedal", number)
            if pedal_type == "start":
                if key in ongoing:
                    eo = ongoing.pop(key)
                    ending_directions.append(eo)

                o = score.SustainPedalDirection(staff=staff, line=pedal_line)

                starting_directions.append(o)

                ongoing[key] = o

            elif pedal_type == "stop":
                o = ongoing.get(key)
                if o is not None:
                    ending_directions.append(o)
                    del ongoing[key]

                else:
                    warnings.warn("Did not find a pedal start element for pedal stop!")

            else:
                if pedal_type in ("change", "continue"):
                    warnings.warn(
                        'pedal types "change" and "continue" are '
                        "not supported. Ignoring direction."
                    )

        else:
            warnings.warn("ignoring direction type: {} {}".format(dt.tag, dt.attrib))

    for dashes_key, dashes_type in dashes_keys.items():
        if dashes_type == "start":
            ongoing[dashes_key] = starting_directions

        elif dashes_type == "stop":
            oo = ongoing.get(dashes_key)
            if oo is None:
                warnings.warn("Dashes end without dashes start")
            else:
                ending_directions.extend(oo)
                del ongoing[dashes_key]

    for o in starting_directions:
        if isinstance(o, score.Tempo):
            _add_tempo_if_unique(position, part, o)
        else:
            part.add(o, position)

    for o in ending_directions:
        part.add(o, None, position)


def get_clefs(e):
    """
    Get the clefs

    Returns
    -------

    (fifths, mode) OR None : tuple of or None.

    """
    clefs = e.xpath("clef")
    result = []
    for clef in clefs:
        result.append(
            dict(
                staff=get_value_from_attribute(clef, "number", int) or 1,
                sign=get_value_from_tag(clef, "sign", str),
                line=get_value_from_tag(clef, "line", int),
                octave_change=get_value_from_tag(clef, "clef-octave-change", int),
            )
        )
    return result


def get_value_from_tag(e, tag, as_type, none_on_error=True):
    """
    Return the text contents of a particular tag in element e, cast as a
    particular type. By default the function will return None if either the tag
    is not found or the value cannot be cast to the desired type.

    Examples
    --------

    >>> e = lxml.etree.fromstring('<note><duration>2</duration></note>')
    >>> get_value_from_tag(e, 'duration', int)
    2
    >>> get_value_from_tag(e, 'duration', float)
    2.0

    >>> e = lxml.etree.fromstring('<note><duration>quarter</duration></note>')
    >>> get_value_from_tag(e, 'duration', float)
    None


    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    tag: string
        Child tag to retrieve
    as_type: function
        Function that casts the string to the desired type (e.g. int, float)
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found in `e` or the
        text inside `tag` cannot be cast to an integer. When True, None is returned
        in such cases.

    Returns
    -------
    object
        The value read from `tag`, cast as `as_type`
    """
    try:
        return as_type(e.find(tag).text)
    except (ValueError, AttributeError):
        if none_on_error:
            return None
        else:
            raise


def get_value_from_attribute(e, attr, as_type, none_on_error=True):
    """
    Return the attribute of a particular attribute of element e, cast as a
    particular type. By default the function will return None if either `e` does
    not have the attribute or the value cannot be cast to the desired type.

    Parameters
    ----------
    e: etree.Element
        An etree Element instance
    attr: string
        Attribute to retrieve
    as_type: function
        Function that casts the string to the desired type (e.g. int, float)
    none_on_error: bool, optional (default: True)
        When False, an exception is raised when `tag` is not found in `e` or the
        text inside `tag` cannot be cast to an integer. When True, None is returned
        in such cases.

    Returns
    -------
    object or None
        The attribute value, or None
    """

    value = e.get(attr)
    if value is None:
        return None
    else:
        try:
            return as_type(value)
        except ValueError:
            if none_on_error:
                return None
            else:
                raise


def get_pitch(e):
    """
    Check whether the element has a pitch. If so return a tuple (pitch, alter,
    octave), otherwise return None.

    Returns
    -------
    tuple : (str, int or None, int) or None
        The tuple contains (pitch, alter, octave)
    """

    pitch = e.find("pitch")
    if pitch is not None:
        step = get_value_from_tag(pitch, "step", str)
        alter = get_value_from_tag(pitch, "alter", int)
        octave = get_value_from_tag(pitch, "octave", int)
        return (step, alter, octave)
    else:
        return None


def _handle_print(e, position, part, ongoing):
    if "new-page" in e.attrib:
        _handle_new_page(position, part, ongoing)
        _handle_new_system(position, part, ongoing)
    if "new-system" in e.attrib:
        _handle_new_system(position, part, ongoing)


def _add_tempo_if_unique(position, part, tempo):
    """
    Add score.Tempo object `tempo` at `position` on `part` if and only if
    there are no starting score.Tempo objects at that position. score.Tempo
    objects are generated by <sound tempo=...> as well as textual directions
    (e.g. "q=100"). This function avoids multiple synchronous tempo indications
    (whether redundant or conflicting)
    """
    point = part.get_point(position)
    if point is not None:
        tempos = point.starting_objects.get(score.Tempo, [])
        if tempos == []:
            part.add(tempo, position)
        else:
            warnings.warn("not adding duplicate or conflicting tempo indication")


def _handle_sound(e, position, part):
    if "tempo" in e.attrib:
        tempo = score.Tempo(int(e.attrib["tempo"]), "q")
        # part.add_starting_object(position, tempo)
        (position, part, tempo)


def _handle_note(e, position, part, ongoing, prev_note, doc_order, prev_beam=None):
    # get some common features of element if available
    duration = get_value_from_tag(e, "duration", int) or 0
    # elements may have an explicit temporal offset
    # offset = get_value_from_tag(e, 'offset', int) or 0
    staff = get_value_from_tag(e, "staff", int) or 1
    voice = get_value_from_tag(e, "voice", int) or 1
    # initialize beam to None
    beam = None

    # get the stem direction of the note if any
    stem_dir = get_value_from_tag(e, "stem", str) or None

    # add support of uppercase "ID" tags
    note_id = (
        get_value_from_attribute(e, "id", str)
        if get_value_from_attribute(e, "id", str)
        else get_value_from_attribute(e, "ID", str)
    )

    symbolic_duration = {}
    dur_type = get_value_from_tag(e, "type", str)
    if dur_type:
        symbolic_duration["type"] = dur_type

    dots = len(e.findall("dot"))
    if dots:
        symbolic_duration["dots"] = dots

    actual_notes = get_value_from_tag(e, "time-modification/actual-notes", int)
    if actual_notes:
        symbolic_duration["actual_notes"] = actual_notes

    normal_notes = get_value_from_tag(e, "time-modification/normal-notes", int)
    if normal_notes:
        symbolic_duration["normal_notes"] = normal_notes

    chord = e.find("chord")
    if chord is not None:
        # this note starts at the same position as the previous note, and has
        # same duration
        assert prev_note is not None
        position = prev_note.start.t
        duration = prev_note.duration
        duration_from_symbolic = prev_note.duration_from_symbolic

    articulations_e = e.find("notations/articulations")
    if articulations_e is not None:
        articulations = get_articulations(articulations_e)
    else:
        articulations = {}

    ornaments_e = e.find("notations/ornaments")
    if ornaments_e is not None:
        ornaments = get_ornaments(ornaments_e)
    else:
        ornaments = {}

    technical_e = e.find("notations/technical")
    if technical_e is not None:
        technical_notations = get_technical_notations(technical_e)
    else:
        technical_notations = {}

    pitch = e.find("pitch")
    unpitch = e.find("unpitched")
    if pitch is not None:
        step = get_value_from_tag(pitch, "step", str)
        alter = get_value_from_tag(pitch, "alter", int)
        octave = get_value_from_tag(pitch, "octave", int)
        # When step is none check for accidental attribute
        if alter is None:
            alter = get_value_from_tag(e, "accidental", str)
            if alter is not None:
                alter = ACCIDENTAL_MAP[alter]

        grace = e.find("grace")

        if grace is not None:
            grace_type, steal_proportion = get_grace_info(grace)
            note = score.GraceNote(
                grace_type=grace_type,
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
                symbolic_duration=symbolic_duration,
                articulations=articulations,
                ornaments=ornaments,
                technical=technical_notations,
                steal_proportion=steal_proportion,
                doc_order=doc_order,
                stem_direction=stem_dir,
            )
            if isinstance(prev_note, score.GraceNote) and prev_note.voice == voice:
                note.grace_prev = prev_note
        else:
            beam = e.find("beam")
            if beam is not None:
                if "number" in beam.attrib.keys():
                    beam_num = beam.attrib["number"]
                    beam = beam.text if beam_num == "1" else None
                else:
                    beam = beam.text

            if beam == "begin":
                prev_beam = score.Beam()
                part.add(prev_beam, position)
                beam = prev_beam
            elif beam == "continue":
                beam = prev_beam
            elif beam == "end":
                beam = prev_beam
                prev_beam = None
            else:
                beam = None
                prev_beam = None

            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
                symbolic_duration=symbolic_duration,
                articulations=articulations,
                ornaments=ornaments,
                technical=technical_notations,
                doc_order=doc_order,
                stem_direction=stem_dir,
            )

        if isinstance(prev_note, score.GraceNote) and prev_note.voice == voice:
            prev_note.grace_next = note

    elif unpitch is not None:
        # note element is unpitched
        step = get_value_from_tag(unpitch, "display-step", str)
        octave = get_value_from_tag(unpitch, "display-octave", int)
        noteheadtag = e.find("notehead")
        noteheadstylebool = True
        notehead = None
        if noteheadtag is not None:
            notehead = get_value_from_tag(e, "notehead", str)
            noteheadstyle = get_value_from_attribute(noteheadtag, "filled", str)
            if noteheadstyle is not None:
                noteheadstylebool = {"no": False, "yes": True}[noteheadstyle]

        note = score.UnpitchedNote(
            step=step,
            octave=octave,
            id=note_id,
            voice=voice,
            staff=staff,
            notehead=notehead,
            noteheadstyle=noteheadstylebool,
            articulations=articulations,
            technical=technical_notations,
            symbolic_duration=symbolic_duration,
            doc_order=doc_order,
            stem_direction=stem_dir,
        )

    else:
        # note element is a rest
        note = score.Rest(
            id=note_id,
            voice=voice,
            staff=staff,
            symbolic_duration=symbolic_duration,
            articulations=articulations,
            technical=technical_notations,
            doc_order=doc_order,
        )
    part.add(note, position, position + duration)

    # After note is assigned to part we can assign the beam to the note if it exists
    if isinstance(beam, score.Beam):
        note.assign_beam(beam)

    ties = e.findall("tie")
    if len(ties) > 0:
        tie_key = ("tie", getattr(note, "midi_pitch", "rest"))
        tie_types = set(tie.attrib["type"] for tie in ties)

        if "stop" in tie_types:
            tie_prev = ongoing.get(tie_key, None)

            if tie_prev:
                note.tie_prev = tie_prev
                tie_prev.tie_next = note
                del ongoing[tie_key]

        if "start" in tie_types:
            ongoing[tie_key] = note

    notations = e.find("notations")

    if notations is not None:
        if notations.find("fermata") is not None:
            fermata = score.Fermata(note)
            part.add(fermata, position)
            note.fermata = fermata

        starting_slurs, stopping_slurs = handle_slurs(
            notations, ongoing, note, position
        )

        for slur in starting_slurs:
            part.add(slur, position)

        for slur in stopping_slurs:
            part.add(slur, end=position + duration)

        starting_tups, stopping_tups = handle_tuplets(notations, ongoing, note)

        for tup in starting_tups:
            part.add(tup, position)

        for tup in stopping_tups:
            part.add(tup, end=position + duration)

    new_position = position + duration

    return new_position, note, prev_beam


def handle_tuplets(notations, ongoing, note):
    starting_tuplets = []
    stopping_tuplets = []
    tuplets = notations.findall("tuplet")

    # this sorts all found tuplets by type (either 'start' or 'stop')
    # in reverse order, so all with type 'stop' will be before
    # the ones with 'start'?!.
    tuplets.sort(key=lambda x: x.attrib["type"], reverse=True)

    # Now that the tuplets are sorted by their type, sort them
    # by their numbers; First note that tuplets do not always
    # have a number attribute, then 1 is implied.
    tuplets.sort(
        key=lambda x: get_value_from_attribute(x, "number", int) or note.voice or 1
    )

    for tuplet_e in tuplets:
        tuplet_number = get_value_from_attribute(tuplet_e, "number", int) or note.voice
        tuplet_type = get_value_from_attribute(tuplet_e, "type", str)
        start_tuplet_key = ("start_tuplet", tuplet_number)
        stop_tuplet_key = ("stop_tuplet", tuplet_number)

        if tuplet_type == "start":
            # Get information about the tuplet if present in the XML
            tuplet_actual = tuplet_e.find("tuplet-actual")
            tuplet_normal = tuplet_e.find("tuplet-normal")
            if tuplet_actual is not None and tuplet_normal is not None:
                tuplet_actual_notes = get_value_from_tag(
                    tuplet_actual, "tuplet-number", int
                )
                tuplet_actual_type = get_value_from_tag(
                    tuplet_actual, "tuplet-type", str
                )
                tuplet_normal_notes = get_value_from_tag(
                    tuplet_normal, "tuplet-number", int
                )
                tuplet_normal_type = get_value_from_tag(
                    tuplet_normal, "tuplet-type", str
                )
            # If no information, try to infer it from the note
            else:
                tuplet_actual_notes = note.symbolic_duration.get("actual_notes", None)
                tuplet_normal_notes = note.symbolic_duration.get("normal_notes", None)
                tuplet_actual_type = note.symbolic_duration.get("type", None)
                tuplet_normal_type = tuplet_actual_type

            # If anyone of the attributes is not set, we set them all to None as we can't really
            # do anything useful with only partial information about the tuplet
            if None in (
                tuplet_actual_notes,
                tuplet_normal_notes,
                tuplet_actual_type,
                tuplet_normal_type,
            ):
                tuplet_actual_notes = None
                tuplet_normal_notes = None
                tuplet_actual_type = None
                tuplet_normal_type = None

            # check if we have a stopped_tuplet in ongoing that corresponds to
            # this start
            tuplet = ongoing.pop(stop_tuplet_key, None)

            if tuplet is None:
                tuplet = score.Tuplet(
                    note,
                    actual_notes=tuplet_actual_notes,
                    normal_notes=tuplet_normal_notes,
                    actual_type=tuplet_actual_type,
                    normal_type=tuplet_normal_type,
                )
                ongoing[start_tuplet_key] = tuplet

            else:
                tuplet.start_note = note
                tuplet.actual_notes = tuplet_actual_notes
                tuplet.normal_notes = tuplet_normal_notes
                tuplet.actual_type = tuplet_actual_type
                tuplet.normal_type = tuplet_normal_type

            starting_tuplets.append(tuplet)

        elif tuplet_type == "stop":
            tuplet = ongoing.pop(start_tuplet_key, None)
            if tuplet is None:
                # tuplet stop occurs before tuplet start in document order, that
                # is a valid scenario
                tuplet = score.Tuplet(None, note)
                ongoing[stop_tuplet_key] = tuplet
            else:
                tuplet.end_note = note

            stopping_tuplets.append(tuplet)

    # assert that starting tuplet times are before stopping tuplet times
    for start_tuplet, stop_tuplet in zip(starting_tuplets, stopping_tuplets):
        assert (
            start_tuplet.start_note.start.t <= stop_tuplet.end_note.start.t
        ), "Tuplet start time is after tuplet stop time"

    # check that tuplets start and end notes belong to the same voice
    for tuplet in starting_tuplets + stopping_tuplets:
        if (
            tuplet.start_note is not None
            and tuplet.end_note is not None
            and (tuplet.start_note.voice != tuplet.end_note.voice)
        ):
            warnings.warn(
                f"Tuplet start and end notes do not belong to the same voice "
                f"({tuplet.start_note.voice} != {tuplet.end_note.voice}). This might "
                f'indicate a missing <tuplet type="start"> or <tuplet type="stop">.'
            )
    return starting_tuplets, stopping_tuplets


def handle_slurs(notations, ongoing, note, position):
    # we need position here to check for erroneous slurs: sometimes a slur stop
    # is encountered before the corresponding slur start. This is a valid use
    # case (e.g. slur starts in staff 2 and ends in staff 1). However, if the
    # stop is before the start in time, then it is just a MusicXML encoding
    # error.

    starting_slurs = []
    stopping_slurs = []
    slurs = notations.findall("slur")

    # this sorts all found slurs by type (either 'start' or 'stop')
    # in reverse order, so all with type 'stop' will be before
    # the ones with 'start'?!.
    slurs.sort(key=lambda x: x.attrib["type"], reverse=True)

    # Now that the slurs are sorted by their type, sort them
    # by their numbers; First note that slurs do not always
    # have a number attribute, then 1 is implied.
    slurs.sort(
        key=lambda x: get_value_from_attribute(x, "number", int) or note.voice or 1
    )

    for slur_e in slurs:
        slur_number = get_value_from_attribute(slur_e, "number", int) or note.voice
        slur_type = get_value_from_attribute(slur_e, "type", str)
        start_slur_key = ("start_slur", slur_number)
        stop_slur_key = ("stop_slur", slur_number)

        if slur_type == "start":
            # check if we have a stopped_slur in ongoing that corresponds to
            # this stop
            slur = ongoing.pop(stop_slur_key, None)

            # if slur.end_note.start.t < position then the slur stop is
            # rogue. We drop it and treat the slur start like a fresh start
            if slur is None or slur.end_note.start.t < position:
                if slur and slur.end_note.start.t < position:
                    msg = (
                        "Dropping slur {} starting at {} ({}) and ending "
                        "at {} ({})".format(
                            slur_number,
                            position,
                            note.id,
                            slur.end_note.start.t,
                            slur.end_note.id,
                        )
                    )
                    warnings.warn(msg)
                    # remove the slur from the timeline
                    slur.end_note.start.remove_ending_object(slur)
                    # remove the reference to the slur in the end note
                    slur.end_note.slur_stops.remove(slur)

                slur = score.Slur(note)
                ongoing[start_slur_key] = slur

            else:
                slur.start_note = note

            starting_slurs.append(slur)

        elif slur_type == "stop":
            slur = ongoing.pop(start_slur_key, None)

            if slur is None or slur.start_note.start.t > position:
                if slur and slur.start_note.start.t > position:
                    msg = (
                        "Dropping slur {} starting at {} ({}) and ending "
                        "at {} ({})".format(
                            slur_number,
                            slur.start_note.start.t,
                            slur.start_note.id,
                            position,
                            note.id,
                        )
                    )
                    warnings.warn(msg)
                    # remove the slur from the timeline
                    slur.start_note.start.remove_starting_object(slur)
                    # remove the reference to the slur in the end note
                    slur.start_note.slur_starts.remove(slur)

                # slur stop occurs before slur start in document order, that
                # is a valid scenario
                slur = score.Slur(None, note)
                ongoing[stop_slur_key] = slur

            else:
                slur.end_note = note

            stopping_slurs.append(slur)

    return starting_slurs, stopping_slurs


def get_grace_info(grace):
    # grace note handling
    grace_type = "grace"
    steal_proportion = None

    slash_text = get_value_from_attribute(grace, "slash", str)
    if slash_text == "yes":
        grace_type = "acciaccatura"

    steal_prc = get_value_from_attribute(grace, "steal-time-following", float)
    if steal_prc is not None:
        steal_proportion = steal_prc / 100
        grace_type = "appoggiatura"

    steal_prc = get_value_from_attribute(grace, "steal-time-previous", float)
    if steal_prc is not None:
        steal_proportion = steal_prc / 100
        grace_type = "acciaccatura"

    return grace_type, steal_proportion


def get_articulations(e):
    # <!ELEMENT articulations
    # 	((accent | strong-accent | staccato | tenuto |
    # 	  detached-legato | staccatissimo | spiccato |
    # 	  scoop | plop | doit | falloff | breath-mark |
    # 	  caesura | stress | unstress | soft-accent |
    # 	  other-articulation)*)>
    articulations = (
        "accent",
        "strong-accent",
        "staccato",
        "tenuto",
        "detached-legato",
        "staccatissimo",
        "spiccato",
        "scoop",
        "plop",
        "doit",
        "falloff",
        "breath-mark",
        "caesura",
        "stress",
        "unstress",
        "soft-accent",
    )
    return [a for a in articulations if e.find(a) is not None]


def get_ornaments(e):
    #  ornaments elements:
    #  trill-mark
    #  turn
    #  delayed-turn
    #  inverted-turn
    #  delayed-inverted-turn
    #  vertical-turn
    #  inverted-vertical-turn
    #  shake
    #  wavy-line
    #  mordent
    #  inverted-mordent
    #  schleifer
    #  tremolo
    #  haydn
    #  other-ornament

    ornaments = (
        "trill-mark",
        "turn",
        "delayed-turn",
        "inverted-turn",
        "delayed-inverted-turn",
        "vertical-turn",
        "inverted-vertical-turn",
        "shake",
        "wavy-line",
        "mordent",
        "inverted-mordent",
        "schleifer",
        "tremolo",
        "haydn",
        "other-ornament",
    )
    return [a for a in ornaments if e.find(a) is not None]


def get_technical_notations(e: etree._Element) -> List[score.NoteTechnicalNotation]:
    # For a full list of technical notations
    # https://usermanuals.musicxml.com/MusicXML/Content/EL-MusicXML-technical.htm
    # for now we only support fingering
    technical_notation_parsers = {
        "fingering": parse_fingering,
    }

    technical_notations = [
        parser(e.find(a))
        for a, parser in technical_notation_parsers.items()
        if e.find(a) is not None
    ]

    return technical_notations


def parse_fingering(e: etree._Element) -> score.Fingering:

    try:
        # There seems to be a few cases with fingerings encoded like 4_1.
        # This is not standard in MusicXML according to the documentation,
        # but since it appears in files from the web, and can be displayed
        # with MuseScore, the solution for now is just to take the fist value.
        finger_info = parse_ints(e.text)
    except Exception as e:
        # Do not raise an error if fingering info cannot be parsed, insted
        # just set it as None.
        warnings.warn(f"Cannot parse fingering info for {e.text}!")
        finger_info = [None]

    is_alternate = e.attrib.get("alternate", False)
    is_substitution = e.attrib.get("substitution", False)
    placement = e.attrib.get("placement", None)

    # If there is more than one finger, only take the first one
    fingering = score.Fingering(
        fingering=finger_info[0],
        is_substitution=is_alternate,
        is_alternate=is_alternate,
        placement=placement,
    )
    return fingering


@deprecated_alias(fn="filename")
def musicxml_to_notearray(
    filename,
    flatten_parts=True,
    include_pitch_spelling=False,
    include_key_signature=False,
    include_time_signature=False,
):
    """Return pitch, onset, and duration information for notes from a
    MusicXML file as a structured array.

    By default a single array is returned by combining the note
    information of all parts in the MusicXML file.

    Parameters
    ----------
    fn : str
        Path to a MusicXML file
    flatten_parts : bool
        If `True`, returns a single array containing all notes.
        Otherwise, returns a list of arrays for each part.
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

    Returns
    -------
    note_arrays : structured array or list of structured arrays
        Structured array or list of structured arrays containing
        score information.
    """

    scr = load_musicxml(
        filename=filename,
        force_note_ids="keep",
    )

    note_arrays = []
    for part in scr.parts:
        # Unfold any repetitions in part
        unfolded_part = score.unfold_part_maximal(part)
        # Compute note array
        note_array = ensure_notearray(
            notearray_or_part=unfolded_part,
            include_pitch_spelling=include_pitch_spelling,
            include_key_signature=include_key_signature,
            include_time_signature=include_time_signature,
        )
        note_arrays.append(note_array)

    if len(note_arrays) == 1:
        return note_arrays[0]
    elif len(note_arrays) > 1 and flatten_parts:
        return np.hstack(note_arrays)
    else:
        return note_arrays
