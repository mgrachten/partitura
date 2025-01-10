#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for importing MEI files.
"""
import os
from collections import OrderedDict
from lxml import etree
from fractions import Fraction
from xmlschema.names import XML_NAMESPACE
import partitura.score as score
from partitura.utils.music import (
    MEI_DURS_TO_SYMBOLIC,
    SYMBOLIC_TO_INT_DURS,
    SIGN_TO_ALTER,
    estimate_symbolic_duration,
)
from partitura.utils import PathLike, get_document_name
from partitura.utils.misc import deprecated_alias
import partitura as pt

try:
    import verovio

    VEROVIO_AVAILABLE = True
except:
    VEROVIO_AVAILABLE = False

import re
import warnings

import numpy as np


@deprecated_alias(mei_path="filename")
def load_mei(filename: PathLike) -> score.Score:
    """
    Loads a Mei score from path and returns a partitura Score object.

    Parameters
    ----------
    filename : PathLike
        The path to an MEI score.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A `Score` object
    """
    parser = MeiParser(filename)
    doc_name = get_document_name(filename)
    # create parts from the specifications in the mei
    parser.create_parts()
    # fill parts with the content from the mei
    parser.fill_parts()

    # TODO: Parse score info (composer, lyricist, etc.)
    scr = score.Score(
        id=doc_name,
        partlist=parser.parts,
    )

    return scr


class MeiParser(object):
    def __init__(self, mei_path: PathLike) -> None:
        # check if the file exists. Verovio won't complain if it doesn't
        if not os.path.exists(mei_path):
            raise FileNotFoundError(f"File {mei_path} not found.")
        document, ns = self._parse_mei(mei_path, use_verovio=VEROVIO_AVAILABLE)
        self.document = document
        self.ns = ns  # the namespace in the MEI file
        self.parts = (
            None  # parts get initialized in create_parts() and filled in fill_parts()
        )
        # find the music tag inside the document
        music_el = self.document.findall(self._ns_name("music", all=True))
        if len(music_el) != 1:
            raise Exception("Only MEI with a single <music> element are supported")
        self.music_el = music_el[0]
        self.repetitions = (
            []
        )  # to be filled when we encounter repetitions and process in the end
        self.barlines = (
            []
        )  # to be filled when we encounter barlines and process in the end
        self.endings = []

    def create_parts(self):
        # handle main scoreDef info: create the part list
        main_partgroup_el = self.music_el.find(self._ns_name("staffGrp", all=True))
        self.parts = self._handle_main_staff_group(main_partgroup_el)

    def fill_parts(self):
        # fill parts with the content of the score
        scores_el = self.music_el.findall(self._ns_name("score", all=True))
        if len(scores_el) != 1:
            raise Exception("Only MEI with a single score element are supported")
        sections_el = scores_el[0].findall(self._ns_name("section"))
        position = 0
        measure_number = 1
        for section_el in sections_el:
            # insert in parts all elements except ties
            position, measure_number = self._handle_section(
                section_el, list(score.iter_parts(self.parts)), position, measure_number
            )

        # handles ties
        self._tie_notes(scores_el[0], self.parts)

        # handle repetitions
        self._insert_repetitions()

        # handle barlines
        self._insert_barlines()

    # -------------- Functions to initialize the xml tree -----------------

    def _ns_name(self, name, ns=None, all=False):
        """
        Combines document namespace tag with element to fetch object from MEI lxml trees.

        Parameters
        ----------
        name : str
            Name of MEI element.
        ns : str or None
            The namespace tag of the document. Default to None.
        all : bool
            if True, search the entire subtree, otherwise only the first level.
        """
        if ns is None:
            ns = self.ns

        if not all:
            return "{" + ns + "}" + name
        else:
            return ".//{" + ns + "}" + name

    def _parse_mei(self, mei_path, use_verovio=True):
        """
        Parses an MEI file from path to an lxml tree.

        Parameters
        ----------
        mei_path : str
            The path of the MEI document.
        Returns
        -------
        document : lxml tree
            An lxml tree of the MEI score.
        """
        parser = etree.XMLParser(
            resolve_entities=False,
            huge_tree=False,
            remove_comments=True,
            remove_blank_text=True,
            recover=True,
        )

        if use_verovio:
            tk = verovio.toolkit(True)
            tk.loadFile(mei_path)
            mei_score = tk.getMEI("basic")
            # document = etree.parse(mei_score, parser)
            root = etree.fromstring(mei_score.encode("utf-8"), parser)
            tree = etree.ElementTree(root)
        else:
            tree = etree.parse(mei_path, parser)
            root = tree.getroot()
        # find the namespace
        ns = root.nsmap[None]
        # --> nsmap fetches a dict of the namespace Map, generally for root the key `None` fetches the namespace of the document.
        return tree, ns

    # functions to parse staves info

    def _handle_metersig(self, staffdef_el, position, part):
        """
        Handles meter signature and adds to part.

        Parameters
        ----------
        staffdef_el : lxml etree
            A lxml substree of a staff's mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        metersig_el = staffdef_el.find(self._ns_name("meterSig"))
        if metersig_el is not None:  # new element inside
            numerator = int(metersig_el.attrib["count"])
            denominator = int(metersig_el.attrib["unit"])
        elif (
            staffdef_el.get("meter.count") is not None
        ):  # all encoded as attributes in staffdef
            numerator = int(staffdef_el.attrib["meter.count"])
            denominator = int(staffdef_el.attrib["meter.unit"])
        else:  # the informatio is encoded in a parent scoredef
            found_ancestor_with_metrical_info = False
            for anc in staffdef_el.iterancestors(tag=self._ns_name("scoreDef")):
                if anc.get("meter.count") is not None:
                    found_ancestor_with_metrical_info = True
                    break
            if found_ancestor_with_metrical_info:
                numerator = int(anc.attrib["meter.count"])
                denominator = int(anc.attrib["meter.unit"])
            else:
                raise Exception(
                    f"The time signature is not encoded in {staffdef_el.get(self._ns_name('id'))} or in any ancestor scoreDef"
                )
        new_time_signature = score.TimeSignature(numerator, denominator)
        part.add(new_time_signature, position)

    def _handle_keysig(self, staffdef_el, position, part):
        """
        Handles key signature and adds to part.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a staff's mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        keysig_el = staffdef_el.find(self._ns_name("keySig"))
        if keysig_el is not None:  # new element inside
            sig = keysig_el.attrib["sig"]
            # now extract partitura keysig parameters
            fifths = self._mei_sig_to_fifths(sig)
            mode = keysig_el.get("mode")
        elif (
            staffdef_el.get("key.sig") is not None
        ):  # all encoded as attributes in staffdef
            sig = staffdef_el.attrib["key.sig"]
            # now extract partitura keysig parameters
            fifths = self._mei_sig_to_fifths(sig)
            mode = staffdef_el.get("key.mode")
        else:  # the information is encoded in a parent scoredef
            found_ancestor_with_key_info = False
            for anc in staffdef_el.iterancestors(tag=self._ns_name("scoreDef")):
                if anc.get("key.sig") is not None:
                    found_ancestor_with_key_info = True
                    break
            if found_ancestor_with_key_info:
                sig = anc.attrib["key.sig"]
                # now extract partitura keysig parameters
                fifths = self._mei_sig_to_fifths(sig)
                mode = anc.get("key.mode")
            else:
                warnings.warn(
                    f"The key signature is not encoded in {staffdef_el.get(self._ns_name('id'))} or in any ancestor scoreDef."
                )
                warnings.warn("A default key signature of C maj is set.")
                fifths = 0
                mode = "major"

        new_key_signature = score.KeySignature(fifths, mode)
        part.add(new_key_signature, position)

    def _compute_clef_octave(self, dis, dis_place):
        if dis is not None:
            sign = -1 if dis_place == "below" else 1
            octave = sign * int(int(dis) / 8)
        else:
            octave = 0
        return octave

    def _mei_sig_to_fifths(self, sig):
        """Produces partitura KeySignature.fifths parameter from the MEI sig attribute."""
        if sig[0] == "0":
            fifths = 0
        else:
            sign = 1 if sig[-1] == "s" else -1
            fifths = sign * int(sig[:-1])
        return fifths

    def _handle_clef(self, element, position, part):
        """Inserts a clef. Element can be either a cleff element or staffdef element.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.

        Returns
        -------
        position : int
            The current position of the note on the timeline.
        """
        # handle the case where we have clef informations inside staffdef el
        if element.tag == self._ns_name("staffDef"):
            clef_el = element.find(self._ns_name("clef"))
            if clef_el is not None:  # if there is a clef element inside
                return self._handle_clef(clef_el, position, part)
            else:  # if all info are in the staffdef element
                number = element.get("n")
                sign = element.get("clef.shape")
                line = element.get("clef.line")
                if (
                    number is not None and sign is not None and line is not None
                ):  # if there is clef info
                    octave = self._compute_clef_octave(
                        element.get("dis"), element.get("dis.place")
                    )
                else:  # no clef info available, go for default
                    warnings.warn("No clef information found, setting G2 as default.")
                    sign = "G"
                    line = 2
                    number = 1
                    octave = 0
        elif element.tag == self._ns_name("clef"):
            if element.get("sameas") is not None:  # this is a copy of another clef
                # it seems this is used in different layers for the same staff
                # we don't handle it to avoid clef duplications
                return position
            else:
                # find the staff number
                parent = element.getparent()
                if parent.tag == self._ns_name("staffDef"):
                    number = parent.attrib.get("n", 1)
                else:  # go back another level to staff element
                    number = parent.getparent().attrib.get("n", 1)
                sign = element.attrib["shape"]
                line = element.attrib["line"]
                octave = self._compute_clef_octave(
                    element.get("dis"), element.get("dis.place")
                )
        else:
            raise Exception("_handle_clef only accepts staffDef or clef elements")
        new_clef = score.Clef(int(number), sign, int(line), octave)
        part.add(new_clef, position)
        return position

    def _handle_staffdef(self, staffdef_el, position, part):
        """
        Derives meter, key and clef from lxml substree and pass them to part.

        Parameters
        ----------
        staffdef_el : lxml tree
            A lxml substree of a mei score.
        position : int
            Is the current position of the note on the timeline.
        part : particular.Part
            The created Partitura Part object.
        """
        # fill with time signature info
        self._handle_metersig(staffdef_el, position, part)
        # fill with key signature info
        self._handle_keysig(staffdef_el, position, part)
        # fill with clef info
        self._handle_clef(staffdef_el, position, part)

    def _intsymdur_from_symbolic(self, symbolic_dur):
        """Produce a int symbolic dur (e.g. 8 is a eight note), a dot number, and a tuplet modifier,
        e.g., (2,3) means there are 3 notes in the space of 2 notes."""
        intsymdur = SYMBOLIC_TO_INT_DURS[symbolic_dur["type"]]
        # deals with tuplets
        if symbolic_dur.get("actual_notes") is not None:
            assert symbolic_dur.get("normal_notes") is not None
            # intsymdur = (
            #     intsymdur * symbolic_dur["actual_notes"] / symbolic_dur["normal_notes"]
            # )
            tuplet_modifier = (
                symbolic_dur["normal_notes"],
                symbolic_dur["actual_notes"],
            )
        else:
            tuplet_modifier = None
        # deals with dots
        dots = symbolic_dur.get("dots") if symbolic_dur.get("dots") is not None else 0
        return intsymdur, dots, tuplet_modifier

    def _find_ppq(self):
        """Finds the ppq for MEI filed that do not explicitely encode this information"""
        els_with_dur = self.music_el.xpath(".//*[@dur]")
        durs = []
        durs_ppq = []
        for el in els_with_dur:
            symbolic_duration = self._get_symbolic_duration(el)
            intsymdur, dots, tuplet_mod = self._intsymdur_from_symbolic(
                symbolic_duration
            )
            if tuplet_mod is not None:
                # consider time modifications keeping the numerator of the minimized fraction
                minimized_fraction = Fraction(intsymdur * tuplet_mod[1], tuplet_mod[0])
                intsymdur = minimized_fraction.numerator
            # double the value if we have dots, to be sure be able to encode that with integers in partitura
            durs.append(intsymdur * (2**dots))
            durs_ppq.append(
                None if el.get("dur.ppq") is None else int(el.get("dur.ppq"))
            )

        if any([dppq is not None for dppq in durs_ppq]):
            # there is at least one element with both dur and dur.ppq
            for dur, dppq in zip(durs, durs_ppq):
                if dppq is not None:
                    return dppq * dur / 4
        else:
            # compute the ppq from the durations
            # add 4 to be sure to not go under 1 ppq
            durs.append(4)
            durs = np.array(durs)
            # remove elements smaller than 1
            durs = durs[durs >= 1]

            least_common_multiple = np.lcm.reduce(durs.astype(int))

            return least_common_multiple / 4

    def _handle_initial_staffdef(self, staffdef_el):
        """
        Handles the definition of a single staff.

        Parameters
        ----------
        staffdef_el : Element tree
            A subtree of a particular Staff from a score.

        Returns
        -------
        part : partitura.Part
            Returns a partitura part filled with meter, time signature, key signature information.
        """
        # Fetch the namespace of the staff.
        id = staffdef_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        label_el = staffdef_el.find(self._ns_name("label"))
        name = label_el.text if label_el is not None else ""
        ppq_attrib = staffdef_el.get("ppq")
        if ppq_attrib is not None:
            ppq = int(ppq_attrib)
        else:
            ppq = self._find_ppq()
        # generate the part
        part = score.Part(id, name, quarter_duration=ppq)
        # fill it with other info, e.g. meter, time signature, key signature
        self._handle_staffdef(staffdef_el, 0, part)
        return part

    def _handle_staffgroup(self, staffgroup_el):
        """
        Handles a staffGrp. WARNING: in MEI piano staves are a staffGrp

        Parameters
        ----------
        staffgroup_el : element tree
            A subtree of Staff Group from a score.

        Returns
        -------
        staff_group : Partitura.PartGroup
            A partitura PartGroup object made by calling and appending as children ever staff separately.
        """
        group_symbol_el = staffgroup_el.find(self._ns_name("grpSym"))
        if group_symbol_el is None:
            group_symbol = staffgroup_el.attrib["symbol"]
        else:
            group_symbol = group_symbol_el.attrib["symbol"]
        label_el = staffgroup_el.find(self._ns_name("label"))
        name = label_el.text if label_el is not None else None
        id = staffgroup_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        staff_group = score.PartGroup(group_symbol, group_name=name, id=id)
        staves_el = staffgroup_el.findall(self._ns_name("staffDef"))
        for s_el in staves_el:
            new_part = self._handle_initial_staffdef(s_el)
            staff_group.children.append(new_part)
        staff_groups_el = staffgroup_el.findall(self._ns_name("staffGrp"))
        for sg_el in staff_groups_el:
            new_staffgroup = self._handle_staffgroup(sg_el)
            staff_group.children.append(new_staffgroup)
        return staff_group

    def _handle_main_staff_group(self, main_staffgrp_el):
        """
        Handles the main staffGrp that contains all other staves or staff groups.

        Parameters
        ----------
        main_staffgrp_el : element_tree

        Returns
        -------
        part_list : list
            Created list of parts filled with key and time signature information.
        """
        staves_el = main_staffgrp_el.findall(self._ns_name("staffDef"))
        staff_groups_el = main_staffgrp_el.findall(self._ns_name("staffGrp"))
        # the list of parts or part groups
        part_list = []
        # process the parts
        # TODO add Parallelization to handle part parsing in parallel
        for s_el in staves_el:
            new_part = self._handle_initial_staffdef(s_el)
            part_list.append(new_part)
        # process the part groups
        for sg_el in staff_groups_el:
            new_staffgroup = self._handle_staffgroup(sg_el)
            part_list.append(new_staffgroup)
        return part_list

    # functions to parse the content of parts

    def _note_el_to_accid_int(self, note_el) -> int:
        """Accidental strings to integer pitch.
        It consider the two values of accid and accid.ges (when the accidental is implicit in the bar)
        """
        if note_el.get("accid") is not None:
            return SIGN_TO_ALTER[note_el.get("accid")]
        elif note_el.get("accid.ges") is not None:
            return SIGN_TO_ALTER[note_el.get("accid.ges")]
        elif note_el.find(self._ns_name("accid")) is not None:
            if note_el.find(self._ns_name("accid")).get("accid") is not None:
                return SIGN_TO_ALTER[note_el.find(self._ns_name("accid")).get("accid")]
            else:
                return SIGN_TO_ALTER[
                    note_el.find(self._ns_name("accid")).get("accid.ges")
                ]
        else:
            return None

    def _pitch_info(self, note_el):
        """
        Given a note element fetches PitchClassName, octave and accidental.

        Parameters
        ----------
        note_el

        Returns
        -------
        step : str
            The note Pitch class name.
        octave : int
            The number of octave
        alter : int
            Accidental string transformed to number.
        """
        step = note_el.attrib["pname"]
        octave = int(note_el.attrib["oct"])
        # accidentals can be accid, accid.ges or accid children elements
        alter = self._note_el_to_accid_int(note_el)
        return step, octave, alter

    def _get_symbolic_duration(self, el):
        symbolic_duration = {}
        symbolic_duration["type"] = MEI_DURS_TO_SYMBOLIC[el.attrib["dur"]]
        if not el.get("dots") is None:
            symbolic_duration["dots"] = int(el.get("dots"))
        # find eventual time modifications
        tuplet_ancestors = list(el.iterancestors(tag=self._ns_name("tuplet")))
        if len(tuplet_ancestors) == 0:
            pass
        elif len(tuplet_ancestors) == 1:
            symbolic_duration["actual_notes"] = int(tuplet_ancestors[0].attrib["num"])
            symbolic_duration["normal_notes"] = int(
                tuplet_ancestors[0].attrib["numbase"]
            )
        else:
            raise Exception("Nested tuplets are not yet supported.")
        return symbolic_duration

    def _duration_info(self, el, part):
        """
        Extract duration info from a xml element.

        It works for example with note_el, chord_el

        Parameters
        ----------
        el : lxml tree
            the xml element to analyze
        part : partitura.Part
            The created partitura part object.

        Returns
        -------
        id :
        duration :
        symbolic_duration :
        """
        # symbolic duration
        symbolic_duration = self._get_symbolic_duration(el)

        # duration in ppq
        if el.get("dur.ppq") is not None or el.get("grace") is not None:
            # find duration in ppq. For grace notes is 0
            duration = 0 if el.get("grace") is not None else int(el.get("dur.ppq"))
        else:
            # compute the duration from the symbolic duration
            intsymdur, dots, tuplet_mod = self._intsymdur_from_symbolic(
                symbolic_duration
            )
            divs = part._quarter_durations[0]  # divs is the same as ppq
            if tuplet_mod is None:
                tuplet_mod = (
                    1,
                    1,
                )  # if no tuplet modifier, set one that does not change the duration
            duration = (divs * 4 * tuplet_mod[0]) / (intsymdur * tuplet_mod[1])
            for d in range(dots):
                duration = duration + 0.5 * duration
            # sanity check to verify the divs are correctly set
            assert duration == int(duration)

        # find id
        id = el.attrib[self._ns_name("id", XML_NAMESPACE)]
        return id, int(duration), symbolic_duration

    def _handle_note(self, note_el, position, voice, staff, part) -> int:
        """
        Handles note elements and imports the to part.

        Parameters
        ----------
        note_el : lxml substree
            The lxml substree of a note element.
        position : int
            The current position on the timeline.
        voice : int
            The currect voice index.
        staff : int
            The current staff index.
        part : partitura.Part
            The created partitura part object.

        Returns
        -------
        position + duration : into
            The updated position on the timeline.
        """
        # find pitch info
        step, octave, alter = self._pitch_info(note_el)
        # find duration info
        note_id, duration, symbolic_duration = self._duration_info(note_el, part)
        # find if it's grace
        grace_attr = note_el.get("grace")
        # find if it has a different staff specification (for staff crossings)
        different_staff = note_el.get("staff")
        if different_staff is not None:
            staff = int(different_staff)
        if grace_attr is None:
            # create normal note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
        else:
            # create grace note
            if grace_attr == "unacc":
                grace_type = "acciaccatura"
            elif grace_attr == "acc":
                grace_type = "appoggiatura"
            else:  # unknow type
                grace_type = "grace"
            note = score.GraceNote(
                grace_type=grace_type,
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
        # add note to the part
        part.add(note, position, position + duration)
        # return duration to update the position in the layer
        return position + duration

    def _handle_rest(self, rest_el, position, voice, staff, part):
        """
        Handles the rest element updates part and position.

        Parameters
        ----------
        rest_el : lxml tree
            A rest element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        Also adds the rest to the partitura part object.
        """
        # find duration info
        rest_id, duration, symbolic_duration = self._duration_info(rest_el, part)
        # find if it has a different staff specification (for staff crossings)
        different_staff = rest_el.get("staff")
        if different_staff is not None:
            staff = int(different_staff)
        # create rest
        rest = score.Rest(
            id=rest_id,
            voice=voice,
            staff=staff,
            symbolic_duration=symbolic_duration,
            articulations=None,
        )
        # add rest to the part
        part.add(rest, position, position + duration)
        # return duration to update the position in the layer
        return position + duration

    def _handle_mrest(self, mrest_el, position, voice, staff, part):
        """
        Handles a rest that spawn the entire measure

        Parameters
        ----------
        mrest_el : lxml tree
            A mrest element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        """
        # find id
        mrest_id = mrest_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        # find closest time signature
        last_ts = list(part.iter_all(cls=score.TimeSignature))[-1]
        # find divs per measure
        ppq = part.quarter_duration_map(position)
        parts_per_measure = int(ppq * 4 * last_ts.beats / last_ts.beat_type)
        # find if it has a different staff specification (for staff crossings)
        different_staff = mrest_el.get("staff")
        if different_staff is not None:
            staff = int(different_staff)

        # create dummy rest to insert in the timeline
        rest = score.Rest(
            id=mrest_id,
            voice=voice,
            staff=staff,
            symbolic_duration=estimate_symbolic_duration(parts_per_measure, ppq),
            articulations=None,
        )
        # add mrest to the part
        part.add(rest, position, position + parts_per_measure)
        # return duration to update the position in the layer
        return position + parts_per_measure

    def _handle_multirest(self, multirest_el, position, voice, staff, part):
        """
        Handles a rest that spawn multiple measures

        Parameters
        ----------
        multirest_el : lxml tree
            A mrest element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        """
        # find id
        multirest_id = multirest_el.attrib[self._ns_name("id", XML_NAMESPACE)]
        # find how many measures
        n_measures = int(multirest_el.attrib["num"])
        if n_measures > 1:
            raise Exception(
                f"Multi-rests with more than 1 measure are not supported yet. Found one with {n_measures}."
            )
        # find closest time signature
        last_ts = list(part.iter_all(cls=score.TimeSignature))[-1]
        # find divs per measure
        ppq = part.quarter_duration_map(position)
        parts_per_measure = int(ppq * 4 * last_ts.beats / last_ts.beat_type)
        # find if it has a different staff specification (for staff crossings)
        different_staff = multirest_el.get("staff")
        if different_staff is not None:
            staff = int(different_staff)

        # create dummy rest to insert in the timeline
        rest = score.Rest(
            id=multirest_id,
            voice=voice,
            staff=staff,
            symbolic_duration=estimate_symbolic_duration(parts_per_measure, ppq),
            articulations=None,
        )
        # add mrest to the part
        part.add(rest, position, position + parts_per_measure)
        # now iterate
        # return duration to update the position in the layer
        return position + parts_per_measure

    def _handle_chord(self, chord_el, position, voice, staff, part):
        """
        Handles a rest that spawn the entire measure

        Parameters
        ----------
        chord_el : lxml tree
            A chord element in the lxml tree.
        position : int
            The current position on the timeline.
        voice : int
            The voice of the section.
        staff : int
            The current staff also refers to a Part.
        part : Partitura.Part
            The created part to add elements to.

        Returns
        -------
        position + duration : int
            Next position on the timeline.
        """
        # find duration info
        chord_id, duration, symbolic_duration = self._duration_info(chord_el, part)
        # find if the entire chord has a different staff specification (for staff crossings)
        different_staff = chord_el.get("staff")
        if different_staff is not None:
            staff = int(different_staff)
        # find notes info
        notes_el = chord_el.findall(self._ns_name("note"))
        for note_el in notes_el:
            note_id = note_el.attrib[self._ns_name("id", XML_NAMESPACE)]
            # find pitch info
            step, octave, alter = self._pitch_info(note_el)
            # find if single notes have a different staff specification
            different_staff = note_el.get("staff")
            if different_staff is not None:
                note_staff = int(different_staff)
            else:
                note_staff = staff
            # create note
            note = score.Note(
                step=step,
                octave=octave,
                alter=alter,
                id=note_id,
                voice=voice,
                staff=note_staff,
                symbolic_duration=symbolic_duration,
                articulations=None,  # TODO : add articulation
            )
            # add note to the part
            part.add(note, position, position + duration)
            # return duration to update the position in the layer
        return position + duration

    def _handle_space(self, e, position, part):
        """Moves current position."""
        try:
            space_id, duration, symbolic_duration = self._duration_info(e, part)
        except (
            KeyError
        ):  # if the space don't have a duration, move to the end of the measure
            # find closest time signature
            last_ts = list(part.iter_all(cls=score.TimeSignature))[-1]
            # find divs per measure
            ppq = part.quarter_duration_map(position)
            parts_per_measure = int(ppq * 4 * last_ts.beats / last_ts.beat_type)
            # find divs elapsed since last barline
            last_barline = list(part.iter_all(cls=pt.score.Measure))[-1]
            duration = position - last_barline.start.t

        return position + duration

    def _handle_barline_symbols(self, measure_el, position: int, left_or_right: str):
        barline = measure_el.get(left_or_right)
        if barline is not None:
            if barline == "rptstart":
                self.repetitions.append({"type": "start", "pos": position})
                self.barlines.append({"type": "heavy-light", "pos": position})
            elif barline == "rptend":
                self.repetitions.append({"type": "stop", "pos": position})
                self.barlines.append({"type": "light-heavy", "pos": position})
            elif barline == "dbl":
                self.barlines.append({"type": "light-light", "pos": position})
            elif barline == "end":
                self.barlines.append({"type": "light-heavy", "pos": position})
            elif barline == "dashed":
                self.barlines.append({"type": "dashed", "pos": position})
            else:
                print(
                    f"{barline} in measure {measure_el.attrib[self._ns_name('id', XML_NAMESPACE)]} is a non supported barline type."
                )

    def _handle_layer_in_staff_in_measure(
        self, layer_el, ind_layer: int, ind_staff: int, position: int, part
    ) -> int:
        for i, e in enumerate(layer_el):
            if e.tag == self._ns_name("note"):
                new_position = self._handle_note(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("chord"):
                new_position = self._handle_chord(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("rest"):
                new_position = self._handle_rest(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("mRest"):  # rest that spawn the entire measure
                new_position = self._handle_mrest(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name(
                "multiRest"
            ):  # rest that spawn more than one measure
                new_position = self._handle_multirest(
                    e, position, ind_layer, ind_staff, part
                )
            elif e.tag == self._ns_name("beam"):
                # TODO : add Beam element
                # recursive call to the elements inside beam
                new_position = self._handle_layer_in_staff_in_measure(
                    e, ind_layer, ind_staff, position, part
                )
            elif e.tag == self._ns_name("tuplet"):
                # TODO : add Tuplet element
                # recursive call to the elements inside Tuplet
                new_position = self._handle_layer_in_staff_in_measure(
                    e, ind_layer, ind_staff, position, part
                )
            elif e.tag == self._ns_name("clef"):
                new_position = self._handle_clef(e, position, part)
            elif e.tag == self._ns_name("space"):
                new_position = self._handle_space(e, position, part)
            else:
                raise Exception("Tag " + e.tag + " not supported")

            # update the current position
            position = new_position
        return position

    def _handle_staff_in_measure(
        self,
        staff_el,
        staff_ind,
        position: int,
        part: pt.score.Part,
        measure_number: int,
    ):
        """
        Handles staffs inside a measure element.

        Parameters
        ----------
        staff_el : lxml etree
            The lxml subtree for a staff element.
        staff_ind : int
            The Staff index.
        position : int
            The current position on the timeline.
        part : Partitura.Part
            The created partitura part object.
        measure_number : int
            The number of the measure. This number is independent of the measure name specified in the score.
            It starts from 1 and always increases by 1 at each measure

        Returns
        -------
        end_positions[0] : int
            The final position on the timeline.
        """
        # add measure
        measure = score.Measure(
            number=measure_number, name=staff_el.getparent().get("n")
        )
        part.add(measure, position)

        layers_el = staff_el.findall(self._ns_name("layer"))
        end_positions = []
        for i_layer, layer_el in enumerate(layers_el):
            end_positions.append(
                self._handle_layer_in_staff_in_measure(
                    layer_el,
                    int(layer_el.attrib.get("n", i_layer + 1)),
                    staff_ind,
                    position,
                    part,
                )
            )
        # check if layers have equal duration (bad encoding, but it often happens)
        if not all([e == end_positions[0] for e in end_positions]):
            warnings.warn(
                f"Warning: voices have different durations in staff {staff_el.attrib[self._ns_name('id',XML_NAMESPACE)]}"
            )

        if (
            len(end_positions) == 0
        ):  # if a measure contains no elements (e.g., a forgotten rest)
            end_positions.append(position)
        # add end time of measure
        part.add(measure, None, max(end_positions))
        return max(end_positions)

    def _find_dir_positions(self, dir_el, bar_position):
        """Compute the position for a <dir> element.
        Returns an array, one position for each part."""
        delta_position_beat = float(dir_el.get("tstamp"))
        return [
            p.inv_beat_map(p.beat_map(bar_position) + delta_position_beat - 1)
            for p in score.iter_parts(self.parts)
        ]

    def _add_in_all_parts(self, tobj, starts):
        for part, start in zip(score.iter_parts(self.parts), starts):
            part.add(tobj, start)

    def _handle_dir_element(self, dir_el, position):
        # find the kind of element
        kind = dir_el.get("type")
        if kind is None:
            return
        dir_pos = self._find_dir_positions(dir_el, position)
        if kind == "fine":
            self._add_in_all_parts(score.Fine(), dir_pos)
        elif kind == "dacapo":
            self._add_in_all_parts(score.DaCapo(), dir_pos)

    def _handle_directives(self, measure_el, position):
        dir_els = measure_el.findall(self._ns_name("dir"))
        for dir_el in dir_els:
            self._handle_dir_element(dir_el, position)

    def _handle_section(self, section_el, parts, position: int, measure_number: int):
        """
        Returns position and fills parts with elements.

        Parameters
        ----------
        section_el : lxml tree
            An lxml substree of a MEI score reffering to a section.
        parts : list()
            A list of partitura Parts.
        position : int
            The current position on the timeline.
        measure_number : int
            The current measure_number

        Returns
        -------
        position : int
            The end position of the section.
        measure_number : int
            The number of the last measure.
        """
        for i_el, element in enumerate(section_el):
            # handle measures
            if element.tag == self._ns_name("measure"):
                # handle left barline symbols
                self._handle_barline_symbols(element, position, "left")
                # handle staves
                staves_el = element.findall(self._ns_name("staff"))
                if len(list(staves_el)) != len(list(parts)):
                    raise Exception(f"Not all parts are specified in measure {i_el}")
                end_positions = []
                for i_s, (part, staff_el) in enumerate(zip(parts, staves_el)):
                    end_positions.append(
                        self._handle_staff_in_measure(
                            staff_el,
                            int(staff_el.attrib.get("n", i_s + 1)),
                            position,
                            part,
                            measure_number,
                        )
                    )
                # handle directives (dir elements)
                self._handle_directives(element, position)
                # sanity check that all layers have equal duration
                max_position = max(end_positions)
                if not all([e == max_position for e in end_positions]):
                    warnings.warn(
                        f"Warning : parts have measures of different duration in measure {element.attrib[self._ns_name('id',XML_NAMESPACE)]}"
                    )
                    # # enlarge measures to the max
                    # for part in parts:
                    #     last_measure = list(part.iter_all(pt.score.Measure))[-1]
                    #     if last_measure.end.t != max_position:
                    #         part.add(
                    #             pt.score.Measure(number=last_measure.number),
                    #             max_position
                    #         )
                    #         part.remove(last_measure)
                position = max_position
                # handle right barline symbol
                self._handle_barline_symbols(element, position, "right")
                measure_number += 1
            # handle staffDef elements
            elif element.tag == self._ns_name("scoreDef"):
                # meter modifications
                metersig_el = element.find(self._ns_name("meterSig"))
                if (metersig_el is not None) or (
                    element.get("meter.count") is not None
                ):
                    for part in parts:
                        self._handle_metersig(element, position, part)
                # key signature modifications
                keysig_el = element.find(self._ns_name("keySig"))
                if (keysig_el is not None) or (element.get("key.sig") is not None):
                    for part in parts:
                        self._handle_keysig(element, position, part)
            # handle nested section
            elif element.tag == self._ns_name("section"):
                position, measure_number = self._handle_section(
                    element, parts, position, measure_number
                )
            elif element.tag == self._ns_name("ending"):
                ending_start = position
                position, measure_number = self._handle_section(
                    element, parts, position, measure_number
                )
                # insert the ending element
                ending_number = int(re.sub("[^0-9]", "", element.attrib["n"]))
                self._add_ending(ending_start, position, ending_number, parts)
            # explicit repetition expansions
            elif element.tag == self._ns_name("expansion"):
                pass
            # system break
            elif element.tag == self._ns_name("sb"):
                pass
            # page break
            elif element.tag == self._ns_name("pb"):
                pass
            else:
                raise Exception(f"element {element.tag} is not yet supported")

        return position, measure_number

    def _add_ending(self, start_ending, end_ending, ending_string, parts):
        for part in score.iter_parts(parts):
            part.add(score.Ending(ending_string), start_ending, end_ending)

    def _tie_notes(self, section_el, part_list):
        """Ties all notes in a part.
        This function must be run after the parts are completely created."""
        # TODO : support ties written as attributes with @tie sintax
        ties_el = section_el.findall(self._ns_name("tie", all=True))
        # create a dict of id : note, to speed up search
        all_notes = [
            note
            for part in score.iter_parts(part_list)
            for note in part.iter_all(cls=score.Note, include_subclasses=True)
        ]
        all_notes_dict = {note.id: note for note in all_notes}
        for tie_el in ties_el:
            start_id = tie_el.get("startid")
            end_id = tie_el.get("endid")
            if start_id is None or end_id is None:
                warnings.warn(
                    f"Warning: tie {tie_el.attrib[self._ns_name('id',XML_NAMESPACE)]} is missing the a startid or endid"
                )
            else:
                # remove the # in first position
                start_id = start_id[1:]
                end_id = end_id[1:]
                # set tie prev and tie next in partitura note objects
                all_notes_dict[start_id].tie_next = all_notes_dict[end_id]
                all_notes_dict[end_id].tie_prev = all_notes_dict[start_id]

    def _insert_repetitions(self):
        if len(self.repetitions) == 0:
            return
        ## sanitize the found repetitions in case a starting rep is missing
        if self.repetitions[0]["type"] == "stop":
            # add a start symbol at 0
            print(
                "WARNING : unmatched repetitions. adding a repetition start at position 0"
            )
            self.repetitions.insert(0, {"type": "start", "pos": 0})

        status = "stop"
        sanitized_repetition_list = []
        # check if start-stop are alternate
        for i_rep, rep in enumerate(self.repetitions):
            if rep["type"] != status:
                sanitized_repetition_list.append(rep)
            else:
                if (
                    rep["type"] == "start"
                ):  # missing stop, inserting one right before start
                    print(
                        f"WARNING : unmatched repetitions. adding a repetition stop at position {rep['pos']}"
                    )
                    sanitized_repetition_list.append(
                        {"type": "stop", "pos": rep["pos"]}
                    )
                else:  # missing start, inserting one at the last stop
                    print(
                        f"WARNING : unmatched repetitions. adding a repetition start at position {sanitized_repetition_list[-1]['pos']}"
                    )
                    sanitized_repetition_list.append(
                        {"type": "start", "pos": sanitized_repetition_list[-1]["pos"]}
                    )
                # proceed by inserting rep
                sanitized_repetition_list.append(rep)
            # switch the status
            status = "stop" if status == "start" else "start"
        # check if ending with a start
        if sanitized_repetition_list[-1] == "start":
            print("WARNING : unmatched repetitions. Ignoring last start")
        ## sanitize the found repetitions to remove duplicates
        sanitized_repetition_list = list(
            OrderedDict(
                (tuple(d.items()), d) for d in sanitized_repetition_list
            ).values()
        )
        self.repetitions = sanitized_repetition_list

        ## insert the repetitions to all parts
        for rep_start, rep_stop in zip(self.repetitions[:-1:2], self.repetitions[1::2]):
            assert (
                rep_start["type"] == "start" and rep_stop["type"] == "stop"
            ), "Something wrong with repetitions"
            for part in score.iter_parts(self.parts):
                part.add(score.Repeat(), rep_start["pos"], rep_stop["pos"])

    def _insert_barlines(self):
        for bl in self.barlines:
            for part in score.iter_parts(self.parts):
                part.add(score.Barline(bl["type"]), bl["pos"])
