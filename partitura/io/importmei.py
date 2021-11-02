from lxml import etree
from xmlschema.names import XML_NAMESPACE
import partitura.score as score
from partitura.utils.music import MEI_DURS, SIGN_TO_ALTER


# functions to initialize the xml tree


def _parse_mei(xml_path):
    parser = etree.XMLParser(
        resolve_entities=False,
        huge_tree=False,
        remove_comments=True,
        remove_blank_text=True,
    )
    document = etree.parse(xml_path, parser)
    # find the namespace
    ns = document.getroot().nsmap[None]

    return document, ns


def _ns_name(name, ns, all=False):
    if not all:
        return "{" + ns + "}" + name
    else:
        return ".//{" + ns + "}" + name


# functions to parse staves info


def _handle_staffdef(staffdef_el, ns):
    """Handles the definition of a single staff"""
    id = staffdef_el.attrib[_ns_name("id", XML_NAMESPACE)]
    label_el = staffdef_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else ""
    ppq = int(staffdef_el.attrib["ppq"])
    return score.Part(id, name, quarter_duration=ppq)


def _handle_staffgroup(staffgroup_el, ns):
    """Handles a staffGrp. WARNING: in MEI piano staves are a staffGrp"""
    group_symbol_el = staffgroup_el.find(_ns_name("grpSym", ns))
    if group_symbol_el is None:
        group_symbol = staffgroup_el.attrib["symbol"]
    else:
        group_symbol = group_symbol_el.attrib["symbol"]
    label_el = staffgroup_el.find(_ns_name("label", ns))
    name = label_el.text if label_el is not None else None
    id = staffgroup_el.attrib[_ns_name("id", XML_NAMESPACE)]
    staff_group = score.PartGroup(group_symbol, group_name=name, id=id)
    staves_el = staffgroup_el.findall(_ns_name("staffDef", ns))
    for s_el in staves_el:
        new_part = _handle_staffdef(s_el, ns)
        staff_group.children.append(new_part)
    return staff_group


def _handle_main_staff_group(main_staffgrp_el, ns):
    """Handles the main staffGrp that contains all other staves or staff groups"""
    staves_el = main_staffgrp_el.findall(_ns_name("staffDef", ns))
    staff_groups_el = main_staffgrp_el.findall(_ns_name("staffGrp", ns))
    # the list of parts or part groups
    part_list = []
    # process the parts
    for s_el in staves_el:
        new_part = _handle_staffdef(s_el, ns)
        part_list.append(new_part)
    # process the part groups
    for sg_el in staff_groups_el:
        new_staffgroup = _handle_staffgroup(sg_el, ns)
        part_list.append(new_staffgroup)
    return part_list


# functions to parse the content of parts


def _accidstring_to_int(accid_string: str) -> int:
    if accid_string is None:
        return None
    else:
        return SIGN_TO_ALTER[accid_string]


def _pitch_info(note_el):
    step = note_el.attrib["pname"]
    octave = int(note_el.attrib["oct"])
    alter = _accidstring_to_int(note_el.get("accid"))
    return step, octave, alter


def _duration_info(el, ns):
    """Extract duration info from a xml element.

    It works for example with note_el, chord_el

    Args:
        el (lxml tree): the xml element to analyze

    Returns:
        id, duration and symbolic duration of the element
    """
    # find duration in ppq
    duration = int(el.attrib["dur.ppq"])
    # find symbolic duration
    symbolic_duration = {}
    symbolic_duration["type"] = el.attrib["dur"]
    if not el.get("dots") is None:
        symbolic_duration["dots"] = int(el.get("dots"))
    # find eventual time modifications
    parent = el.getparent()
    if parent.tag == _ns_name("tuplet", ns):
        symbolic_duration["actual_notes"] = parent.attrib["num"]
        symbolic_duration["normal_notes"] = parent.attrib["numbase"]
    # find id
    id = el.attrib[_ns_name("id", XML_NAMESPACE)]
    return id, duration, symbolic_duration


def _handle_note(note_el, position, voice, staff, part, ns):
    # find pitch info
    step, octave, alter = _pitch_info(note_el)
    # find duration info
    note_id, duration, symbolic_duration = _duration_info(note_el, ns)
    # create note
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
    # add note to the part
    part.add(note, position, position + duration)
    # return duration to update the position in the layer
    return position + duration


def _handle_rest(rest_el, position, voice, staff, part, ns):
    # find duration info
    rest_id, duration, symbolic_duration = _duration_info(rest_el, ns)
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


def _handle_chord(chord_el, position, voice, staff, part, ns):
    # find duration info
    chord_id, duration, symbolic_duration = _duration_info(chord_el, ns)
    # find notes info
    notes_el = chord_el.findall(_ns_name("note", ns))
    for note_el in notes_el:
        note_id = note_el.attrib[_ns_name("id", XML_NAMESPACE)]
        # find pitch info
        step, octave, alter = _pitch_info(note_el)
        # create note
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
        # add note to the part
        part.add(note, position, position + duration)
        # return duration to update the position in the layer
    return position + duration


def _handle_layer_in_staff_in_measure(
    layer_el, ind_layer: int, ind_staff: int, position: int, part, ns
) -> int:
    for i, e in enumerate(layer_el):
        if e.tag == _ns_name("note", ns):
            new_position = _handle_note(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("chord", ns):
            duration = _handle_chord(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("rest", ns):
            new_position = _handle_rest(e, position, ind_layer, ind_staff, part, ns)
        elif e.tag == _ns_name("beam", ns):
            # TODO : add Beam element
            # recursive call to the elements inside beam
            new_position = _handle_layer_in_staff_in_measure(
                e, ind_layer, ind_staff, position, part, ns
            )
        elif e.tag == _ns_name("tuplet", ns):
            # TODO : add Tuplet element
            # recursive call to the elements inside Tuplet
            new_position = _handle_layer_in_staff_in_measure(
                e, ind_layer, ind_staff, position, part, ns
            )
        else:
            raise Exception("Tag " + e.tag + " not supported")

        # update the current position
        position = new_position
    return position


def _handle_staff_in_measure(staff_el, staff_ind, position: int, part, ns):
    # add measure
    measure = score.Measure(number=staff_el.getparent().get("n"))
    part.add(measure, position)

    layers_el = staff_el.findall(_ns_name("layer", ns))
    end_positions = []
    for i_layer, layer_el in enumerate(layers_el):
        end_positions.append(
            _handle_layer_in_staff_in_measure(
                layer_el, i_layer, staff_ind, position, part, ns
            )
        )
    # sanity check that all layers have equal duration
    if not all([e == end_positions[0] for e in end_positions]):
        raise Exception("Different voices have different durations")
    return end_positions[0]


def _handle_staves_content(parts, measures_el, ns):
    position = 0
    for i_m, measure in enumerate(measures_el):
        staves_el = measure.findall(_ns_name("staff", ns))
        if len(list(staves_el)) != len(list(parts)):
            raise Exception("Not all parts are specified in measure" + i_m)
        end_positions = []
        for i_s, (part, staff_el) in enumerate(zip(parts, staves_el)):
            end_positions.append(
                _handle_staff_in_measure(staff_el, i_s, position, part, ns)
            )
        # sanity check that all layers have equal duration
        if not all([e == end_positions[0] for e in end_positions]):
            raise Exception("Different parts have measures of different duration")
        position = end_positions[0]
    return position


def _tie_notes(section, part, ns):
    """Ties all notes in a part.
    This function must be run after the parts are completely created."""
    ties = section.findall(_ns_name("tie", ns, True))
    for tie in ties:
        pass


def load_mei(xml_path: str):
    # parse xml file
    document, ns = _parse_mei(xml_path)

    # handle staff and staff groups info
    main_partgroup_el = document.find(_ns_name("staffGrp", ns, True))
    part_list = _handle_main_staff_group(main_partgroup_el, ns)

    # fill the content of the score
    sections_el = document.findall(_ns_name("section", ns, True))
    if len(sections_el) != 1:
        raise Exception("Only MEI with a single section are supported")
    measures_el = sections_el[0].findall((_ns_name("measure", ns)))
    _handle_staves_content(list(score.iter_parts(part_list)), list(measures_el), ns)

    return part_list

