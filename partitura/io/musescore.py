#!/usr/bin/env python

"""This module contains functionality to use the MuseScore program as a
backend for loading and rendering scores.

"""

import platform
import logging
import os
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile

LOGGER = logging.getLogger(__name__)

from partitura.io.importmusicxml import load_musicxml
from partitura.io.exportmusicxml import save_musicxml


class MuseScoreNotFoundException(Exception):
    pass


class FileImportException(Exception):
    pass


def find_musescore3():
    # # possible way to detect MuseScore... executable
    # for p in os.environ['PATH'].split(':'):
    #     c = glob.glob(os.path.join(p, 'MuseScore*'))
    #     if c:
    #         print(c)
    #         break

    result = shutil.which("musescore")

    if result is None:
        result = shutil.which("mscore")

    if platform.system() == "Linux":
        pass

    elif platform.system() == "Darwin":

        result = shutil.which("/Applications/MuseScore 3.app/Contents/MacOS/mscore")

    elif platform.system() == "Windows":
        pass

    return result


def load_via_musescore(fn, ensure_list=False, validate=False, force_note_ids=True):
    """Load a score through through the MuseScore program.

    This function attempts to load the file in MuseScore, export it as
    MusicXML, and then load the MusicXML. This should enable loading
    of all file formats that for which MuseScore has import-support
    (e.g. MIDI, and ABC, but currently not MEI).

    Parameters
    ----------
    fn : str
        Filename of the score to load
    ensure_list : bool, optional
        When True return a list independent of how many part or
        partgroup elements were created from the MIDI file. By
        default, when the return value of `load_musicxml` produces a
    single : class:`partitura.score.Part` or
        :Class:`partitura.score.PartGroup` element, the element itself
        is returned instead of a list containing the element. Defaults
        to False.
    validate : bool, optional
        When True the validity of the MusicXML generated by MuseScore is checked
        against the MusicXML 3.1 specification before loading the file. An
        exception will be raised when the MusicXML is invalid.
        Defaults to False.
    force_note_ids : bool, optional.
        When True each Note in the returned Part(s) will have a newly
        assigned unique id attribute. Existing note id attributes in
        the MusicXML will be discarded.

    Returns
    -------
    :class:`partitura.score.Part`, :class:`partitura.score.PartGroup`, or a list of these
        One or more part or partgroup objects

    """

    mscore_exec = find_musescore3()

    if not mscore_exec:

        raise MuseScoreNotFoundException()

    with NamedTemporaryFile(suffix=".musicxml") as xml_fh:

        cmd = [mscore_exec, "-o", xml_fh.name, fn]

        try:

            ps = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            if ps.returncode != 0:

                raise FileImportException(
                    "Command {} failed with code {}. MuseScore error messages:\n {}".format(
                        cmd, ps.returncode, ps.stderr.decode("UTF-8")
                    )
                )
        except FileNotFoundError as f:

            raise FileImportException(
                'Executing "{}" returned  {}.'.format(" ".join(cmd), f)
            )

        return load_musicxml(
            xml_fh.name,
            ensure_list=ensure_list,
            validate=validate,
            force_note_ids=force_note_ids,
        )


def render_musescore(part, fmt, out_fn=None, dpi=90):
    """Render a part using musescore.

    Parameters
    ----------
    part : Part
        Part to be rendered
    fmt : {'png', 'pdf'}
        Output image format
    out_fn : str or None, optional
        The path of the image output file, if not specified, the
        rendering will be saved to a temporary filename. Defaults to
        None.
    dpi : int, optional
        Image resolution. This option is ignored when `fmt` is
        'pdf'. Defaults to 90.

    """
    mscore_exec = find_musescore3()

    if not mscore_exec:

        return None

    if fmt not in ("png", "pdf"):

        LOGGER.warning("warning: unsupported output format")
        return None

    with NamedTemporaryFile(suffix=".musicxml") as xml_fh, NamedTemporaryFile(
        suffix=".{}".format(fmt)
    ) as img_fh:

        save_musicxml(part, xml_fh.name)

        cmd = [
            mscore_exec,
            "-T",
            "10",
            "-r",
            "{}".format(dpi),
            "-o",
            img_fh.name,
            xml_fh.name,
        ]
        try:

            # ps = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ps = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if ps.returncode != 0:
                LOGGER.error(
                    "Command {} failed with code {}; stdout: {}; stderr: {}".format(
                        cmd,
                        ps.returncode,
                        ps.stdout.decode("UTF-8"),
                        ps.stderr.decode("UTF-8"),
                    )
                )
                return None

        except FileNotFoundError as f:

            LOGGER.error('Executing "{}" returned  {}. {}'.format(" ".join(cmd), f))
            return None

        LOGGER.error(
            "Command {} returned with code {}; stdout: {}; stderr: {}".format(
                cmd, ps.returncode, ps.stdout.decode("UTF-8"), ps.stderr.decode("UTF-8")
            )
        )

        name, ext = os.path.splitext(img_fh.name)
        if fmt == "png":
            img_fn = "{}-1{}".format(name, ext)
        else:
            img_fn = "{}{}".format(name, ext)

        # print(img_fn, os.path.exists(img_fn))
        if os.path.exists(img_fn):
            if out_fn:
                shutil.copy(img_fn, out_fn)
                return out_fn
            else:
                return img_fn
        else:
            return None
