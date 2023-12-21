"""
Weather tools for aircraft performance calculations.
"""
import os
import re
import typing

# noinspection PyPackageRequirements
from metar import Metar

__author__ = "Yaseen Reza"


def parsemetarfile(filepath: typing.Union[str, os.PathLike]) -> list[str]:
    """
    Reads a text file containing a METAR on each line.

    Args:
        filepath: The path to the METAR file.

    Returns:
        A list of METAR strings.

    Examples:

        >>> parsemetarfile("mymetars.txt")
        ['METAR EGHI 181450Z 22010KT 180V260 9999 BKN014 13/10 Q1028',
        'METAR EGHI 181420Z 22010KT 180V270 9999 BKN015 12/09 Q1028']

    """
    with open(filepath, "r") as f:
        text = f.read()

    # Findtext that does not involve escape characters and looks like METARs
    strings = re.findall(r"METAR.+", text)

    # Filter out illegal characters (by keeping only legal characters)
    strings = [re.match(r"[A-z\d\s/\\+\-$]*", s).group() for s in strings]

    return strings


def decodemetar(metar: str) -> Metar.Metar:
    """
    Decodes any METAR presented as a string.

    Args:
        metar: The METAR string.

    Returns:
        A Metar object from the python-metar package.

    Notes:
        Wraps the python-metar package, working around a date parsing error
        (python-metar fails if the dat is past the current day and out of range
        for the previous month. This would make it unsuitable for processing
        historical METARs).

    """

    try:
        decoded_obs = Metar.Metar(metar)

    except Metar.ParserError as _:
        # No month has fewer than 28 days, so this should be safe
        metar = re.sub(
            r"(METAR [A-Z]{4}) (\d{2})(\d{4})",  # 3 capture groups
            r"\1 28\3",  # Re-use capture group 1 and 3, replace 2
            metar
        )
        decoded_obs = Metar.Metar(metar)

    return decoded_obs
