import numpy as np
import re
from pathlib import Path
from typing import Tuple


def _extract_block(lines, marker) -> np.ndarray:
    """
    Return a numpy array of all numbers that appear after a given marker line.
    The numbers are read until the next non‑numeric line (or EOF).
    """
    # Find the line that contains the marker
    try:
        start = next(i for i, l in enumerate(lines) if marker in l)
    except StopIteration:
        raise ValueError(f"Marker '{marker}' not found in file")

    # All following lines that can be parsed as float are part of the block
    data = []
    for l in lines[start + 1 :]:
        # Stop if we hit an empty line or a line that doesn't look like a number
        l_strip = l.strip()
        if not l_strip:
            break
        # Try converting to float – if it fails we treat it as the end of the block
        try:
            data.append(float(l_strip))
        except ValueError:
            break

    return np.array(data, dtype=float)


def parse_lidort_cfg(filename) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a LIDORT input file and return three arrays:

        szds : Solar zenith angles (degrees)
        vzds : Viewing zenith angles (degrees)
        azds : Relative azimuth angles (degrees)


    Parameters
    ----------
    filename : str or Path
        Path to the LIDORT text file.

    Returns
    -------
    tuple of numpy.ndarray
        (szds, vzds, azds)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    szds = _extract_block(lines, "LIDORT - Solar zenith angles (degrees)")
    vzds = _extract_block(lines, "LIDORT - User-defined viewing zenith angles (degrees)")
    azds = _extract_block(lines, "LIDORT - User-defined relative azimuth angles (degrees)")

    return szds, vzds, azds


def parse_lidort_all(
    filebasename: str, row: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    filebasename : str
        _description_
    row : int
        _description_

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        0 is TOA, 1 is BOA
    """
    szds, vzds, azds = parse_lidort_cfg(f"{filebasename}.cfg")
    intensities = np.empty([len(szds), len(vzds), len(azds), 2])

    usecols = list(range(4, 5 + row))

    id_geo = 0
    for i, _ in enumerate(szds):
        for j, _ in enumerate(vzds):
            for k, _ in enumerate(azds):
                id_geo += 1
                data = np.loadtxt(
                    f"{filebasename}.all", skiprows=1 + 5 * id_geo, max_rows=4, usecols=usecols
                ).reshape(4, -1)[[0, -1], :]
                lidort_ans = np.array(data[:, row])
                intensities[i, j, k, :] = lidort_ans
    return szds, vzds, azds, intensities


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Replace 'lidort_input.txt' with the path to your file
    szds, rads, vzds = parse_lidort_cfg("lidort_input.txt")
    print("Solar zenith angles (deg):", szds)
    print("Relative azimuth angles (deg):", rads)
    print("Viewing zenith angles (deg):", vzds)
