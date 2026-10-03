"""Shared text-file reading helpers for the log parsers."""

from pathlib import Path
from typing import List, Optional

# Logs are normally UTF-8; UTF-16 and latin-1 are fallbacks for odd exports.
_ENCODINGS = ("utf-8", "utf-16", "latin-1")


def _sniff_bomless_utf16(log_path: Path) -> Optional[str]:
    """Return "utf-16-le" or "utf-16-be" for UTF-16 text written without a BOM.

    JSON logs start with an ASCII character, so in UTF-16 the first two bytes
    are that character plus a NUL. The "utf-16" codec needs a BOM, and the
    utf-8 attempt would otherwise succeed on the NUL bytes and give mojibake.
    """
    with open(log_path, "rb") as f:
        head = f.read(2)
    if len(head) == 2:
        if head[0] != 0 and head[1] == 0:
            return "utf-16-le"
        if head[0] == 0 and head[1] != 0:
            return "utf-16-be"
    return None


def _read_lines_with_fallback(log_path: Path) -> Optional[List[str]]:
    """Read a text log, retrying with fallback encodings.

    Returns the lines, or None if no encoding could decode the file.

    FileNotFoundError and other OSErrors propagate to the caller.
    """
    sniffed = _sniff_bomless_utf16(log_path)
    encodings = (sniffed,) + _ENCODINGS if sniffed else _ENCODINGS
    for encoding in encodings:
        try:
            with open(log_path, encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    return None
