import hashlib
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "src" / "rbbench" / "vendor" / "openstl"


def test_vendored_cell_unmodified():
    recorded = dict(
        line.split(": ", 1) for line in (VENDOR / "SOURCE.txt").read_text().splitlines() if ": " in line
    )
    actual = hashlib.sha256((VENDOR / "convlstm_modules.py").read_bytes()).hexdigest()
    assert actual == recorded["sha256"].strip()
    assert (VENDOR / "LICENSE").read_text().splitlines()[0].strip() == "Apache License"
