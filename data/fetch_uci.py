"""Refresh and verify the study CSVs from authoritative UCI archives.

Run from the repository root: ``python data/fetch_uci.py``. Network access is
needed only when refreshing; the checked-in CSVs support offline experiments.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from pandas.testing import assert_frame_equal


DATA_DIR = Path(__file__).resolve().parent
ARCHIVES = {
    "student": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "wine": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "energy": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
}
EXPECTED_SOURCE_SHA256 = {
    "student-por.csv": "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
    "winequality-red.csv": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
    "ENB2012_data.xlsx": "0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4",
}


def _official_file(archive: str, member: str, nested: str | None = None) -> bytes:
    with urlopen(ARCHIVES[archive], timeout=30) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as container:
        if nested is not None:
            with ZipFile(BytesIO(container.read(nested))) as inner:
                data = inner.read(member)
        else:
            data = container.read(member)
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256[member]:
        raise ValueError(f"Unexpected upstream checksum for {member}: {digest}")
    return data


def main() -> None:
    student_raw = _official_file("student", "student-por.csv", nested="student.zip")
    student_official = pd.read_csv(BytesIO(student_raw), sep=";")
    existing = pd.read_csv(DATA_DIR / "student-por.csv")
    assert_frame_equal(existing, student_official, check_exact=True)
    print("student-por.csv: 649 x 33, matches UCI source by parsed values")

    wine_raw = _official_file("wine", "winequality-red.csv")
    wine = pd.read_csv(BytesIO(wine_raw), sep=";")
    if wine.shape != (1599, 12) or "quality" not in wine:
        raise ValueError("Unexpected red-wine table shape or target")
    wine.to_csv(DATA_DIR / "winequality-red.csv", index=False)
    print("winequality-red.csv: 1599 x 12, target quality")

    energy_raw = _official_file("energy", "ENB2012_data.xlsx")
    energy = pd.read_excel(BytesIO(energy_raw))
    if energy.shape != (768, 10) or list(energy.columns) != [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"
    ]:
        raise ValueError("Unexpected energy table shape or columns")
    energy = energy.drop(columns="Y2").rename(columns={"Y1": "heating_load"})
    energy.to_csv(DATA_DIR / "energy-heating.csv", index=False)
    print("energy-heating.csv: 768 x 9, target heating_load; Y2 excluded")

    for name in ("student-por.csv", "winequality-red.csv", "energy-heating.csv"):
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        print(f"{name} SHA256 {digest}")


if __name__ == "__main__":
    main()
