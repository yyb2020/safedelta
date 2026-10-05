#!/usr/bin/env python3
"""Download publisher-preprocessed public H5AD expression matrices, verify archives.

Source: https://zenodo.org/records/14607156 . These files are expression matrices,
not raw sequencing reads. Do not apply raw-count normalization a second time.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ARCHIVES = {
    "Haber": "29b81705e358ca8546f8a5cff70dfd86",
    "KaggleCrossCell": "37a60cd8f77687c9627ccb73a38442ee",
    "McFarland": "1540ad44297a02eeb83e285a31e42d78",
    "sciplex3": "963e324e2a1605b4e7c0cf7e36c925ca",
}


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--datasets", nargs="+", choices=list(ARCHIVES), default=list(ARCHIVES)
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in a.datasets:
        archive = a.output / f"{name}.h5ad.gz"
        url = f"https://zenodo.org/records/14607156/files/{name}.h5ad.gz?download=1"
        if not archive.exists():
            temporary = archive.with_suffix(".gz.part")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as f,
            ):
                shutil.copyfileobj(response, f, 8 * 1024 * 1024)
            if digest(temporary, "md5") != ARCHIVES[name]:
                raise RuntimeError(f"checksum mismatch: {name}")
            temporary.replace(archive)
        if digest(archive, "md5") != ARCHIVES[name]:
            raise RuntimeError(f"checksum mismatch: {name}")
        target = a.output / f"{name}.h5ad"
        temporary = target.with_suffix(".h5ad.part")
        with gzip.open(archive, "rb") as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
        temporary.replace(target)
        records.append(
            {
                "dataset": name,
                "url": url,
                "archive_md5": ARCHIVES[name],
                "matrix_sha256": digest(target, "sha256"),
                "matrix_bytes": target.stat().st_size,
            }
        )
        print(name, "verified", flush=True)
    (a.output / "download_receipt.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
