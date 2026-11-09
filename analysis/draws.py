"""Lossless float64 storage for coefficient draws and chain/iteration columns."""

from pathlib import Path
import csv
import gzip
import hashlib
import json
import numpy as np


def read_draws(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if str(z["format"]) != "sclara-float64-byte-columns-1":
                raise ValueError("Unknown coefficient storage format")
            shape = tuple(int(x) for x in z["shape"])
            b = z["bytes"]
            if (
                len(shape) != 2
                or b.dtype != np.uint8
                or b.shape != (8, shape[1], shape[0])
            ):
                raise ValueError("Invalid coefficient byte dimensions")
            a = (
                b.transpose(1, 2, 0)
                .copy()
                .view("<f8")
                .reshape(shape[1], shape[0])
                .T.copy()
            )
            header = z["header"].tolist()
            if len(header) != shape[1] or not all(isinstance(s, str) for s in header):
                raise ValueError("Invalid coefficient column names")
        return header, a
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
        a = np.loadtxt(f, delimiter=",", ndmin=2)
    return header, a


def write_draws(path, header, values):
    """Shuffle byte positions before standard ZIP compression; never quantize."""
    a = np.asarray(values, dtype="<f8")
    if a.ndim != 2 or a.shape[1] != len(header) or not np.isfinite(a).all():
        raise ValueError("Finite coefficient matrix and matching columns required")
    byte_columns = (
        a.T.copy()
        .view(np.uint8)
        .reshape(a.shape[1], a.shape[0], 8)
        .transpose(2, 0, 1)
        .copy()
    )
    np.savez_compressed(
        path,
        format=np.array("sclara-float64-byte-columns-1"),
        shape=np.array(a.shape, dtype="<i8"),
        header=np.array(header),
        bytes=byte_columns,
    )
    actual_header, actual = read_draws(path)
    if actual_header != list(header) or not np.array_equal(
        actual.view("<u8"), a.view("<u8")
    ):
        raise ValueError("Lossless draw round-trip failed")
    return hashlib.sha256(a.tobytes()).hexdigest()


def compact_directory(directory):
    """Convert verified CSV exports in a new collection; keep source hashes."""
    directory = Path(directory)
    converted = []
    for source in sorted(directory.glob("*_draws.csv.gz")):
        job = source.name.removesuffix("_draws.csv.gz")
        target = source.with_name(job + "_draws.npz")
        receipt_path = source.with_name(job + ".json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != receipt["draws_sha256"]:
            raise ValueError("CSV source does not match sampling receipt")
        if target.exists():
            raise FileExistsError(target)
        header, a = read_draws(source)
        value_digest = write_draws(target, header, a)
        receipt["draw_storage"] = dict(
            format="sclara-float64-byte-columns-1",
            original_csv_gzip_sha256=digest,
            float64_values_sha256=value_digest,
            rule="Exact float64 values read from the sampler CSV, including every coefficient, chain and iteration. No rounding, thinning or quantization.",
        )
        receipt["draws_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        converted.append(
            dict(
                job=job,
                original_bytes=source.stat().st_size,
                stored_bytes=target.stat().st_size,
                values_sha256=value_digest,
            )
        )
        source.unlink()
    return converted
