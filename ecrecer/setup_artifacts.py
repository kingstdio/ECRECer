"""Prepare ECRECer runtime artifacts for local inference."""

from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://tibd-public-datasets.s3.amazonaws.com/ecrecer"

MODEL_FILES = {
    "model/isenzyme.h5": BASE_URL + "/model/isenzyme.h5",
    "model/howmany_enzyme.h5": BASE_URL + "/model/howmany_enzyme.h5",
    "model/ec.h5": BASE_URL + "/model/ec.h5",
    "data/featureBank/embd_esm32.feather": BASE_URL + "/data/featureBank/embd_esm32.feather",
    "data/uniprot/sprot_latest.feather": "https://github.com/kingstdio/ECRECer/releases/download/v1.0.5/sprot_latest.feather",
}

OPTIONAL_FILES = {
    "data/uniprot_blast_db/production_blast.dmnd": "https://github.com/kingstdio/ECRECer/releases/download/v1.0.5/production_blast.dmnd",
}

BUNDLED_FILES = {
    "data/dict/dict_label_task1.h5": "dict/dict_label_task1.h5",
    "data/dict/dict_label_task2.h5": "dict/dict_label_task2.h5",
    "data/dict/dict_label_task3.h5": "dict/dict_label_task3.h5",
    "data/sample_10.fasta": "sample_10.fasta",
}


def _format_size(size: int | None) -> str:
    if not size:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _progress(name: str):
    last = {"percent": -1}

    def report(blocks: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        percent = min(100, int(blocks * block_size * 100 / total_size))
        if percent >= last["percent"] + 10 or percent == 100:
            last["percent"] = percent
            print(f"  {name}: {percent}% ({_format_size(total_size)})", flush=True)

    return report


def _copy_bundled(target: Path, overwrite: bool = False) -> None:
    package_root = resources.files("ecrecer.assets")
    for relative_target, relative_source in BUNDLED_FILES.items():
        dest = target / relative_target
        if dest.exists() and not overwrite:
            print(f"exists: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with resources.as_file(package_root / relative_source) as src:
            shutil.copy2(src, dest)
        print(f"copied: {dest}")


def _download_files(target: Path, files: dict[str, str], overwrite: bool = False) -> None:
    for relative_target, url in files.items():
        dest = target / relative_target
        if dest.exists() and not overwrite:
            print(f"exists: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        print(f"downloading: {url}")
        urlretrieve(url, tmp, _progress(dest.name))
        tmp.replace(dest)
        print(f"saved: {dest}")


def prepare_artifacts(target: Path, with_hybrid: bool = False, overwrite: bool = False) -> Path:
    target = target.expanduser().resolve()
    for subdir in ("data/dict", "data/featureBank", "data/uniprot", "model", "results", "tmp"):
        (target / subdir).mkdir(parents=True, exist_ok=True)
    _copy_bundled(target, overwrite=overwrite)
    _download_files(target, MODEL_FILES, overwrite=overwrite)
    if with_hybrid:
        _download_files(target, OPTIONAL_FILES, overwrite=overwrite)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare ECRECer runtime artifacts.")
    parser.add_argument("--target", required=True, help="Artifact directory to create, for example ~/ecrecer_artifacts")
    parser.add_argument("--with-hybrid", action="store_true", help="Also download the DIAMOND database used by hybrid mode, about 186 MB.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifact files.")
    args = parser.parse_args(argv)

    target = prepare_artifacts(Path(args.target), with_hybrid=args.with_hybrid, overwrite=args.overwrite)
    print("\nSetup complete.")
    print(f"Set ECRECER_ROOT before running ECRECer:\n  export ECRECER_ROOT={target}")
    print("Run a sample prediction:\n  ecrecer -i <artifact-root>/data/sample_10.fasta -o ecrecer_sample10.tsv -mode r -topk 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_artifacts", "main"]
