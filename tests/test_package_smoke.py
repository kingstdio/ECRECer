import importlib
import os
import subprocess
import sys

import pandas as pd


def test_package_version_is_available():
    import ecrecer

    assert ecrecer.__version__


def test_console_entrypoint_imports():
    from ecrecer.cli import main

    assert callable(main)


def test_config_uses_ecrecer_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ECRECER_ROOT", str(tmp_path))
    sys.modules.pop("config", None)

    config = importlib.import_module("config")

    assert config.ROOTDIR == str(tmp_path) + os.sep
    assert config.DATADIR == str(tmp_path / "data") + os.sep


def test_esm_empty_frame_schema():
    from tools.embedding_esm import get_rep_multi_sequence

    sequences = pd.DataFrame(columns=["id", "seq"])
    rep0, rep32, rep33 = get_rep_multi_sequence(sequences, model="esm1b_t33_650M_UR50S")

    assert rep0.shape == (0, 1281)
    assert rep32.shape == (0, 1281)
    assert rep33.shape == (0, 1281)
    assert rep32.columns[:5].tolist() == ["id", "f1", "f2", "f3", "f4"]


def test_module_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "ecrecer.cli", "-h"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "-mode" in result.stdout


def test_production_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "production", "-h"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "-mode" in result.stdout


def test_recommendation_format_uses_six_decimals():
    from production import format_recommendations

    result = format_recommendations([0.1, 0.987654321, 1.0e-7], ["1.1.1.1", "2.2.2.2", "3.3.3.3"], 3)

    assert result == "[('2.2.2.2', 0.987654), ('1.1.1.1', 0.100000), ('3.3.3.3', 0.000000)]"


def test_setup_artifacts_copies_bundled_files(monkeypatch, tmp_path):
    from ecrecer import setup_artifacts

    downloaded = []

    def fake_download(target, files, overwrite=False):
        downloaded.extend(sorted(files))

    monkeypatch.setattr(setup_artifacts, "_download_files", fake_download)

    root = setup_artifacts.prepare_artifacts(tmp_path)

    assert (root / "data/sample_10.fasta").exists()
    assert (root / "data/dict/dict_label_task1.h5").exists()
    assert "model/ec.h5" in downloaded
    assert "data/featureBank/embd_esm32.feather" in downloaded
    assert "data/uniprot/sprot_latest.feather" in downloaded
