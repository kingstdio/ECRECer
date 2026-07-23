"""Minimal ESM inference adapter used by ECRECer production workflows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class EsmModelSpec:
    loader_name: str
    repr_layers: Tuple[int, ...]
    embedding_dim: int


MODEL_SPECS = {
    # Default model used to build the existing ECRECer feature bank and Keras heads.
    "esm1b_t33_650M_UR50S": EsmModelSpec(
        loader_name="esm1b_t33_650M_UR50S",
        repr_layers=(0, 32, 33),
        embedding_dim=1280,
    ),
    # Compatible tensor width, but not behavior-equivalent to ESM-1b.
    # Use only for experiments unless downstream models are retrained or validated.
    "esm2_t33_650M_UR50D": EsmModelSpec(
        loader_name="esm2_t33_650M_UR50D",
        repr_layers=(0, 32, 33),
        embedding_dim=1280,
    ),
}

_MODEL_CACHE = {}


def cut_text(text: str, lenth: int) -> List[str]:
    """Split a sequence into fixed-size chunks while preserving the tail."""
    if lenth <= 0:
        raise ValueError("lenth must be positive")
    chunks = re.findall(".{" + str(lenth) + "}", text)
    tail = text[(len(chunks) * lenth) :]
    if tail:
        chunks.append(tail)
    return chunks


def _empty_feature_frame(embedding_dim: int) -> pd.DataFrame:
    columns = ["id"] + ["f" + str(i) for i in range(1, embedding_dim + 1)]
    frame = pd.DataFrame(columns=columns)
    return frame.astype({column: "float32" for column in columns[1:]})


def _resolve_device(device: str | None = None):
    import torch

    if isinstance(device, torch.device):
        return device

    requested = device or os.environ.get("ECRECER_ESM_DEVICE", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _load_model(model_name: str, device=None):
    import esm

    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError("Unsupported ESM model: %s. Supported: %s" % (model_name, supported))

    resolved_device = _resolve_device(device)
    cache_key = (model_name, str(resolved_device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    loader = getattr(esm.pretrained, spec.loader_name)
    model, alphabet = loader()
    model = model.to(resolved_device)
    model.eval()
    if resolved_device.type == "cuda":
        print("Transferred model to GPU")

    batch_converter = alphabet.get_batch_converter()
    _MODEL_CACHE[cache_key] = (model, batch_converter, spec, resolved_device)
    return _MODEL_CACHE[cache_key]


def _mean_representations(results, batch_strs: Sequence[str], layers: Iterable[int]) -> Dict[int, np.ndarray]:
    representations = {layer: tensor.cpu() for layer, tensor in results["representations"].items()}
    layer_values = {}

    for layer in layers:
        chunk_means = []
        tensor = representations[layer]
        for i, seq in enumerate(batch_strs):
            chunk_means.append(tensor[i, 1 : len(seq) + 1].mean(0).numpy())
        layer_values[layer] = np.mean(chunk_means, axis=0)

    return layer_values


def get_rep_single_seq(seqid, sequence, model, batch_converter, seqthres=1022, repr_layers=(0, 32, 33), device=None):
    """Embed one protein sequence and return mean representations by layer."""
    import torch

    sequence = str(sequence)
    data = [(seqid, item) for item in cut_text(sequence, seqthres)] if len(sequence) >= seqthres else [(seqid, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    resolved_device = _resolve_device(device)
    batch_tokens = batch_tokens.to(device=resolved_device, non_blocking=resolved_device.type == "cuda")

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=list(repr_layers), return_contacts=False)

    return {
        "label": batch_labels[0],
        "mean_representations": _mean_representations(results, batch_strs, repr_layers),
    }


def get_rep_multi_sequence(sequences, model="esm1b_t33_650M_UR50S", repr_layers=None, seqthres=1022, device=None):
    """Embed a DataFrame with ``id`` and ``seq`` columns.

    Returns three DataFrames for layers 0, 32, and 33, matching the historical
    ECRECer feature schema: ``id``, then ``f1`` through ``f1280``.
    """
    spec = MODEL_SPECS.get(model)
    if spec is None:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError("Unsupported ESM model: %s. Supported: %s" % (model, supported))

    layers = tuple(repr_layers or spec.repr_layers)
    if tuple(layers) != (0, 32, 33):
        raise ValueError("ECRECer currently expects repr_layers=(0, 32, 33)")

    if len(sequences) == 0:
        empty = _empty_feature_frame(spec.embedding_dim)
        return empty.copy(), empty.copy(), empty.copy()

    esm_model, batch_converter, _, resolved_device = _load_model(model, device=device)

    final_label_list = []
    final_reps = {layer: [] for layer in layers}
    for i in tqdm(range(len(sequences))):
        apd = get_rep_single_seq(
            seqid=sequences.iloc[i].id,
            sequence=sequences.iloc[i].seq,
            model=esm_model,
            batch_converter=batch_converter,
            seqthres=seqthres,
            repr_layers=layers,
            device=resolved_device,
        )
        final_label_list.append(np.array(apd["label"]))
        for layer in layers:
            final_reps[layer].append(np.array(apd["mean_representations"][layer]))

    frames = []
    for layer in layers:
        frame = pd.DataFrame(final_reps[layer])
        frame.insert(loc=0, column="id", value=np.array(final_label_list).flatten())
        frame.columns = ["id"] + ["f" + str(i) for i in range(1, frame.shape[1])]
        frames.append(frame)

    return tuple(frames)


if __name__ == "__main__":
    import config as cfg

    SEQTHRES = 1022
    train = pd.read_feather(cfg.DATADIR + "train.feather").iloc[:, :6]
    rep0, rep32, rep33 = get_rep_multi_sequence(sequences=train, model="esm1b_t33_650M_UR50S", seqthres=SEQTHRES)

    rep0.to_feather(cfg.DATADIR + "train_rep0.feather")
    rep32.to_feather(cfg.DATADIR + "train_rep32.feather")
    rep33.to_feather(cfg.DATADIR + "train_rep33.feather")

    print("Embedding Success!")
