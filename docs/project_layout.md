# Runtime File Layout

ECRECer separates the Python package from large runtime artifacts. The pip package contains source code and command-line entrypoints only. Data files, feature banks, and trained model weights are stored outside the package and prepared with `ecrecer-setup`, then selected with `ECRECER_ROOT`.

## Package Files

- `ecrecer/`: package metadata and the `ecrecer` command-line entrypoint.
- `tools/`: FASTA parsing, ESM embedding, Keras compatibility, and helper utilities.
- `production.py`: production inference workflow used by the command-line interface.
- `config.py`: path configuration based on `ECRECER_ROOT`.

## Artifact Root

Run `ecrecer-setup --target /path/to/ecrecer_artifacts`, then set `ECRECER_ROOT` to a directory with this structure:

```text
ECRECER_ROOT/
  data/
    dict/
    featureBank/
    uniprot/
  model/
  results/
  tmp/
```

The package reads runtime files from `data/` and `model/`, then writes user outputs to the path provided with `-o`. Use `ecrecer-setup --target /path/to/ecrecer_artifacts --with-hybrid` when hybrid mode should download the optional DIAMOND database.

## Example Command

```bash
export ECRECER_ROOT=/path/to/ecrecer_artifacts
ecrecer -i "$ECRECER_ROOT/data/sample_10.fasta" -o /tmp/ecrecer_sample10.tsv -mode p -topk 5
```

Keep generated output files, private FASTA inputs, and local scratch files outside the installed Python package.
