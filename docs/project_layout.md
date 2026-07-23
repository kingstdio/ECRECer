# Runtime File Layout

ECRECer separates the Python package from large runtime artifacts. The pip package contains source code and command-line entrypoints only. Data files, feature banks, and trained model weights must be stored outside the package and selected with `ECRECER_ROOT`.

## Package Files

- `ecrecer/`: package metadata and the `ecrecer` command-line entrypoint.
- `tools/`: FASTA parsing, ESM embedding, Keras compatibility, and helper utilities.
- `production.py`: production inference workflow used by the command-line interface.
- `config.py`: path configuration based on `ECRECER_ROOT`.

## Artifact Root

Set `ECRECER_ROOT` to a directory with this structure:

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

The package reads runtime files from `data/` and `model/`, then writes user outputs to the path provided with `-o`.

## Example Command

```bash
export ECRECER_ROOT=/path/to/ecrecer_artifacts
ecrecer -i "$ECRECER_ROOT/data/sample_10.fasta" -o /tmp/ecrecer_sample10.tsv -mode p -topk 5
```

Keep generated output files, private FASTA inputs, and local scratch files outside the installed Python package.
