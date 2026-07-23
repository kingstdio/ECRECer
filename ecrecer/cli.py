"""Command-line entrypoint for ECRECer."""

import argparse


def main(argv=None):
    import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file (fasta format)", type=str, default=cfg.DATADIR + "sample_10.fasta")
    parser.add_argument("-o", help="output file (tsv table)", type=str, default=cfg.RESULTSDIR + "sample_10_2023_07_18.tsv")
    parser.add_argument("-mode", help="compute mode. p: prediction, r: recommendation, h:hybrid", type=str, default="r")
    parser.add_argument("-topk", help="recommendation records, min=1, max=20", type=int, default=50)
    args = parser.parse_args(argv)

    from production import initialize_pandarallel, step_by_step_run

    initialize_pandarallel()
    step_by_step_run(input_fasta=args.i, output_tsv=args.o, mode=args.mode, topnum=args.topk)


if __name__ == "__main__":
    main()


__all__ = ["main"]
