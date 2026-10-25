#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = DATA / "hox_nox_steady_state_results.csv"
TRAINING = DATA / "training_data_po3.csv"
TRAINER_PATH = HERE.parent / "airborne" / "po3_nn_train.py"

def load_trainer():
    spec = importlib.util.spec_from_file_location("po3_nn_train", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="Steady-state results CSV (default: paper/data/hox_nox_steady_state_results.csv)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=TRAINING,
        help="Training CSV to write, then fit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA,
        help="Directory for the model, scaler, and training summary",
    )
    args, forwarded = parser.parse_known_args()

    trainer = load_trainer()
    print(f"Building the training table from {args.results}")
    trainer.export_training_table(args.results, args.data)

    sys.argv = [
        str(TRAINER_PATH),
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        *forwarded,
    ]
    trainer.main()

if __name__ == "__main__":
    main()
