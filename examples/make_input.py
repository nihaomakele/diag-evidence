"""Create a synthetic HTTP-run input without real measurements or labels."""
import argparse
import json
from pathlib import Path

from diag_evidence.demo import synthetic_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump({"signal": synthetic_signal(), "input_is_synthetic": True,
                   "sample_info": {"source_file": "mathematical waveform"}}, handle, allow_nan=False)
    print(args.out)


if __name__ == "__main__":
    main()
