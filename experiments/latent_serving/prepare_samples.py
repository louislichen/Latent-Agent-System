"""Freeze the 100 GSM8K inputs used by the benchmark drivers."""

import argparse
import json
from pathlib import Path
import random
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="GSM8K main test.jsonl")
    parser.add_argument("out", type=Path, help="Path for the frozen samples.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    indices = random.Random(17).sample(range(2, len(records)), 100)
    samples = []
    for index in indices:
        item = records[index]
        answer = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", item["answer"])
        if answer is None:
            raise ValueError(f"GSM8K row {index} has no numeric gold answer")
        samples.append(
            {
                "dataset_index": index,
                "question": item["question"],
                "solution": item["answer"],
                "gold": answer.group(1).replace(",", ""),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(samples, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
