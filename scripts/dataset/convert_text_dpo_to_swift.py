"""Convert text DPO JSONL to Swift's chosen/rejected message format."""

import argparse
import json
from pathlib import Path


def convert_row(row: dict) -> dict | None:
    query = row.get("question") or row.get("query") or ""
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if chosen is None or rejected is None:
        return None

    system = row.get("system")
    messages = []
    rejected_messages = []
    if system:
        system_message = {"role": "system", "content": system}
        messages.append(system_message.copy())
        rejected_messages.append(system_message.copy())
    messages.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": chosen},
        ]
    )
    rejected_messages.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": rejected},
        ]
    )
    return {
        "query": query,
        "messages": messages,
        "rejected_messages": rejected_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="input .jsonl files")
    args = parser.parse_args()
    for path in args.files:
        out_path = path.parent / f"swift_{path.name}"
        kept = dropped = 0
        with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                converted = convert_row(json.loads(line))
                if converted is None:
                    dropped += 1
                    continue
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{path}: {kept} rows converted, {dropped} dropped -> {out_path}")


if __name__ == "__main__":
    main()
