"""Convert LF-style multimodal jsonl to swift's native format.

Non-conforming rows are rewritten into ``swift_<name>.jsonl`` next to the
input, images following tag order of appearance. ``--check-only`` reports
conformance without writing; exit code 1 on any drop or violation.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_CHECK_PATH = Path(__file__).resolve().parents[3] / "src" / "torchturbo" / "adapters" / "msswift" / "data_check.py"
_spec = importlib.util.spec_from_file_location("data_check", _CHECK_PATH)
data_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_check)
check_row = data_check.check_row

_INDEXED_TAG = re.compile(r"<image:(\d+)>")


def convert_row(row: dict):
    """Rewrite LF/xg indexed tags into the swift standard, or None if invalid."""
    raw = row.get("image", row.get("images", ""))
    paths = [p.strip() for p in (raw if isinstance(raw, list) else str(raw).split(",")) if p.strip()]
    numbers = [int(n) for n in _INDEXED_TAG.findall(row["text"])]
    if not numbers or any(not 1 <= n <= len(paths) for n in numbers):
        return None  # nothing indexed to map, or an out-of-range reference
    out = {key: value for key, value in row.items() if key not in ("text", "image", "images")}
    out["text"] = _INDEXED_TAG.sub("<image>", row["text"])
    out["images"] = [paths[n - 1] for n in numbers]
    return out


def convert_file(path: Path, check_only: bool) -> bool:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    n_conform = sum(1 for row in rows if check_row(row) is None)

    out_rows, dropped = [], []
    for lineno, row in enumerate(rows, 1):
        if check_row(row) is None:
            out_rows.append(row)
            continue
        converted = convert_row(row)
        if converted is not None and check_row(converted) is None:
            out_rows.append(converted)
        else:
            dropped.append(lineno)

    print(f"{path}: {len(rows)} rows, {n_conform} already conform, "
          f"{len(out_rows) - n_conform} converted, {len(dropped)} dropped")
    for lineno in dropped:
        print(f"{path.name}:{lineno}: cannot be converted to the swift standard", file=sys.stderr)

    if check_only:
        return not dropped and n_conform == len(rows)
    out_path = path.parent / f"swift_{path.name}"
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in out_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path} -> {out_path}")
    return not dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="input .jsonl files")
    parser.add_argument("--check-only", action="store_true", help="only report conformance, write nothing")
    args = parser.parse_args()
    results = [convert_file(path, args.check_only) for path in args.files]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
