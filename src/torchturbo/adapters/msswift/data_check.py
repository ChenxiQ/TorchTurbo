"""Gate: images as a list of single paths, one <image> tag per image."""

import logging

logger = logging.getLogger(__name__)

_TEXT_KEYS = ("messages", "text", "query", "response", "history")


def _entry_path(entry) -> str:
    """Path of an images entry: the string itself, or 'path' of a dict produced
    by swift's preprocessor. '' when the entry has no usable path."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("path") or ""
    return ""


def check_row(row: dict) -> "str | None":
    """Return the violation string, or None when the row meets the standard."""
    images = row.get("images", row.get("image"))
    if images is None:  # text-only row
        return None
    if isinstance(images, str):
        return "comma-joined image path" if "," in images else "images must be a list of paths"
    if not isinstance(images, list):
        return "images must be a list of paths"
    for entry in images:
        path = _entry_path(entry)
        if not path:
            return f"image entry without a path: {entry!r:.60}"
        if "," in path:
            return "comma-joined image path"
    text = " ".join(str(row.get(key)) for key in _TEXT_KEYS if key in row)
    n_tags = text.count("<image>")
    if n_tags != len(images):
        return f"<image> tag count {n_tags} != image count {len(images)}"
    return None


def validate_raw_dataset(dataset, name: str = "train") -> None:
    """Raise ValueError on the first violating row; image-less and streaming datasets are skipped."""
    if dataset is None:
        return
    columns = getattr(dataset, "column_names", None)
    if columns is not None and not ({"images", "image"} & set(columns)):
        return
    if not hasattr(dataset, "__len__"):
        return  # streaming dataset
    for i, row in enumerate(dataset):
        reason = check_row(row)
        if reason is not None:
            raise ValueError(
                f"{name} dataset row {i}: {reason}. Normalize the data offline"
                " with scripts/dataset/convert_lf_to_swift.py first."
            )
    logger.info("%s dataset: %d rows conform to the swift data standard", name, len(dataset))
