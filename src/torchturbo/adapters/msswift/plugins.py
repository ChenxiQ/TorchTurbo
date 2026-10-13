"""Swift adapter's plugin host: installs ``--tt_plugin`` plugins and dispatches each to a
kind-specific bridge (``DatasetPluginBridge``)."""

from __future__ import annotations

from typing import Callable, Optional

from ...engine.plugins import (
    DatasetPlugin,
    DatasetPrepareContext,
    PluginConsumer,
    PluginManager,
)
from ...logging import get_logger


class SwiftPluginHost(PluginConsumer):
    """Consumes turbo plugins: each ``DatasetPlugin`` is routed to the
    :class:`DatasetPluginBridge`."""

    def __init__(self) -> None:
        self.dataset_bridge = DatasetPluginBridge()

    def matches(self, plugin) -> bool:
        return isinstance(plugin, DatasetPlugin)

    def consume(self, plugin) -> None:
        self.dataset_bridge.add(plugin)

    def install_plugins(self, manager: PluginManager) -> None:
        manager.try_consume(self)


class DatasetPluginBridge:
    """Routes each ``DatasetPlugin`` name per ``provides_dataset``: provided (pack-grouped)
    datasets take over preparation via :meth:`try_takeover`; raw rows register as a swift
    ``DatasetMeta`` so the native pipeline tokenizes + packs via ``samples()``."""

    TAG_TO_SWIFT_KW = {
        "user_tag": "user_role",
        "assistant_tag": "assistant_role",
        "system_tag": "system_role",
        "role_tag": "role_key",
        "content_tag": "content_key",
    }

    def __init__(self) -> None:
        # name -> plugin for every provided dataset
        self._provided: dict[str, object] = {}

    def add(self, plugin) -> None:
        names = plugin.dataset_names()
        provides = [n for n in names if plugin.provides_dataset(n)]
        raws = [n for n in names if n not in provides]
        for name in provides:
            self._provided[name] = plugin
        if raws:
            self._register_swift_dataset(plugin, raws)
        get_logger(__name__).info_rank0(
            "Dataset plugin '%s' ready (provides=%s, raw=%s).",
            plugin.plugin_name, provides, raws,
        )

    def _register_swift_dataset(self, plugin, names: list) -> None:
        """Register ``plugin``'s raw-row datasets under ``names`` as swift ``DatasetMeta``."""
        from swift.dataset import DatasetMeta, register_dataset

        for name in names:
            info = plugin.dataset_info(name)
            register_dataset(
                DatasetMeta(
                    dataset_name=name,
                    dataset_path="torchturbo://dataset-plugin",
                    loader=self._make_loader(plugin, name),
                    preprocess_func=self._build_preprocessor(info),
                ),
                exist_ok=True,
            )

    @staticmethod
    def _make_loader(plugin, name: str):
        from swift.dataset.loader import DatasetLoader

        class _Loader(DatasetLoader):
            def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
                raw = plugin.samples(name, cache_dir=None)
                return dataset_meta.preprocess_func(
                    raw,
                    num_proc=getattr(self, "num_proc", 1),
                    load_from_cache_file=getattr(self, "load_from_cache_file", True),
                    strict=getattr(self, "strict", False),
                    enable_auto_mapping=not getattr(self, "disable_auto_column_mapping", False),
                )

        return _Loader

    # -------------------------------------- dataset-prepare takeover (provides_dataset)
    def try_takeover(self, *, args, template, show_dataset=None,
                     cache_path: str | None = None) -> Optional[list]:
        """Let the plugin that provides ``args.dataset`` build it (``build_dataset``);
        return ``[train, val]``, or ``None`` when no installed plugin provides it."""
        route = self.resolve_dataset(args)
        if route is None:
            return None
        name, plugin = route
        ctx = self.make_ctx(
            plugin.dataset_info(name), args, template,
            cache_path=cache_path, show_dataset=show_dataset,
        )
        train_dataset, val_dataset = plugin.build_dataset(name, ctx)
        # Engage the packing collator iff the dataset is pack-grouped
        if isinstance(train_dataset[0], list):
            template.packing = template.padding_free = True
        get_logger(__name__).info_rank0(
            "Dataset plugin '%s' built '%s' (packed=%s): train=%d, val=%s (max_length=%s).",
            plugin.plugin_name, name, template.packing, len(train_dataset),
            "none" if val_dataset is None else len(val_dataset), ctx.max_length,
        )
        return [train_dataset, val_dataset]

    def collator_for(self, args) -> Optional[Callable]:
        """The collator of the route ``args`` resolves to (the same routing
        ``try_takeover`` applies); ``None`` without a route or when the plugin
        emits pack-grouped rows for the default collator."""
        route = self.resolve_dataset(args)
        if route is None:
            return None
        return route[1].get_collate_fn()

    def resolve_dataset(self, args) -> Optional[tuple[str, object]]:
        """Return ``(name, plugin)`` for the dataset to take over; ``None`` keeps the
        native path. ``args.dataset`` naming a provided dataset wins; with no dataset
        or val_dataset named, a lone provided dataset takes over; several are
        ambiguous and raise."""
        requested = self._requested_names(args)
        for name in requested:
            plugin = self._provided.get(name)
            if plugin is not None:
                return name, plugin
        # Implicit fallback: only when nothing at all was named.
        if requested or getattr(args, "val_dataset", None):
            return None
        if not self._provided:
            return None
        if len(self._provided) > 1:
            raise ValueError(
                f"several plugin-provided datasets and no --dataset to select one: "
                f"{sorted(self._provided)}")
        return next(iter(self._provided.items()))

    @staticmethod
    def _requested_names(args) -> list:
        """The dataset names ``args.dataset`` asks for (str is comma-separated)."""
        raw = getattr(args, "dataset", None)
        if raw is None:
            return []
        if isinstance(raw, str):
            return [n.strip() for n in raw.split(",") if n.strip()]
        return [str(n).strip() for n in raw if str(n).strip()]

    def make_ctx(self, entry: dict, args, template, *, cache_path: str | None = None,
                 show_dataset=None) -> DatasetPrepareContext:
        """Bridge swift ``args``/``template`` into a :class:`DatasetPrepareContext`;
        ``cache_path`` defaults to ``--cached_dataset`` (``List[str]`` -> first)."""
        if cache_path is None:
            cache_path = getattr(args, "cached_dataset", None)
            if isinstance(cache_path, (list, tuple)):
                cache_path = cache_path[0] if cache_path else None
        seed = getattr(args, "data_seed", None)
        if seed is None:
            seed = getattr(args, "seed", 42)
        # Length cap: ``--packing_length``, else ``--max_length`` / the template's.
        max_length = int(
            getattr(args, "packing_length", None) or getattr(args, "max_length", None)
            or template.max_length)
        return DatasetPrepareContext(
            encode_fn=self.make_encode_fn(template),
            preprocess=self.make_preprocess(entry, args),  # threads dataset_num_proc et al.
            show_dataset=show_dataset,
            max_length=max_length,
            cache_path=cache_path,
            num_proc=int(getattr(args, "dataset_num_proc", 1) or 1),
            seed=int(seed),
        )

    # --------------------------------------------- default packed-dataset bridge
    def make_encode_fn(self, template) -> Callable[[dict], dict]:
        """Wrap ``template.encode(record, return_length=True)``; coerce pyarrow proxies to dict."""

        def encode_fn(record) -> dict:
            if not isinstance(record, dict):
                record = dict(record)
            return template.encode(record, return_length=True)

        return encode_fn

    def make_preprocess(self, entry: dict, args) -> Callable:
        """Wrap swift's native ``MessagesPreprocessor`` with dataset args' num_proc et al."""
        preprocessor = self._build_preprocessor(entry)
        num_proc = getattr(args, "dataset_num_proc", 1) or 1
        load_from_cache_file = getattr(args, "load_from_cache_file", True)
        strict = getattr(args, "strict", False)
        enable_auto_mapping = not getattr(args, "disable_auto_column_mapping", False)

        def preprocess(raw_ds):
            return preprocessor(
                raw_ds,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict,
                enable_auto_mapping=enable_auto_mapping,
            )

        return preprocess

    @classmethod
    def _build_preprocessor(cls, entry: dict):
        """Build a swift preprocessor from one ``dataset_info`` entry (keys ``tags``, ``columns``):
        native ``MessagesPreprocessor`` with the entry's role tags, or ``AutoPreprocessor`` when
        no role tags are declared."""
        from swift.dataset.preprocessor.core import AutoPreprocessor, MessagesPreprocessor

        role_tags = entry.get("tags") or {}
        column_map = entry.get("columns") or {}
        if not role_tags:
            return AutoPreprocessor(columns=dict(column_map))
        kwargs = {swift_kw: role_tags[tag] for tag, swift_kw in cls.TAG_TO_SWIFT_KW.items() if role_tags.get(tag)}
        return MessagesPreprocessor(**kwargs, columns=dict(column_map))


__all__ = ["SwiftPluginHost", "DatasetPluginBridge"]
