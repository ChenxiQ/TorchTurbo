"""LLaDA2 block-diffusion (BD) data helpers.

Each ``BDDataset.__getitem__`` returns one complete CPU BD batch
(``[noisy|clean]`` + 4D block mask + labels). The dataloader is built by
``TorchTurboTrainerBase._get_bd_train_dataloader``; the BD loss math lives in
``FSDP2Backend.compute_diffusion_loss``. Both paths are routed by the model's
``is_diffusion_model`` attribute, not by a model-type string switch.

Highlights
----------
- Pre-tokenize cache uses the model's native chat template, so caching is required
  for precision alignment).
- Per-sample MDM noise: ``sigma ~ U[bd_mask_low, bd_mask_high]`` (default 0.3/0.8 via
  ``--tt_bd_mask_low/high``), ``noise_seed = int(index)``. Same index -> same noise, so
  every SP rank in a group (same index via the SP-aware sampler) gets identical noise.
- LLaDA2 is a diffusion model; BD is its intrinsic training path (gated by
  ``is_diffusion_model`` on the TurboModel subclass, no switch).
- Length derivation: per-sample real truncation limit = LF ``--cutoff_len`` (e.g. 8K);
  total beam length = ``2 * cutoff_len`` (noisy half + clean half, e.g. 16K). Injected via
  ``BDDataset(cutoff_len=...)`` (threaded from ``engine._lf_data_args.cutoff_len`` by the
  trainer). Falls back to module defaults (``BEAM_LEN`` / ``MAX_SEQ_LEN_PACKED``)
  when ``cutoff_len`` is None.
- ``block_diag_attn`` (neat_packing): FFD-packs multiple short samples into one beam;
  cross-sample isolation via the block mask; padding excluded from noise/labels.

Reference: dFactory ``tasks/dataset/data_transform.py`` (``sft_noise_transition``,
``apply_chat_template_mdm``) and ``tasks/train_llada2_common.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch.utils.data import Dataset

from torchturbo.logging import get_logger


logger = get_logger(__name__)

# ``-100`` is the HF / PyTorch / LlamaFactory convention for the loss ignore index. Reuse LF's
# constant in production; fall back to the literal for pure-CPU tests that load this file via
# importlib without llamafactory installed.
try:
    from llamafactory.extras.constants import IGNORE_INDEX
except ImportError:
    IGNORE_INDEX = -100

# dFactory defaults (``llada2_mini_bd_sft.yaml``). At runtime ``mask_id``/``pad_id`` are derived
# from the tokenizer and ``cutoff``/``mask range`` are injected via ``BDDataset``; these literals
# are only defaults for direct unit-test callers and the fallback path. Values match
# LLaDA2.1-mini exactly.
MASK_ID = 156895   # <|mask|>  — runtime: tok.mask_token_id; literal is fallback/test only.
PAD_ID = 156892    # <|endoftext|> = eos = pad — runtime: tok.pad_token_id (= config.pad_token_id).
MASK_LOW, MASK_HIGH = 0.3, 0.8   # default sigma sampling range; overridden by --tt_bd_mask_low/high.
BLOCK_SIZE = 32                  # dFactory block_size (default; overridable via --tt_bd_block_size -> BDDataset).

# Length defaults: used only when ``cutoff_len`` is None (keeps legacy behavior / no drift in the
# fallback path). Production always injects ``cutoff_len``.
BEAM_LEN = 8192  # default packing beam slot length (2^13; divisible by SP=2/4/8/16/32).
MAX_SEQ_LEN_PACKED = BEAM_LEN // 2  # 4096 — default per-sample truncation limit; cutoff_len wins.

# Segment markers for the block mask (internal, not user-facing).
SEG_NOISY, SEG_CLEAN, SEG_PAD = 0, 1, 2

def _align_cutoff_to_block(cutoff_len: int, block_size: int = BLOCK_SIZE) -> int:
    """Ceil ``cutoff_len`` up to a multiple of ``block_size``.

    LlamaFactory subtracts 1 from ``cutoff_len`` under ``neat_packing``/``packing``.
    The BD path aligns on blocks itself, independent of that -1, so we undo it and ceil.
    Already-aligned values pass through.
    """
    if cutoff_len % block_size == 0:
        return cutoff_len
    return ((cutoff_len + block_size - 1) // block_size) * block_size


def resolve_bd_data_file(data_args: Any) -> tuple:
    """Resolve the raw JSONL file + its dataset_info columns/tags.

    Returns ``(file_path, columns, tags)`` so ``build_bd_cache`` can map arbitrary
    sharegpt/conversations schemas to standard ``messages`` (role/content), mirroring
    how LlamaFactory interprets the same ``dataset_info.json`` entry.
    """
    dataset_dir = getattr(data_args, "dataset_dir", None)
    dataset = getattr(data_args, "dataset", None)
    if not dataset_dir or not dataset:
        raise RuntimeError("BD mode needs --dataset_dir and --dataset to resolve the data file.")
    if isinstance(dataset, (list, tuple)):
        dataset_name = dataset[0].strip()
    else:
        dataset_name = dataset.split(",")[0].strip()
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    except OSError as exc:
        raise RuntimeError(f"Cannot read dataset_info.json at {info_path}: {exc}") from exc
    entry = info.get(dataset_name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Dataset {dataset_name!r} not found in {info_path}.")
    file_name = entry.get("file_name")
    if not file_name:
        raise RuntimeError(f"Dataset {dataset_name!r} has no 'file_name' in {info_path}.")
    columns = entry.get("columns", {}) or {}
    tags = entry.get("tags", {}) or {}
    return os.path.join(dataset_dir, file_name), columns, tags


# ---------------------------------------------------------------------------
# Pre-tokenize cache
# ---------------------------------------------------------------------------

def apply_chat_template_mdm(messages, tokenizer, max_length, pad_to_block=False,
                            pad_id=PAD_ID, block_size=BLOCK_SIZE):
    """Apply the chat template with three over-long-prompt defenses.

    Defenses: (1) truncate the prompt too, (2) clamp ``prompt_length <= L``, (3) discard if the
    prompt fills the whole sequence.

    pad_to_block=False: tensor padded to ``max_length`` (non-packing, = cutoff_len).
    pad_to_block=True:  tensor padded to ``ceil(L/block_size)*block_size`` = T_i (packing).
    Returns ``(tensor, prompt_length, L)`` or ``None`` (defense 3 discarded it).
    """
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    # Defense 1: truncate the prompt too (align with the full truncation).
    prompt_ids = tokenizer(prompt_str, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(inputs_str, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]

    prompt_length = len(prompt_ids)
    L = len(full_ids)

    # Defense 2: clamp prompt_length <= L.
    prompt_length = min(prompt_length, L)

    # Defense 3: prompt fills the whole sequence -> discard.
    if prompt_length >= L:
        return None

    # Pad to the target length.
    target = ((L + block_size - 1) // block_size) * block_size if pad_to_block else max_length  # ceil(L/B)*B
    padded = full_ids + [pad_id] * (target - L)
    return torch.tensor(padded, dtype=torch.long), prompt_length, L


def build_bd_cache(model_path: str, data_file: str, pad_to_block: bool = False,
                   max_seq_len: int = MAX_SEQ_LEN_PACKED, block_size: int = BLOCK_SIZE,
                   columns: dict | None = None, tags: dict | None = None):
    """Pre-tokenize every sample (chat template) into a list.

    Returns ``(cache, mask_id, pad_id)`` where ``cache`` is
    ``list[(tensor, prompt_length, L_real)]`` and ``mask_id``/``pad_id`` are derived from the
    tokenizer (returned here so downstream ``build_*`` callers reuse them without re-querying).

    pad_to_block=False: tensor padded to ``max_seq_len`` (= cutoff_len, non-packing).
    pad_to_block=True:  tensor = ``ceil(L/block_size)*block_size`` (= T_i, packing).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_id = tok.mask_token_id if tok.mask_token_id is not None else MASK_ID
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else PAD_ID

    # Map dataset_info columns/tags -> standard messages(role/content). Handles native messages
    # ({role,content}) and sharegpt ({from,value}) via the configured column/tag names.
    columns = columns or {}
    tags = tags or {}
    msgs_col = columns.get("messages", "messages")
    sys_col = columns.get("system")
    role_tag = tags.get("role_tag", "role")
    content_tag = tags.get("content_tag", "content")
    user_tag = tags.get("user_tag", "user")
    asst_tag = tags.get("assistant_tag", "assistant")

    cache: list = []
    dropped = 0
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = []
            for m in row[msgs_col]:
                role = m[role_tag]
                role = "user" if role == user_tag else ("assistant" if role == asst_tag else role)
                msgs.append({"role": role, "content": m[content_tag]})
            sys_text = row.get(sys_col) if sys_col else None
            if sys_text:
                # Prepend as a system turn so apply_chat_template renders it (keeps the BD cache
                # consistent with inference; system is context, not a supervised target).
                msgs = [{"role": "system", "content": sys_text}] + msgs
            # max_seq_len has identical semantics on both paths: per-sample real truncation limit
            # (= cutoff_len). Non-packing: also the tensor pad target; packing: the per-sample
            # truncation cap before FFD.
            item = apply_chat_template_mdm(msgs, tok, max_seq_len, pad_to_block=pad_to_block,
                                           pad_id=pad_id, block_size=block_size)
            if item is None:
                dropped += 1
                continue
            cache.append(item)

    mode = "packed" if pad_to_block else "single"
    logger.info_rank0(
        "[BD-CACHE] mode=%s | loaded %d samples | dropped %d | "
        "max_len=%d | mask_id=%d pad_id=%d | %s",
        mode,
        len(cache),
        dropped,
        max_seq_len,
        mask_id,
        pad_id,
        data_file,
    )
    return cache, mask_id, pad_id


# ---------------------------------------------------------------------------
# Noise + block mask
# ---------------------------------------------------------------------------

def gen_noise_per_sample(T, prompt_length, real_len, seed, mask_low=MASK_LOW, mask_high=MASK_HIGH):
    """Per-sample sigma noise. Returns ``(move[T] bool cpu, sigma float)``.

    ``move`` is only set in the maskable region (``prompt_length <= pos < real_len``), excluding
    prompt and pad. Deterministic: same seed -> identical ``move``.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    sigma = (torch.rand(1, generator=g) * (mask_high - mask_low) + mask_low).item()
    ar = torch.arange(T)
    maskable = (ar >= prompt_length) & (ar < real_len)
    move = (torch.rand(T, generator=g) < sigma) & maskable
    return move, sigma


def block_diffusion_mask_4d(
    sample_ids, segs, n_per_sample, beam_plan, beam_len, block_size, dtype=torch.float32
):
    """Unified 4D block mask. Returns ``[1,1,beam_len,beam_len]`` additive (0/-inf).

    - Per-sample block ids: noisy/clean segments re-number from 0 within each sample.
    - same_sample mask: cross-sample / pad-sample positions are mutually invisible.
    - M_BD / M_OBC / M_BC: intra-block bidirectional / noisy sees earlier clean / clean-segment block-causal.
    - Tail-pad key suppression: ``seg_k == PAD`` -> ``-inf``.
    """
    device = sample_ids.device
    idx = torch.arange(beam_len, device=device)

    starts = []
    cur = 0
    for (_, T_i, _) in beam_plan:
        starts.append(cur)
        cur += 2 * T_i

    seg = segs
    sid = sample_ids
    starts_arr = torch.tensor(starts, device=device, dtype=torch.long)
    T_arr = torch.tensor([t for (_, t, _) in beam_plan], device=device, dtype=torch.long)
    starts_per_pos = starts_arr[sid.clamp(min=0)]
    T_per_pos = T_arr[sid.clamp(min=0)]
    local_pos = idx - starts_per_pos
    block = torch.where(
        seg == SEG_NOISY,
        local_pos // block_size,
        torch.where(seg == SEG_CLEAN, (local_pos - T_per_pos) // block_size, torch.full_like(idx, -2)),
    )

    sid_q, sid_k = sid[:, None], sid[None, :]
    seg_q, seg_k = seg[:, None], seg[None, :]
    bq, bk = block[:, None], block[None, :]
    same_sample = (sid_q == sid_k) & (sid_q != -1)
    bd = (bq == bk) & (seg_q == seg_k)
    obc = (bq > bk) & (seg_k == SEG_CLEAN) & (seg_q == SEG_NOISY)
    bc = (bq >= bk) & (seg_k == SEG_CLEAN) & (seg_q == SEG_CLEAN)
    allow = same_sample & (bd | obc | bc) & (seg_k != SEG_PAD)

    return allow.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Packing (block_diag_attn path)
# ---------------------------------------------------------------------------

def greedy_knapsack(items, beam_len=BEAM_LEN, block_size=BLOCK_SIZE, max_seq_len=MAX_SEQ_LEN_PACKED):
    """FFD greedy bin-packing. ``items``: list of ``(idx, L_real)``. Returns beam_plans.

    Each beam_plan = ``[(idx, T_i, L_clamped), ...]``, where
    ``T_i = ceil(L/block_size)*block_size`` and ``sum(2*T_i) <= beam_len``.
    """
    enriched = []
    for idx, L_real in items:
        L_clamped = min(L_real, max_seq_len)
        T_i = ((L_clamped + block_size - 1) // block_size) * block_size
        enriched.append((idx, T_i, L_clamped, 2 * T_i))
    enriched.sort(key=lambda x: -x[3])

    beams = []
    for idx, T_i, L_clamped, occupy in enriched:
        placed = False
        for beam in beams:
            if beam["used"] + occupy <= beam_len:
                beam["plan"].append((idx, T_i, L_clamped))
                beam["used"] += occupy
                placed = True
                break
        if not placed:
            beams.append({"plan": [(idx, T_i, L_clamped)], "used": occupy})
    return [b["plan"] for b in beams]


def pack_beams(cache, beam_len=BEAM_LEN, block_size=BLOCK_SIZE, max_seq_len=MAX_SEQ_LEN_PACKED):
    """FFD-pack the cache into beam_plans. ``prompt_len`` is taken from the cache's real value."""
    items = [(i, item[2]) for i, item in enumerate(cache)]  # (idx, L_real)
    plans = greedy_knapsack(items, beam_len=beam_len, block_size=block_size, max_seq_len=max_seq_len)
    real_plans = []
    for plan in plans:
        real_plan = [(idx, T_i, cache[idx][1]) for (idx, T_i, _) in plan]  # cache[idx]=(tensor,pl,L)
        real_plans.append(real_plan)
    return real_plans


def build_packed_beam(beam_plan, cache, device="cpu", block_size=BLOCK_SIZE, beam_len=BEAM_LEN,
                      noise_seed_base=0, mask_low=MASK_LOW, mask_high=MASK_HIGH,
                      mask_id=MASK_ID, pad_id=PAD_ID):
    """Pack one beam_plan into a packed batch (packing path, pad excluded).

    Returns ``{input_ids, attention_mask, position_ids, labels}`` (CPU tensors).
    ``mask_low``/``mask_high`` are threaded to ``gen_noise_per_sample``; ``mask_id``/``pad_id``
    are the runtime-derived tokenizer values.
    """
    full_list = []
    pos_list = []
    labels_list = []
    sample_id_list = []
    seg_list = []

    cur = 0
    for s_i, (idx, T_i, pl_i) in enumerate(beam_plan):
        ids_real, pl_real, L_real = cache[idx]
        ids_real = list(ids_real)
        noisy_base = ids_real[:T_i] + [pad_id] * max(0, T_i - len(ids_real[:T_i]))
        clean_base = ids_real[:T_i] + [pad_id] * max(0, T_i - len(ids_real[:T_i]))

        move, sigma = gen_noise_per_sample(T_i, pl_i, min(L_real, T_i), seed=noise_seed_base + idx,
                                           mask_low=mask_low, mask_high=mask_high)
        noisy_noised = list(noisy_base)
        for p in range(T_i):
            if move[p]:
                noisy_noised[p] = mask_id
        labels_i = [IGNORE_INDEX] * T_i
        for p in range(T_i):
            if move[p] and noisy_base[p] != pad_id:
                labels_i[p] = noisy_base[p]
        labels_clean = [IGNORE_INDEX] * T_i

        full_list += noisy_noised + clean_base
        pos_list += list(range(T_i)) + list(range(T_i))
        labels_list += labels_i + labels_clean
        sample_id_list += [s_i] * (2 * T_i)
        seg_list += [SEG_NOISY] * T_i + [SEG_CLEAN] * T_i
        cur += 2 * T_i

    pad_n = beam_len - cur
    if pad_n > 0:
        full_list += [pad_id] * pad_n
        pos_list += [0] * pad_n
        labels_list += [IGNORE_INDEX] * pad_n
        sample_id_list += [-1] * pad_n
        seg_list += [SEG_PAD] * pad_n

    full = torch.tensor([full_list], dtype=torch.long, device=device)
    pos = torch.tensor([pos_list], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    sample_ids = torch.tensor(sample_id_list, dtype=torch.long, device=device)
    segs = torch.tensor(seg_list, dtype=torch.long, device=device)
    n_per_sample = [t for (_, t, _) in beam_plan]
    mask = block_diffusion_mask_4d(sample_ids, segs, n_per_sample, beam_plan, beam_len, block_size)
    return {
        "input_ids": full,
        "attention_mask": mask,
        "position_ids": pos,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Non-packing path (single-sample)
# ---------------------------------------------------------------------------

def bd_build_batch(tensor_padded: torch.Tensor, prompt_length: int, L_real: int, index: int,
                   block_size: int = BLOCK_SIZE,
                   mask_low: float = MASK_LOW, mask_high: float = MASK_HIGH,
                   mask_id: int = MASK_ID, pad_id: int = PAD_ID):
    """Build a full BD batch (CPU) from one cached sample (non-packing path, pad-isolated).

    Equivalent to the single-sample special case of ``build_packed_beam`` (pad excluded). ``T``
    is taken from ``tensor_padded.shape[0]`` (= cutoff_len in the non-packing path, injected by
    BDDataset).

    Returns dict consumed by ``FSDP2Backend.compute_diffusion_loss``:
    - ``input_ids``:      ``[1, 2T]`` = ``cat([noisy, clean])``
    - ``attention_mask``: ``[1, 1, 2T, 2T]`` additive block mask
    - ``position_ids``:   ``[1, 2T]``
    - ``labels``:         ``[1, 2T]`` (noisy-half masked-response positions supervised, rest -100)
    """
    T = tensor_padded.shape[0]  # = max_seq_len (= cutoff_len in the non-packing path)

    # Pad-isolated noise: maskable = prompt..real_len (excludes pad).
    move, sigma = gen_noise_per_sample(T, prompt_length, min(L_real, T), seed=index,
                                       mask_low=mask_low, mask_high=mask_high)

    noisy_base = list(tensor_padded)
    clean_base = list(tensor_padded)
    noisy_noised = list(noisy_base)
    for p in range(T):
        if move[p]:
            noisy_noised[p] = mask_id

    # labels: masked non-pad positions keep the original token, rest -100.
    labels_i = [IGNORE_INDEX] * T
    for p in range(T):
        if move[p] and noisy_base[p] != pad_id:
            labels_i[p] = noisy_base[p]
    labels_clean = [IGNORE_INDEX] * T

    full = torch.tensor([noisy_noised + clean_base], dtype=torch.long)
    pos = torch.tensor([list(range(T)) + list(range(T))], dtype=torch.long)
    labels = torch.tensor([labels_i + labels_clean], dtype=torch.long)
    # Single sample -> sample_id must index the (only) beam_plan entry (index 0), not `index`.
    sid = torch.zeros(2 * T, dtype=torch.long)
    seg = torch.cat([torch.full((T,), SEG_NOISY, dtype=torch.long),
                     torch.full((T,), SEG_CLEAN, dtype=torch.long)])
    block_mask = block_diffusion_mask_4d(sid, seg, [T], [(index, T, prompt_length)], 2 * T, block_size)

    return {
        "input_ids": full,
        "attention_mask": block_mask,
        "position_ids": pos,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BDDataset(Dataset):
    """LLaDA2 block-diffusion dataset.

    Each ``__getitem__`` returns one complete CPU BD batch.
    - ``block_diag_attn=False``: single sample (pad-included); cache padded to ``max_seq_len``
      (= cutoff_len).
    - ``block_diag_attn=True``: FFD-packs multiple samples into a beam; cache padded to T_i
      (block_size multiple); padding excluded from noise/labels.

    Length derivation:
    - ``cutoff_len`` = per-sample real truncation limit (LF ``--cutoff_len``, e.g. 8K).
    - ``beam_len``   = ``2 * cutoff_len`` (noisy half + clean half, e.g. 16K).
      When ``cutoff_len`` is None, falls back to module defaults (``BEAM_LEN`` / ``MAX_SEQ_LEN_PACKED``).

    ``noise_seed = sample/beam index``.
    """

    def __init__(self, model_path: str, data_file: str, block_diag_attn: bool = False,
                 cutoff_len: int | None = None,
                 mask_low: float = MASK_LOW, mask_high: float = MASK_HIGH,
                 block_size: int = BLOCK_SIZE,
                 columns: dict | None = None, tags: dict | None = None) -> None:
        self.block_diag_attn = block_diag_attn
        if block_size < 1:
            raise ValueError(
                f"block_size must be a positive integer, got {block_size}."
            )
        self.block_size = int(block_size)

        # Per-sample real truncation limit = cutoff_len; total beam length = 2*cutoff_len
        # (fall back to module defaults when cutoff_len is None).
        if cutoff_len:
            assert cutoff_len > 0, f"cutoff_len must be positive, got {cutoff_len}"
            # Undo LF neat_packing's ``cutoff_len -= 1`` and ceil to a self.block_size
            # multiple (see _align_cutoff_to_block). self.block_size defaults to
            # BLOCK_SIZE (32) and is overridable via --tt_bd_block_size.
            raw = cutoff_len
            cutoff_len = _align_cutoff_to_block(cutoff_len, self.block_size)
            if cutoff_len != raw:
                logger.info_rank0(
                    "[BD] cutoff_len %d not a block_size(%d) multiple "
                    "(LF neat_packing subtracts 1) -> aligned to %d",
                    raw,
                    self.block_size,
                    cutoff_len,
                )
            self.max_seq_len = cutoff_len
            self.beam_len = 2 * cutoff_len
        else:
            self.max_seq_len = MAX_SEQ_LEN_PACKED
            self.beam_len = BEAM_LEN

        # Noise range: threaded in by patches.py from --tt_bd_mask_low/high (default 0.3/0.8);
        # these defaults apply for direct construction.
        self.mask_low = float(mask_low)
        self.mask_high = float(mask_high)

        if block_diag_attn:
            self.samples, self.mask_id, self.pad_id = build_bd_cache(
                model_path, data_file, pad_to_block=True,
                max_seq_len=self.max_seq_len, block_size=self.block_size,
                columns=columns, tags=tags,
            )
            self.beam_plans = pack_beams(
                self.samples, self.beam_len, self.block_size, self.max_seq_len
            )
            total = sum(len(p) for p in self.beam_plans)
            fill = sum(
                2 * sum(t for _, t, _ in p) for p in self.beam_plans
            ) / max(len(self.beam_plans), 1)
            logger.info_rank0(
                "[BD-PACK] block_diag_attn ON | beams=%d | "
                "packed=%d/%d | avg_fill=%.0f/%d | "
                "max_len=%d mask~U[%s,%s]",
                len(self.beam_plans),
                total,
                len(self.samples),
                fill,
                self.beam_len,
                self.max_seq_len,
                self.mask_low,
                self.mask_high,
            )
        else:
            self.samples, self.mask_id, self.pad_id = build_bd_cache(
                model_path, data_file, pad_to_block=False,
                max_seq_len=self.max_seq_len, block_size=self.block_size,
                columns=columns, tags=tags,
            )
            self.beam_plans = None

    def __len__(self) -> int:
        return len(self.beam_plans) if self.block_diag_attn else len(self.samples)

    def __getitem__(self, index: int):
        if self.block_diag_attn:
            beam_plan = self.beam_plans[index]
            return build_packed_beam(
                beam_plan, self.samples, device="cpu",
                block_size=self.block_size, beam_len=self.beam_len, noise_seed_base=0,
                mask_low=self.mask_low, mask_high=self.mask_high,
                mask_id=self.mask_id, pad_id=self.pad_id,
            )

        tensor_padded, prompt_length, L_real = self.samples[index]
        return bd_build_batch(
            tensor_padded, prompt_length, L_real, index,
            block_size=self.block_size,
            mask_low=self.mask_low, mask_high=self.mask_high,
            mask_id=self.mask_id, pad_id=self.pad_id,
        )
