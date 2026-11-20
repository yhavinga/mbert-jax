#!/usr/bin/env python3
"""Arm G4-6 (draft amendment G4-6): preference training (DPO) of Gemma 4 31B-it with LoRA in the translator's SFT stack
(MaxText 0.2.5 train_sft, Tunix's PeftTrainer, qwix LoRA), with G4-0 (the model without the adapter) as the reference.

A pair is (English, chosen Dutch, rejected Dutch): chosen a human Dutch text (a novel's translation, a book's or a
back-translation's original), rejected G4-0's greedy translation of the same English (natural-v1, thinking off). Each
side is an example as MaxText's SFT makes it (g4lora.prompt_text + completion_text, the loss on the Dutch and
<turn|>\\n). The two sides of a pair are two segments next to each other in one packed row, chosen first (segments 2j-1
and 2j), so that the loss reads the pairs from targets_segmentation alone and a batch keeps MaxText's six keys.

The reference policy is the base model: the model before its first step, whose every LoRA B is zero (qwix adds
x A B x scale, which is then exactly zero), so that no second copy of the weights is needed. Its per-segment sums of
log-probabilities are computed once before training, forward only (amendment G4-6.4a: reference_sums, one pass over
every batch of the training file and the preference eval files), and travel with the batches as ref_seg; the training
step runs only the policy's forward and backward pass. Tunix's DPOTrainer (tunix/sft/dpo/dpo_trainer.py
@ a8d7058) takes a separate ref_model (MaxText 0.2.5's train_dpo.py passes nnx.clone(model) and applies no LoRA) and pads
every prompt and response to fixed lengths; its loss is the one of dpo_terms with loss "sigmoid": -(1 - eps) log
sigmoid(beta h) - eps log sigmoid(-beta h), rewards beta x the log-ratio, accuracy chosen reward > rejected reward.

Pure parts (numpy, no JAX): packing pairs into rows, a row as MaxText's SFT batch row (targets shifted as ShiftData
does), the per-process batches of a packed file, and the loss's arithmetic (dpo_terms, numpy or jax.numpy).
JAX parts (on the TPU, imported by sft_lora.py --dpo): without_lora, the per-token log-probabilities from the decoder's
hidden states in pieces along the sequence (MaxText's output head and cross_entropy_with_logits per piece, under
jax.checkpoint), the reference's per-segment sums, and the loss function in the signature of MaxText's SFT loss (with the
batch's ref_seg).
"""

import json
import math
from pathlib import Path

import numpy as np

KEYS = ("inputs", "inputs_position", "inputs_segmentation", "targets", "targets_position", "targets_segmentation")
PAD_ID = 0                # Gemma 4's <pad>: the id MaxText's SFT masks the prompt with (hf_data_processing._get_pad_id)
BOS_ID = 2                # Gemma 4's <bos>: ShiftData's second ignored id
MAX_SEGMENTS = 64         # segments a packed row may hold (32 pairs); the loss's one-hot over the segment ids
REF_KEY = "ref_seg"       # amendment G4-6.4a: a batch's precomputed reference, [rows, max_segments] float32
ROW_KEY = "row"           # the file's row index of each row of a batch (the reference pass only)
LOSSES = ("sigmoid", "ipo")
# end_tokens (amendment G4-6c): the target tokens at the end of each segment (the turn's end, <turn|>\n: 2) that the
# log-ratio leaves out; the NLL term keeps them. 0: every target token in both (G4-6, G4-6b)
DEFAULTS = {"loss": "sigmoid", "beta": 0.1, "label_smoothing": 0.0, "nll_alpha": 1.0, "length_norm": False,
            "max_segments": MAX_SEGMENTS, "end_tokens": 0}
# The sums dpo_terms gives per batch (aux "pref_<name>" in the training stack); a file's eval adds them over its batches.
# The log-probabilities and tokens are those of the log-ratio; the *_all ones those of every target token (the NLL term)
SUMS = ("pairs", "dpo_sum", "correct", "margin_sum", "chosen_reward_sum", "rejected_reward_sum", "chosen_logp_sum",
        "rejected_logp_sum", "ref_chosen_logp_sum", "ref_rejected_logp_sum", "chosen_tokens", "rejected_tokens",
        "chosen_logp_all_sum", "rejected_logp_all_sum", "chosen_tokens_all", "rejected_tokens_all")


def spec(**kw):
    """The preference loss's settings (DEFAULTS, overridden), checked"""
    out = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    unknown = sorted(set(out) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown settings {unknown}")
    if out["loss"] not in LOSSES:
        raise ValueError(f"loss {out['loss']!r}: one of {LOSSES}")
    if not out["beta"] > 0 or not 0 <= out["label_smoothing"] < 0.5 or out["nll_alpha"] < 0:
        raise ValueError(f"beta > 0, 0 <= label_smoothing < 0.5, nll_alpha >= 0: {out}")
    if out["loss"] == "ipo" and out["label_smoothing"]:
        raise ValueError("label smoothing is a setting of the sigmoid loss")
    if out["max_segments"] < 2 or out["max_segments"] % 2:
        raise ValueError(f"max_segments {out['max_segments']}: even, at least 2")
    if isinstance(out["end_tokens"], bool) or not isinstance(out["end_tokens"], int) or out["end_tokens"] < 0:
        raise ValueError(f"end_tokens {out['end_tokens']!r}: an int >= 0")
    out["length_norm"] = bool(out["length_norm"])
    return out


def read_spec(path):
    return spec(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------------------------------- packing

def pack_pairs(lengths, seq, rows, max_segments=MAX_SEGMENTS):
    """[batch: [row: [pair index, ...]]] of pairs with lengths [(chosen side, rejected side), ...] in tokens: the order
    is kept; a pair (both sides, one row) goes to the first row of the open batch with room for its tokens and two more
    segments; a pair that fits no row closes the batch and opens the next (grain's PackAndBatchOperation, MaxText's SFT
    packing, with a pair as the unit). The last batch may have empty rows."""
    batches, cur, free, segs = [], None, None, None
    for k, (a, b) in enumerate(lengths):
        n = a + b
        if n > seq:
            raise ValueError(f"pair {k}: {n} tokens, more than a row of {seq}")
        if cur is None:
            cur, free, segs = [[] for _ in range(rows)], [seq] * rows, [0] * rows
        for i in range(rows):
            if free[i] >= n and segs[i] + 2 <= max_segments:
                break
        else:
            batches.append(cur)
            cur, free, segs = [[] for _ in range(rows)], [seq] * rows, [0] * rows
            i = 0
        cur[i].append(k)
        free[i] -= n
        segs[i] += 2
    if cur is not None:
        batches.append(cur)
    return batches


def row_unshifted(sides, seq, pad_id=PAD_ID):
    """One packed row before ShiftData (KEYS, int32 of length seq), sides = [(prompt ids, completion ids), ...] in
    segment order, as MaxText's SFT pipeline makes it: inputs the concatenation; targets the completion only (the prompt
    as pad_id, SFTPromptMasking); segmentation 1, 2, ... and positions from 0 per example, zeros after (packing,
    ReformatPacking: the targets' segmentation and positions are the inputs')"""
    inputs, targets, seg, pos = (np.zeros(seq, np.int32) for _ in range(4))
    at = 0
    for s, (prompt, completion) in enumerate(sides, 1):
        n = len(prompt) + len(completion)
        if at + n > seq:
            raise ValueError(f"the examples take more than {seq} tokens")
        inputs[at:at + n] = list(prompt) + list(completion)
        targets[at:at + len(prompt)] = pad_id
        targets[at + len(prompt):at + n] = completion
        seg[at:at + n] = s
        pos[at:at + n] = np.arange(n)
        at += n
    return {"inputs": inputs, "inputs_position": pos, "inputs_segmentation": seg, "targets": targets,
            "targets_position": pos.copy(), "targets_segmentation": seg.copy()}


def shift_row(row, pad_id=PAD_ID, bos_id=BOS_ID):
    """ShiftData's shift_and_refine on one row: targets and targets_segmentation one place to the left (the last
    target pad_id, its segment 0), the segmentation 0 where the target is pad_id or bos_id; the positions unchanged"""
    targets = np.append(row["targets"][1:], np.int32(pad_id)).astype(np.int32)
    seg = np.append(row["targets_segmentation"][1:], np.int32(0)).astype(np.int32)
    for ignored in (pad_id, bos_id):
        seg = np.where(targets != ignored, seg, 0).astype(np.int32)
    return {**row, "targets": targets, "targets_segmentation": seg}


def row_arrays(sides, seq, pad_id=PAD_ID, bos_id=BOS_ID):
    """One row of MaxText's SFT batch of packed examples: row_unshifted, then shift_row"""
    return shift_row(row_unshifted(sides, seq, pad_id), pad_id, bos_id)


def pack_rows(pairs, seq, rows, max_segments=MAX_SEGMENTS, pad_id=PAD_ID, bos_id=BOS_ID):
    """The packed rows of pairs ({"id", "prompt", "chosen", "rejected"}, token ids) as a list of dicts (KEYS plus
    "batch", "pairs": the pair ids in segment order, "tokens"), every batch `rows` rows (empty rows all zero)"""
    lengths = [(len(p["prompt"]) + len(p["chosen"]), len(p["prompt"]) + len(p["rejected"])) for p in pairs]
    out = []
    for b, batch in enumerate(pack_pairs(lengths, seq, rows, max_segments)):
        for row in batch:
            sides = []
            for k in row:
                sides += [(pairs[k]["prompt"], pairs[k]["chosen"]), (pairs[k]["prompt"], pairs[k]["rejected"])]
            arrays = row_arrays(sides, seq, pad_id, bos_id)
            out.append({**{key: arrays[key].tolist() for key in KEYS}, "batch": b,
                        "pairs": [pairs[k]["id"] for k in row], "tokens": int(np.count_nonzero(arrays["inputs_segmentation"]))})
    return out


def write_rows(rows, path):
    """A packed file: one parquet row per packed row (KEYS as int32 lists, batch, pairs, tokens)"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([*((k, pa.list_(pa.int32())) for k in KEYS), ("batch", pa.int64()),
                        ("pairs", pa.list_(pa.string())), ("tokens", pa.int64())])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd", row_group_size=1024)


def read_packed(path):
    """{key: int32 array [rows, seq]} of a packed file, and its batch index per row"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[*KEYS, "batch"])
    arrays = {k: np.asarray(table.column(k).to_pylist(), np.int32) for k in KEYS}
    return arrays, np.asarray(table.column("batch").to_pylist(), np.int64)


def batch_count(batch, rows):
    """The global batches of a packed file: its rows in order, `rows` per batch"""
    n = len(batch)
    if n % rows or not np.array_equal(batch, np.repeat(np.arange(n // rows), rows)):
        raise SystemExit(f"a packed file of {n} rows is not batches of {rows} rows in order")
    return n // rows


def local_batches(arrays, process, processes, rows, batches, padding=0, ref=None, row_ids=False):
    """This process's part of each global batch (rows / processes consecutive rows), the first `batches` of the file,
    then `padding` all-zero batches: Tunix loads the next batch before it checks max_steps, so a run of every batch
    still finds one (every process the same, so that none leaves the loop another way). With ref (the reference's
    per-segment sums of the file's rows, [rows, max_segments]) each batch carries its rows' REF_KEY (a padding batch
    zeros); with row_ids each batch carries ROW_KEY, its rows' indices in the file (the reference pass)."""
    if rows % processes:
        raise SystemExit(f"{rows} rows per batch over {processes} processes")
    local = rows // processes
    seq = arrays["inputs"].shape[1]
    if ref is not None and ref.shape[0] < batches * rows:
        raise SystemExit(f"a reference of {ref.shape[0]} rows for {batches} batches of {rows} rows")
    for b in range(batches):
        lo = b * rows + process * local
        out = {k: arrays[k][lo:lo + local] for k in KEYS}
        if ref is not None:
            out[REF_KEY] = np.asarray(ref[lo:lo + local], np.float32)
        if row_ids:
            out[ROW_KEY] = np.arange(lo, lo + local, dtype=np.int32)
        yield out
    for _ in range(padding):
        out = {k: np.zeros((local, seq), np.int32) for k in KEYS}
        if ref is not None:
            out[REF_KEY] = np.zeros((local, ref.shape[1]), np.float32)
        yield out


def global_iterator(path, mesh, rows, batches, padding=0, ref=None, row_ids=False):
    """MaxText's multi-host iterator over a packed file: each process puts its rows of a batch on its devices"""
    import jax
    from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator

    arrays, batch = read_packed(path)
    have = batch_count(batch, rows)
    if batches > have:
        raise SystemExit(f"{path}: {have} batches, {batches} wanted")
    gen = list(local_batches(arrays, jax.process_index(), jax.process_count(), rows, batches, padding, ref, row_ids))
    return MultiHostDataLoadIterator(gen, mesh)


def packed_info(path, rows, seq):
    """{batches, rows, tokens, loss_tokens, pairs} of a packed file of rows of `seq` tokens, `rows` per batch"""
    import pyarrow.parquet as pq

    arrays, batch = read_packed(path)
    if arrays["inputs"].shape[1] != seq:
        raise SystemExit(f"{path}: rows of {arrays['inputs'].shape[1]} tokens, the run's max_target_length is {seq}")
    pairs = sum(len(p) for p in pq.read_table(path, columns=["pairs"]).column("pairs").to_pylist())
    return {"batches": batch_count(batch, rows), "rows": len(batch), "pairs": pairs,
            "tokens": int(np.count_nonzero(arrays["inputs_segmentation"])),
            "loss_tokens": int(np.count_nonzero(arrays["targets_segmentation"]))}


# ---------------------------------------------------------------------------------------------------- the loss

def log_sigmoid(x, xp):
    return -xp.logaddexp(0.0, -x)


def segment_sums(values, segmentation, max_segments, xp=np):
    """[B, max_segments] float32: per row the sum of values [B, L] over each segment id 1..max_segments"""
    onehot = (segmentation[..., None] == xp.arange(1, max_segments + 1)).astype(xp.float32)
    return xp.einsum("bl,bls->bs", values.astype(xp.float32), onehot)


def end_mask(segmentation, n, xp=np):
    """[B, L] float32: 1 at the last n target tokens of each segment, else 0 (amendment G4-6c: the turn's end
    <turn|>\n, n = 2). A segment's target tokens are one run (the completion of MaxText's SFT row after the shift;
    check_ends checks a packed file), so a target token is among the last n of its segment exactly when the token n
    places further is not of its segment."""
    if n == 0:
        return xp.zeros(segmentation.shape, xp.float32)
    pad = xp.zeros((segmentation.shape[0], n), segmentation.dtype)
    ahead = xp.concatenate([segmentation, pad], axis=1)[:, n:]
    return ((segmentation > 0) & (ahead != segmentation)).astype(xp.float32)


def segment_columns(logp, segmentation, n_end, max_segments, xp=np):
    """[K, B, max_segments] float32: per row and segment, in one contraction with one one-hot of the segment ids, the
    sums of the log-probabilities and of the target tokens and, with n_end, the same without the last n_end target
    tokens of every segment (K = 4; else K = 2). Addendum G4-6c.6b: separate sums took the eval step at 4,096 past a v4
    chip's memory. The policy's sums (dpo_terms) and the reference's (reference_sums) come from this one function, so
    that at step 0 they are the same numbers."""
    f32 = xp.float32
    target = (segmentation != 0).astype(f32)
    cols = [logp.astype(f32), target]
    if n_end:
        keep = 1.0 - end_mask(segmentation, n_end, xp)
        cols += [cols[0] * keep, target * keep]
    onehot = (segmentation[..., None] == xp.arange(1, max_segments + 1)).astype(f32)
    return xp.einsum("blk,bls->kbs", xp.stack(cols, axis=-1), onehot)


def ratio_sums(summed, n_end):
    """The log-ratio's per-segment sums of log-probabilities and tokens of segment_columns' output"""
    return (summed[2], summed[3]) if n_end else (summed[0], summed[1])


def check_ends(path, n, end_ids=None):
    """Amendment G4-6c: in every row of a packed file each segment's target tokens are one run of more than n tokens
    (so end_mask takes exactly its last n) and, with end_ids, its last n targets are end_ids. Returns {"rows",
    "segments"}; SystemExit names the first row that does not hold."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["targets", "targets_segmentation"])
    rows = table.num_rows
    seg = np.asarray(table.column("targets_segmentation").combine_chunks().flatten(), np.int32).reshape(rows, -1)
    tgt = np.asarray(table.column("targets").combine_chunks().flatten(), np.int32).reshape(rows, -1)
    mask = end_mask(seg, n)
    segments = 0
    for r in range(rows):
        nz = np.flatnonzero(seg[r])
        if not len(nz):
            continue
        ids = seg[r][nz]
        cut = np.flatnonzero((np.diff(nz) != 1) | (np.diff(ids) != 0)) + 1
        starts, stops = np.concatenate([[0], cut]), np.concatenate([cut, [len(nz)]])
        if len(np.unique(ids[starts])) != len(starts):
            raise SystemExit(f"{path} row {r}: a segment's target tokens are not one run")
        if (stops - starts <= n).any():
            raise SystemExit(f"{path} row {r}: a segment of {int((stops - starts).min())} target tokens, not more than {n}")
        want = np.zeros(seg.shape[1], np.float32)
        for a, b in zip(starts, stops):
            want[nz[b - n:b]] = 1.0
            if end_ids is not None and tgt[r][nz[b - n:b]].tolist() != list(end_ids):
                raise SystemExit(f"{path} row {r} segment {int(ids[a])}: ends in {tgt[r][nz[b - n:b]].tolist()}, "
                                 f"not {list(end_ids)}")
        if not np.array_equal(mask[r], want):
            raise SystemExit(f"{path} row {r}: end_mask is not the last {n} target tokens of each segment")
        segments += len(starts)
    return {"rows": rows, "segments": segments}


def dpo_terms(logp, ref_logp, segmentation, opts, xp=np, ref_seg=None):
    """The preference loss of a batch and its sums (SUMS). logp and ref_logp: per-token log-probabilities of the
    targets [B, L] (0 where targets_segmentation is 0) under the policy and the reference; or, in place of ref_logp
    (then None), ref_seg: the reference's per-segment sums [B, max_segments] (segment_sums of its log-probabilities,
    precomputed: amendment G4-6.4a; with end_tokens, of the log-ratio's tokens as reference_sums gives them);
    segmentation: targets_segmentation. A pair is segments 2j-1 (chosen) and 2j
    (rejected) of a row, both with target tokens; an unpaired or empty segment counts nowhere. Per pair h = (chosen - ref chosen) - (rejected - ref rejected) of the
    summed log-probabilities (each divided by its tokens with length_norm); sigmoid: -(1 - eps) log sigmoid(beta h) - eps
    log sigmoid(-beta h) (Tunix's dpo_loss_fn); ipo: (h - 1 / (2 beta))^2. The loss is the mean over the batch's pairs
    plus nll_alpha x the token-weighted NLL of the chosen sides (the SFT loss of G4-4 on them). Rewards beta x (logp -
    ref); accuracy: chosen reward > rejected reward. With end_tokens = n (amendment G4-6c) the log-ratio, its rewards
    and the sums without _all leave out the last n target tokens of every side; the NLL term takes every target token."""
    f32 = xp.float32
    s = int(opts["max_segments"])
    n_end = int(opts.get("end_tokens", 0))
    if (ref_logp is None) == (ref_seg is None):
        raise ValueError("dpo_terms: either ref_logp or ref_seg")
    summed = segment_columns(logp, segmentation, n_end, s, xp)
    pol_all, ntok_all = summed[0], summed[1]
    pol, ntok = ratio_sums(summed, n_end)
    if ref_seg is None:
        ref = ratio_sums(segment_columns(ref_logp, segmentation, n_end, s, xp), n_end)[0]
    else:
        ref = ref_seg.astype(f32)
    if ref.shape != pol.shape:
        raise ValueError(f"dpo_terms: the reference's sums are {ref.shape}, the policy's {pol.shape}")
    pc, pr, rc, rr = pol[:, 0::2], pol[:, 1::2], ref[:, 0::2], ref[:, 1::2]
    nc, nr = ntok[:, 0::2], ntok[:, 1::2]
    nc_all, nr_all = ntok_all[:, 0::2], ntok_all[:, 1::2]
    valid = ((nc_all > 0) & (nr_all > 0)).astype(f32)
    if opts["length_norm"]:
        dc, dr = xp.maximum(nc, 1.0), xp.maximum(nr, 1.0)
        a, b, c, d = pc / dc, pr / dr, rc / dc, rr / dr
    else:
        a, b, c, d = pc, pr, rc, rr
    h = (a - c) - (b - d)
    beta = float(opts["beta"])
    if opts["loss"] == "sigmoid":
        eps = float(opts["label_smoothing"])
        losses = -(log_sigmoid(beta * h, xp) * (1.0 - eps) + log_sigmoid(-beta * h, xp) * eps)
    elif opts["loss"] == "ipo":
        losses = (h - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"loss {opts['loss']!r}")
    chosen_reward, rejected_reward = beta * (a - c), beta * (b - d)
    pairs = valid.sum()
    chosen_tokens = (nc * valid).sum()
    sums = {"pairs": pairs, "dpo_sum": (losses * valid).sum(),
            "correct": ((chosen_reward > rejected_reward).astype(f32) * valid).sum(),
            "margin_sum": ((chosen_reward - rejected_reward) * valid).sum(),
            "chosen_reward_sum": (chosen_reward * valid).sum(), "rejected_reward_sum": (rejected_reward * valid).sum(),
            "chosen_logp_sum": (pc * valid).sum(), "rejected_logp_sum": (pr * valid).sum(),
            "ref_chosen_logp_sum": (rc * valid).sum(), "ref_rejected_logp_sum": (rr * valid).sum(),
            "chosen_tokens": chosen_tokens, "rejected_tokens": (nr * valid).sum()}
    if n_end:
        sums.update({"chosen_logp_all_sum": (pol_all[:, 0::2] * valid).sum(),
                     "rejected_logp_all_sum": (pol_all[:, 1::2] * valid).sum(),
                     "chosen_tokens_all": (nc_all * valid).sum(), "rejected_tokens_all": (nr_all * valid).sum()})
    else:
        sums.update({"chosen_logp_all_sum": sums["chosen_logp_sum"], "rejected_logp_all_sum": sums["rejected_logp_sum"],
                     "chosen_tokens_all": chosen_tokens, "rejected_tokens_all": sums["rejected_tokens"]})
    loss = sums["dpo_sum"] / xp.maximum(pairs, 1.0) + float(opts["nll_alpha"]) * (
        -sums["chosen_logp_all_sum"] / xp.maximum(sums["chosen_tokens_all"], 1.0))
    return loss, sums


def pref_record(name, step, parts, batches, seconds):
    """One preference file's eval at a step from the per-batch sums (dicts of SUMS): pairs, the mean loss, accuracy,
    margin and rewards per pair, and per token the policy's NLL of the chosen and of the rejected sides and the
    reference's (at step 0 the same), over the log-ratio's tokens; *_all: the policy's over every target token"""
    if len(parts) != batches:
        raise SystemExit(f"eval {name} at step {step}: {len(parts)} batches ran, {batches} planned")
    t = {k: float(sum(p[k] for p in parts)) for k in SUMS}
    if t["pairs"] <= 0:
        raise SystemExit(f"eval {name} at step {step}: no pair")
    n, ct, rt = t["pairs"], max(t["chosen_tokens"], 1.0), max(t["rejected_tokens"], 1.0)
    return {"step": step, "set": name, "kind": "pref", "batches": batches, "pairs": int(round(n)),
            "loss": t["dpo_sum"] / n, "accuracy": t["correct"] / n, "margin": t["margin_sum"] / n,
            "chosen_reward": t["chosen_reward_sum"] / n, "rejected_reward": t["rejected_reward_sum"] / n,
            "chosen_nll": -t["chosen_logp_sum"] / ct, "rejected_nll": -t["rejected_logp_sum"] / rt,
            "ref_chosen_nll": -t["ref_chosen_logp_sum"] / ct, "ref_rejected_nll": -t["ref_rejected_logp_sum"] / rt,
            "chosen_tokens": int(round(t["chosen_tokens"])), "rejected_tokens": int(round(t["rejected_tokens"])),
            "chosen_nll_all": -t["chosen_logp_all_sum"] / max(t["chosen_tokens_all"], 1.0),
            "rejected_nll_all": -t["rejected_logp_all_sum"] / max(t["rejected_tokens_all"], 1.0),
            "chosen_tokens_all": int(round(t["chosen_tokens_all"])),
            "rejected_tokens_all": int(round(t["rejected_tokens_all"])), "seconds": round(seconds, 1)}


def step_record(step, sums):
    """A training step's preference measures (the batch's sums as floats)"""
    n = max(sums["pairs"], 1.0)
    return {"step": step, "pairs": int(round(sums["pairs"])), "loss": sums["dpo_sum"] / n,
            "accuracy": sums["correct"] / n, "margin": sums["margin_sum"] / n,
            "chosen_nll": -sums["chosen_logp_sum"] / max(sums["chosen_tokens"], 1.0),
            "rejected_nll": -sums["rejected_logp_sum"] / max(sums["rejected_tokens"], 1.0),
            "chosen_nll_all": -sums["chosen_logp_all_sum"] / max(sums["chosen_tokens_all"], 1.0)}


# ---------------------------------------------------------------------------------------------------- JAX

def without_lora(model):
    """The reference policy: a copy of the model's graph with every LoRA parameter at zero (the base kernels are the
    same arrays; nothing is copied on the device)"""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    graphdef, lora, rest = nnx.split(model, nnx.LoRAParam, ...)
    if not jax.tree_util.tree_leaves(lora):
        raise SystemExit("--dpo: the model has no LoRA parameter (the reference is the model with them at zero)")
    return nnx.merge(graphdef, jax.tree.map(jnp.zeros_like, lora), rest, copy=True)


def chunked_logp(piece_fn, args, hidden, targets, pieces):
    """Per-token log-probabilities [B, L] (float32) of the targets from the hidden states [B, L, D], in `pieces` pieces
    along the sequence (every row keeps its place on its device): piece_fn(*args, hidden piece, target piece) gives the
    per-token log-probabilities of one piece; under jax.checkpoint one piece at a time (lax.map), so that the backward
    pass recomputes one piece's logits instead of keeping all of them (as MaxText's vocabulary tiling)"""
    import jax
    import jax.numpy as jnp

    b, length, d = hidden.shape
    if length % pieces:
        raise SystemExit(f"--dpo: a sequence of {length} in {pieces} pieces")
    c = length // pieces
    h = jnp.swapaxes(hidden.reshape(b, pieces, c, d), 0, 1)
    y = jnp.swapaxes(targets.reshape(b, pieces, c), 0, 1)
    one = jax.checkpoint(piece_fn)
    out = jax.lax.map(lambda hy: one(*args, hy[0], hy[1]).astype(jnp.float32), (h, y))
    return jnp.swapaxes(out, 0, 1).reshape(b, length)


def model_logp(model, config, data, dropout):
    """The per-token log-probabilities of data's targets (0 off the target tokens) under the model, as MaxText's loss_fn
    runs it (the decoder's hidden states with num_vocab_tiling > 1, its output head, cross_entropy_with_logits without
    z-loss), in num_vocab_tiling pieces along the sequence"""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import max_utils, maxtext_utils

    if config.num_vocab_tiling < 2:
        raise SystemExit("--dpo needs num_vocab_tiling > 1 (the decoder then gives its hidden states)")
    model(decoder_input_tokens=data["inputs"], decoder_positions=data["inputs_position"],
          decoder_segment_ids=data["inputs_segmentation"], encoder_images=None, encoder_image_masks=None,
          enable_dropout=dropout, decoder_target_tokens=data["targets"], decoder_target_mask=data["targets_segmentation"])
    intermediates = nnx.pop(model, nnx.Intermediate).to_pure_dict()
    hidden = maxtext_utils.get_nested_value(intermediates, ("decoder", "hidden_states"))[0]
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    deterministic = not dropout

    def piece(params, rest, h, y):
        # the params as arguments, not captured: MaxText's vocabulary tiling does so for the tied embedding
        local = nnx.merge(graphdef, params, rest, copy=True)
        logits = local.logits_from_hidden_states_for_vocab_tiling(h, deterministic, "train")
        xent, _ = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(y, config.vocab_size), z_loss=0.0)
        return -xent

    logp = chunked_logp(piece, (params, rest), hidden, data["targets"], config.num_vocab_tiling)
    return logp * (data["targets_segmentation"] != 0).astype(jnp.float32)


def reference_sums(model, config, data, max_segments, end_tokens=0):
    """Amendment G4-6.4a: the reference's per-segment sums of log-probabilities [B, max_segments] of a batch, forward
    only: model_logp of the model before its first step (every LoRA B zero, which the caller checks: the LoRA term
    x A B is then exactly zero, so this is the model with its LoRA at zero) on the batch's eval rows, as the eval step's
    policy pass computes them; with end_tokens (amendment G4-6c) without the last end_tokens target tokens of each
    segment, the log-ratio's tokens of dpo_terms"""
    import jax.numpy as jnp

    data = {k: data[k][:config.micro_batch_size_to_eval_on] for k in KEYS}
    seg = data["targets_segmentation"]
    logp = model_logp(model, config, data, False)
    return ratio_sums(segment_columns(logp, seg, end_tokens, max_segments, jnp), end_tokens)[0]


def lora_b_zero(model):
    """Whether every LoRA B of the model is all zero (the model is then its base model), read on this process"""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    leaves = [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(
        model, nnx.LoRAParam)))[0] if str(getattr(path[-1], "key", path[-1])).endswith("_lora_b")]
    if not leaves:
        raise SystemExit("--dpo: the model has no LoRA B")
    zero = jax.jit(lambda xs: jnp.all(jnp.stack([jnp.all(x == 0) for x in xs])))(leaves)
    return bool(np.asarray(zero.addressable_data(0) if hasattr(zero, "addressable_data") else zero))


_REFERENCE_STEPS = {}


def reference_step(config, max_segments, end_tokens=0):
    """The jitted reference pass of a batch -> (reference_sums, the batch's ROW_KEY), one per run (every file the same
    batch shape, so it compiles once)"""
    from flax import nnx

    key = (id(config), max_segments, end_tokens)
    if key not in _REFERENCE_STEPS:
        _REFERENCE_STEPS[key] = nnx.jit(lambda m, d: (reference_sums(m, config, d, max_segments, end_tokens), d[ROW_KEY]))
    return _REFERENCE_STEPS[key]


def precompute_reference(model, config, path, mesh, rows, batches, max_segments, end_tokens=0):
    """Amendment G4-6.4a: reference_sums of the first `batches` global batches of a packed file, one jitted forward
    step, gathered from every process: [batches x rows, max_segments] float32 in the file's row order (each row placed
    by its ROW_KEY, so the mesh's order of the processes does not matter), and the seconds of each batch; with end_tokens
    (amendment G4-6c) the sums leave out the last end_tokens target tokens of each segment, as dpo_terms does"""
    import time

    import jax
    from jax.experimental import multihost_utils

    if not lora_b_zero(model):
        raise SystemExit("--dpo: a LoRA B is not zero before the reference pass (the reference is the base model)")
    step = reference_step(config, max_segments, end_tokens)
    out = np.full((batches * rows, max_segments), np.nan, np.float32)
    seconds = []
    for batch in global_iterator(path, mesh, rows, batches, row_ids=True):
        t0 = time.perf_counter()
        sums, ids = jax.block_until_ready(step(model, batch))
        sums = np.asarray(multihost_utils.process_allgather(sums, tiled=True), np.float32)
        ids = np.asarray(multihost_utils.process_allgather(ids, tiled=True)).reshape(-1)
        if sums.shape != (rows, max_segments) or len(ids) != rows:
            raise SystemExit(f"{path}: the reference pass gave {sums.shape} for rows {ids.shape}")
        out[ids] = sums
        seconds.append(time.perf_counter() - t0)
    if np.isnan(out).any():
        raise SystemExit(f"{path}: {int(np.isnan(out).any(axis=1).sum())} rows without a reference (or a NaN)")
    return out, seconds


def reference_rows(path, ref, batches, rows):
    """Per pair of the first `batches` batches of a packed file: its reference sums (chosen: segment 2j-1, rejected:
    2j of its row), for the record (dpo_ref.parquet)"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["batch", "pairs"]).to_pydict()
    out = []
    for r in range(batches * rows):
        for j, pid in enumerate(table["pairs"][r]):
            out.append({"file": Path(path).name, "batch": int(table["batch"][r]), "row": r, "pair": pid,
                        "ref_chosen_logp": float(ref[r, 2 * j]), "ref_rejected_logp": float(ref[r, 2 * j + 1])})
    return out


def make_loss(config, opts, is_train):
    """The preference loss in the signature of MaxText's SFT loss (train_sft.use_maxtext_loss_function), with the
    batch's precomputed reference (REF_KEY, amendment G4-6.4a): only the policy's pass; aux holds xent_sum and
    total_weights over every target token (the SFT eval files' loss, as MaxText's) and the pairs' sums (pref_<sum>). A
    batch without REF_KEY is an SFT eval file's (aux without the pairs' sums); a training batch always has it."""
    import jax.numpy as jnp

    micro = config.micro_batch_size_to_train_on if is_train else config.micro_batch_size_to_eval_on
    dropout = bool(config.enable_dropout) if is_train else False

    def loss_func(model, inputs, inputs_position, inputs_segmentation, targets, targets_position,
                  targets_segmentation, ref_seg=None):
        data = {"inputs": inputs, "inputs_position": inputs_position, "inputs_segmentation": inputs_segmentation,
                "targets": targets, "targets_position": targets_position,
                "targets_segmentation": targets_segmentation}
        data = {k: v[:micro] for k, v in data.items()}
        logp = model_logp(model, config, data, dropout)
        aux = {"xent_sum": -jnp.sum(logp), "total_weights": jnp.sum(data["targets_segmentation"] != 0)}
        if ref_seg is None:
            if is_train:
                raise ValueError(f"--dpo: a training batch without {REF_KEY} (the precomputed reference)")
            return aux["xent_sum"] / jnp.maximum(aux["total_weights"], 1), aux
        loss, sums = dpo_terms(logp, None, data["targets_segmentation"], opts, jnp, ref_seg=ref_seg[:micro])
        return loss, {**aux, **{f"pref_{k}": v for k, v in sums.items()}}

    return loss_func


def expected_step0(opts):
    """The preference measures at step 0, where the policy is the reference (LoRA's B is zero): h = 0 for every pair,
    so the loss is log 2 (sigmoid) or 1 / (4 beta^2) (ipo), margin and rewards 0, accuracy 0 (no strict win)"""
    loss = math.log(2.0) if opts["loss"] == "sigmoid" else (1.0 / (2.0 * opts["beta"])) ** 2
    return {"loss": loss, "margin": 0.0, "accuracy": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0}
