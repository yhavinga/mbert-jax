#!/usr/bin/env python3
"""A translation model in MaxText on the chips of one host: this worker's segments from English to Dutch (or, with
--source-lang and --target-lang, between other languages of LANGUAGES), greedy, in resumable shards.

Usage (translate.sh, on each worker, as a TPU of its own 4 chips):
    python translate.py --segments <plan>/segments-w0.parquet --out gs://.../out/w0 --ckpt gs://.../0/items \\
        --tokenizer /dev/shm/flan_nl/tokenizer-27b --model gemma3-27b --length-class small
The translation is in the column text_<target language> of the shards (text_nl for English to Dutch).

--model names a model of translate_models.yml (beside this file): its MaxText config, its checkpoint per region (the
default of --ckpt, for the region of this VM), its stop tokens, its prompt, its length classes and its slots. The
models are TranslateGemma 12B and 27B (gemma3-12b, gemma3-27b) and Gemma 4 31B-it (gemma4-31b).

The model runs in MaxText's MaxEngine: a KV cache of fixed shape (slots x max_target_length), prompts padded to a few
prefill lengths, continuous batching over the slots (decode_all), greedy decoding, a stop at the model's stop tokens.
The tensor axis of the mesh is the process's chips. Segments are taken in shards of consecutive ids (similar lengths,
as the plan sorts them); a shard is written to the bucket at once when it is done, so a restart skips every shard
that is there. A segment that hits the decode limit is written with ended = false; the assemble step drops its
examples.

The prompt is a setting of the model. TranslateGemma's (prompt translategemma) is its own chat template
(chat_template.jinja of the Hugging Face repo): a user turn with {type: text, source_lang_code: en, target_lang_code:
nl, text} (the defaults of --source-lang and --target-lang), which the template turns into "You are a professional
English (en) to Dutch (nl) translator. ... Please translate the following English text into Dutch:\n\n\n<text>".
Other models get TranslateGemma's words (prompt chat-v1, user_text) in one user turn of their own chat format.
A segment with context (a column `context` of the segments: a list of {label, source, translation}, the earlier turns
of its conversation, and optionally a column `label`, the speaker of the segment) gets the model's context_prompt,
which puts the earlier turns before the current turn; only the current turn is translated.
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import yaml

MODELS_FILE = Path(__file__).with_name("translate_models.yml")
CONFIG = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
MODELS = CONFIG["models"]
PROMPTS = CONFIG["prompts"]
# TranslateGemma's classes and slots, under the names of FLAN-nl's translate.py
LENGTH_CLASSES = CONFIG["length_classes"]
CLASS_ORDER = list(LENGTH_CLASSES)
SLOTS = {name: model["slots"] for name, model in MODELS.items()}
# generation_config.json of TranslateGemma: <eos> and <end_of_turn>; a model's own are its eos_ids
EOS_IDS = [1, 106]
_MARK = "\ue000"  # a private-use character, never in FLAN: where the text goes in the rendered template


# The language codes this script takes, with the names the chat template gives them
LANGUAGES = {"en": "English", "nl": "Dutch"}


def messages(text, source="en", target="nl"):
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": source, "target_lang_code": target,
                                          "text": text}]}]


class PromptCodec:
    """Prompt ids of a segment: the template rendered once around a mark, checked against the template itself."""

    def __init__(self, tokenizer, source="en", target="nl"):
        self.tokenizer, self.source, self.target = tokenizer, source, target
        rendered = tokenizer.apply_chat_template(messages(_MARK, source, target), tokenize=False,
                                                 add_generation_prompt=True)
        self.prefix, self.suffix = rendered.split(_MARK)
        pair = f"{LANGUAGES[source]} ({source}) to {LANGUAGES[target]} ({target})"
        if pair not in self.prefix or not self.suffix.endswith("<start_of_turn>model\n"):
            raise RuntimeError(f"unexpected chat template: {rendered!r}")

    def encode(self, text):
        return self.tokenizer(self.prefix + text + self.suffix, add_special_tokens=False)["input_ids"]

    def check(self, texts):
        for text in texts:
            expected = self.tokenizer.apply_chat_template(messages(text, self.source, self.target), tokenize=True,
                                                          add_generation_prompt=True)
            if isinstance(expected, dict) or hasattr(expected, "keys"):
                expected = expected["input_ids"]
            if list(expected) != self.encode(text):
                raise RuntimeError(f"prompt ids differ from the chat template's for {text[:60]!r}")


def speaker_attribute(label):
    """' speaker="<label>"' for a turn's tag, or nothing without a label."""
    if not label:
        return ""
    if '"' in label or "\n" in label:
        raise ValueError(f"a speaker label without quotes or newlines, not {label!r}")
    return f' speaker="{label}"'


def user_text(prompt, text, source="en", target="nl", context=None, label=None):
    """The user turn of a prompt of kind instruct (translate_models.yml). Without context: the plain request with the
    text (chat-v1: TranslateGemma's own words). With context, the earlier turns of the conversation, oldest first, each
    {label, source, translation} (label: the speaker, e.g. Gebruiker, Assistent or "De heer Thijssen (PvdA)"; may be
    absent), and then the current turn, of speaker label. The same function makes the prompt of a translation and of a
    fine-tune example, so that the model sees at translation what it learned on; the model's chat turn goes around it.
    """
    if prompt.get("kind") != "instruct":
        raise ValueError(f"user_text needs a prompt of kind instruct, not {prompt.get('kind')!r}")
    names = {"source": LANGUAGES[source], "target": LANGUAGES[target], "source_code": source, "target_code": target}
    if not context:
        return prompt["plain"].format(text=text.strip(), **names)
    turns = "\n".join(prompt["turn"].format(speaker=speaker_attribute(turn.get("label")),
                                            source_text=turn["source"].strip(),
                                            translation=turn["translation"].strip(), **names)
                      for turn in context)
    return prompt["context"].format(turns=turns, text=text.strip(), speaker=speaker_attribute(label), **names)


class TurnCodec:
    """Prompt ids of a segment in a prompt of kind instruct: user_text between the strings of the model's user turn.
    With check, the strings are checked against the tokenizer's chat template (a plain-text user turn)."""

    def __init__(self, tokenizer, prompt, turn, source="en", target="nl"):
        self.tokenizer, self.prompt, self.turn, self.source, self.target = tokenizer, prompt, turn, source, target

    def text(self, text, context=None, label=None):
        return (self.turn["prefix"] + user_text(self.prompt, text, self.source, self.target, context, label)
                + self.turn["suffix"])

    def encode(self, text, context=None, label=None):
        return self.tokenizer(self.text(text, context, label), add_special_tokens=False)["input_ids"]

    def check(self, texts, contexts=None, labels=None):
        if not self.turn.get("check"):
            return
        n = len(texts)
        for text, context, label in zip(texts, contexts or [None] * n, labels or [None] * n):
            user = user_text(self.prompt, text, self.source, self.target, context, label)
            expected = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                                          add_generation_prompt=True)
            if isinstance(expected, dict) or hasattr(expected, "keys"):
                expected = expected["input_ids"]
            if list(expected) != self.encode(text, context, label):
                raise RuntimeError(f"prompt ids differ from the chat template's for {text[:60]!r}")


class ModelPrompts:
    """A model's prompt ids: its prompt for a segment without context, its context_prompt for one with (the speaker
    label of the current turn counts only with context)."""

    def __init__(self, tokenizer, model, source="en", target="nl"):
        self.source, self.target = source, target
        self.plain = self._codec(tokenizer, model, model["prompt"])
        name = model.get("context_prompt")
        self.with_context = self._codec(tokenizer, model, name) if name else None

    def _codec(self, tokenizer, model, name):
        prompt = PROMPTS[name]
        if prompt["kind"] == "translategemma":
            return PromptCodec(tokenizer, self.source, self.target)
        return TurnCodec(tokenizer, prompt, model["turn"], self.source, self.target)

    def encode(self, text, context=None, label=None):
        if not context:
            return self.plain.encode(text)
        if self.with_context is None:
            raise ValueError("this model has no context_prompt, and the segment has context")
        return self.with_context.encode(text, context, label)

    def check(self, texts, contexts=None, labels=None):
        n = len(texts)
        rows = list(zip(texts, contexts or [None] * n, labels or [None] * n))
        plain = [t for t, c, _ in rows if not c]
        with_context = [r for r in rows if r[1]]
        if plain:
            self.plain.check(plain)
        if with_context:
            self.with_context.check(*(list(x) for x in zip(*with_context)))


def vm_region():
    """The region of this VM (europe-west4 for a zone europe-west4-a), from the metadata server."""
    request = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/zone",
                                     headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=10) as response:
        zone = response.read().decode().rsplit("/", 1)[-1]
    return zone.rsplit("-", 1)[0]


def checkpoint_of(model, region):
    """The model's checkpoint in the bucket of region; no checkpoint there is an error (nothing crosses regions)."""
    try:
        return model["checkpoints"][region]
    except KeyError:
        raise SystemExit(f"{model['maxtext_model']} has no checkpoint in {region} (translate_models.yml); "
                         f"give --ckpt") from None


def make_result_tokens_a_pytree():
    """Without JetStream, MaxText's engine returns a plain stand-in for JetStream's ResultTokens (a flax struct
    dataclass, so a pytree), and jit cannot return a plain object. Registered here as JetStream's is: arrays as children,
    the index tuples as static data. A MaxText whose stub is a pytree already (the fork yhavinga/maxtext, branch
    fix-stub-resulttokens-pytree) refuses a second registration with a ValueError, which is then the sign that nothing
    is needed, as in MaxText's own sliding_window_decode_test.py. Returns whether it registered the stub."""
    import jax
    from maxtext.inference.maxengine import maxengine

    api = maxengine.engine_api
    if not getattr(api, "_IS_STUB", False):
        return False
    cls = api.ResultTokens

    def flatten(r):
        return (r.data, r.log_prob), (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)

    def unflatten(aux, children):
        return cls(data=children[0], log_prob=children[1], tokens_idx=aux[0], valid_idx=aux[1], length_idx=aux[2],
                   samples_per_slot=aux[3])

    try:
        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    except ValueError as error:
        if "Duplicate" not in str(error):
            raise
        return False
    return True


def decode_cache_fix_present():
    """Whether MaxText makes the NNX decode cache from shapes inside its init jit (the fork yhavinga/maxtext, branch
    fix-nnx-decode-cache-abstract-init: jax.eval_shape over _nnx_init_cache_dict in _init_decode_state_nnx)."""
    import inspect

    from maxtext.inference.maxengine import maxengine

    return "eval_shape" in inspect.getsource(maxengine.MaxEngine._init_decode_state_nnx)


def make_decode_cache_sharded():
    """MaxText 0.2.5 makes the zeros of the whole decode cache eagerly, unsharded, on one chip, before its jit makes them
    again sharded (MaxEngine._init_decode_state_nnx): a cache larger than one chip's free memory fails there with
    RESOURCE_EXHAUSTED (pilot of 8 Nov, 47 GB of cache). That is the one call for the autoregressive cache outside a
    jit; given shapes instead of arrays, the jit alone makes the zeros, sharded. The prefill calls are traced and keep
    MaxText's own code."""
    import jax
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
    from maxtext.inference.maxengine import maxengine

    original = maxengine.MaxEngine._nnx_init_cache_dict

    def init_cache_dict(self, mode=maxengine.MODEL_MODE_PREFILL):
        if mode != MODEL_MODE_AUTOREGRESSIVE:
            return original(self, mode)
        _, cache_state, _ = nnx.split(self._abstract_model_for_mode(mode), nnx.Cache, ...)
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nnx.to_pure_dict(cache_state))

    maxengine.MaxEngine._nnx_init_cache_dict = init_cache_dict


def maxtext_source():
    """The installed MaxText: its version and, for an install from git, the URL and commit (the version on main and
    on the fork is still 0.2.5)."""
    import importlib.metadata

    dist = importlib.metadata.distribution("maxtext")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {"version": dist.version, "url": direct.get("url"), "commit": direct.get("vcs_info", {}).get("commit_id")}


def patch_maxtext():
    """The two workarounds, each only where this MaxText still needs it: {name: applied}."""
    applied = {"result_tokens_pytree": make_result_tokens_a_pytree(), "decode_cache_sharded": False}
    if not decode_cache_fix_present():
        make_decode_cache_sharded()
        applied["decode_cache_sharded"] = True
    return applied


def make_engine(args, shapes, chips=4):
    """MaxText's MaxEngine with the converted checkpoint: (engine, params)."""
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine

    max_prefill = shapes["prefill_lengths"][-1]
    config = pyconfig.initialize(
        [""], model_name=MODELS[args.model]["maxtext_model"], load_parameters_path=args.ckpt, run_name="flan_nl",
        log_config=False,
        # Not the home directory: the boot disk is nearly full
        base_output_directory=f"{args.work_dir}/maxtext_output",
        # pyconfig points JAX's compilation cache here; its default is ~/jax_cache on the boot disk
        jax_cache_dir=f"{args.work_dir}/jax_cache",
        enable_checkpointing=False, async_checkpointing=False, skip_jax_distributed_system=True, scan_layers=False,
        use_multimodal=False, attention="dot_product", dtype="bfloat16", weight_dtype="bfloat16",
        ici_fsdp_parallelism=1, ici_tensor_parallelism=chips, per_device_batch_size=shapes["slots"] / chips,
        max_prefill_predict_length=max_prefill, max_target_length=max_prefill + shapes["max_decode"],
        decode_sampling_strategy="greedy", return_log_prob=False, tokenizer_type="huggingface",
        tokenizer_path=args.tokenizer)
    engine = maxengine.MaxEngine(config)
    return engine, engine.load_params()


def lora_modules(adapter):
    """{module path: (lora_a, lora_b)} from an adapter of sft_lora.py (keys <module path>/kernel_lora_a and _b)."""
    modules = {}
    for key, value in adapter.items():
        module, _, name = key.rpartition("/")
        if name not in ("kernel_lora_a", "kernel_lora_b"):
            raise ValueError(f"unexpected key in the adapter: {key}")
        modules.setdefault(module, {})[name[-1]] = value
    incomplete = sorted(m for m, sides in modules.items() if set(sides) != {"a", "b"})
    if incomplete:
        raise ValueError(f"adapter modules without both lora_a and lora_b: {incomplete[:3]}")
    return {m: (sides["a"], sides["b"]) for m, sides in modules.items()}


def kernel_nodes(tree, path=()):
    """{"/"-joined module path: the dict that holds its "kernel"} of a nested dict of parameters."""
    nodes = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            nodes.update(kernel_nodes(value, path + (str(key),)))
        elif key == "kernel":
            nodes["/".join(path)] = tree
    return nodes


def lora_targets(adapter, tree):
    """[(module, node, lora_a, lora_b)]: each adapter module with the one kernel whose path ends in its path."""
    nodes = kernel_nodes(tree)
    targets = []
    for module, (a, b) in sorted(lora_modules(adapter).items()):
        matches = [p for p in nodes if p == module or p.endswith("/" + module)]
        if len(matches) != 1:
            raise ValueError(f"adapter module {module}: {len(matches)} kernels match ({matches[:3]})")
        targets.append((module, nodes[matches[0]], a, b))
    return targets


def lora_delta(a, b, shape, scale, xp=np):
    """The kernel's update that qwix's LoRA adds in the forward pass, x.A.B.scale, as an array of the kernel's shape.
    A spans the kernel's contracted (leading) axes and the rank, B the rank and the remaining axes; qwix's dot_general
    path keeps each flattened, (prod(contracted), rank) and (rank, prod(remaining)), its einsum path keeps the axes. The
    update is A.B over the rank, reshaped in the kernel's (row-major) axis order."""
    rank = a.shape[-1]
    if b.shape[0] != rank:
        raise ValueError(f"lora_a {a.shape} and lora_b {b.shape} differ in rank")
    a2 = xp.reshape(xp.asarray(a, dtype=xp.float32), (-1, rank))
    b2 = xp.reshape(xp.asarray(b, dtype=xp.float32), (rank, -1))
    leading = [int(np.prod(shape[:k])) for k in range(1, len(shape))]
    if a2.shape[0] not in leading or a2.shape[0] * b2.shape[1] != int(np.prod(shape)):
        raise ValueError(f"lora_a {a.shape} and lora_b {b.shape} do not split a kernel of shape {tuple(shape)}")
    return xp.reshape(a2 @ b2, tuple(shape)) * scale


def merge_lora(tree, adapter, scale):
    """Adds every adapter module's update to its kernel (float32 sum, cast back to the kernel's dtype and sharding), in
    place in a nested dict of the engine's parameters. Returns what the merge changed."""
    import jax
    import jax.numpy as jnp

    delta_sq = weight_sq = changed = total = 0.0
    targets = lora_targets(adapter, tree)
    for module, node, a, b in targets:
        w = node["kernel"]
        try:
            delta = lora_delta(a, b, w.shape, scale, jnp)
        except ValueError as error:
            raise ValueError(f"{module}: {error}") from error
        merged = jax.device_put((w.astype(jnp.float32) + delta).astype(w.dtype), w.sharding)
        delta_sq += float(jnp.sum(delta ** 2))
        weight_sq += float(jnp.sum(w.astype(jnp.float32) ** 2))
        changed += float(jnp.sum(merged != w))
        total += w.size
        node["kernel"] = merged
    return {"modules": len(targets), "relative_update_norm": (delta_sq / weight_sq) ** 0.5,
            "elements_changed": changed / total, "kernel_dtype": str(targets[0][1]["kernel"].dtype)}


def load_lora(engine_params, lora_dir):
    """The engine's parameters with the adapter of sft_lora.py in lora_dir (adapter.npz, adapter.json) merged."""
    from flax import nnx

    meta = json.loads((Path(lora_dir) / "adapter.json").read_text())
    with np.load(Path(lora_dir) / "adapter.npz") as npz:
        adapter = {k: npz[k] for k in npz.files}
    tree = nnx.to_pure_dict(engine_params)
    stats = merge_lora(tree, adapter, meta["scale"])
    nnx.replace_by_pure_dict(engine_params, tree)
    print(f"LORA {json.dumps({'dir': str(lora_dir), 'rank': meta['rank'], 'alpha': meta['alpha'], **stats})}",
          flush=True)
    return engine_params


class EngineSlots:
    """The device side of decode_all: MaxEngine's prefill, insert and generate on one decode state."""

    def __init__(self, engine, params, prefill_lengths):
        self.engine, self.params, self.prefill_lengths = engine, params, prefill_lengths
        self.state = engine.init_decode_state()

    def prefill(self, tokens):
        import jax.numpy as jnp

        length = next((n for n in self.prefill_lengths if n >= len(tokens)), None)
        if length is None:
            raise RuntimeError(f"a prompt of {len(tokens)} tokens is longer than the largest prefill length "
                               f"{self.prefill_lengths[-1]}")
        padded = np.zeros(length, np.int32)
        padded[:len(tokens)] = tokens
        prefix, first = self.engine.prefill(params=self.params, padded_tokens=jnp.asarray(padded), true_length=len(tokens))
        return prefix, first.data[0, 0]

    def insert(self, prefix, slot):
        self.state = self.engine.insert(prefix, self.state, slot=slot)

    def generate(self):
        self.state, result = self.engine.generate(self.params, self.state)
        return result.data[:, 0]

    @staticmethod
    def fetch(arrays):
        import jax

        return np.asarray(jax.device_get(arrays))


def decode_all(prompts, slots, device, max_decode, sync_every, eos_ids=EOS_IDS):
    """Greedy completions of prompts over a fixed number of slots: [(tokens without the stop token, ended)].

    Continuous batching with the engine's own pieces: every free slot gets the next prompt (prefill, insert; the
    prefill's token is the first of the completion), then sync_every generate steps run for all slots at once and come
    back to the host in one transfer. A slot whose completion ends (a stop token, or max_decode tokens) is free for
    the next prompt; the steps it ran past its end are dropped. MaxText's OfflineEngine does the same with a thread
    per token and reached 344 tokens/s on a v4 host where these pieces allow several times that (8 Nov).
    Prompts go longest first, so the last steps of a shard are spent on short ones.
    """
    queue = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    queue.reverse()  # pop() from the end gives the longest first
    outputs = [None] * len(prompts)
    owner = [-1] * slots
    tokens = [[] for _ in range(slots)]

    def finished(s):
        return bool(tokens[s]) and (tokens[s][-1] in eos_ids or len(tokens[s]) >= max_decode)

    while queue or any(i >= 0 for i in owner):
        firsts = []
        for s in range(slots):
            if owner[s] < 0 and queue:
                i = queue.pop()
                prefix, first = device.prefill(prompts[i])
                device.insert(prefix, s)
                owner[s], tokens[s] = i, []
                firsts.append((s, first))
        steps = [device.generate() for _ in range(sync_every)]
        if firsts:
            for (s, _), token in zip(firsts, device.fetch([f for _, f in firsts])):
                tokens[s].append(int(token))
        for step in device.fetch(steps):
            for s in range(slots):
                if owner[s] >= 0 and not finished(s):
                    tokens[s].append(int(step[s]))
        for s in range(slots):
            if owner[s] >= 0 and finished(s):
                ended = tokens[s][-1] in eos_ids
                outputs[owner[s]] = (tokens[s][:-1] if ended else tokens[s], ended)
                owner[s] = -1
    return outputs


def shards_of(ids, texts, shard_size):
    """[(name, ids, texts)]: the same names for the same segments at every start, so a restart finds its shards."""
    return [(f"shard-{k // shard_size:05d}.parquet", ids[k:k + shard_size], texts[k:k + shard_size])
            for k in range(0, len(ids), shard_size)]


def translate_pending(shards, done, translate, store, max_shards=None):
    """translate(ids, texts) -> (rows, seconds), then store(name, rows, seconds), for every shard not in done; a shard
    is stored whole or not at all, so a crash in the middle of one costs only that one. Returns the names stored."""
    stored = []
    for name, shard_ids, shard_texts in shards:
        if name in done:
            continue
        if max_shards is not None and len(stored) == max_shards:
            break
        rows, seconds = translate(shard_ids, shard_texts)
        if rows["id"] != list(shard_ids):
            raise RuntimeError(f"{name}: the translations are not those of the shard's segments")
        store(name, rows, seconds)
        stored.append(name)
    return stored


# The bucket through the Python client, not gsutil: on the v6e image gsutil is a snap, and a snap refuses to start
# ("need to run as root or suid") under a process with no_new_privs, which this one has once libtpu is up (pilot of
# 9 Nov). google-cloud-storage comes with maxtext[tpu].
def gcs_client():
    from google.cloud import storage

    return storage.Client()


def split_gs(url):
    if not url.startswith("gs://"):
        raise ValueError(f"not a gs:// URL: {url}")
    bucket, _, name = url[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url}")
    return bucket, name


def done_shards(out, client=None):
    """The shard files directly under the bucket directory out (as gsutil ls out/ listed them); none if it is empty."""
    bucket, prefix = split_gs(out.rstrip("/"))
    blobs = (client or gcs_client()).list_blobs(bucket, prefix=f"{prefix}/" if prefix else "", delimiter="/")
    return {Path(blob.name).name for blob in blobs if blob.name.endswith(".parquet")}


def upload(path, url, client=None):
    from google.cloud.storage.retry import DEFAULT_RETRY

    bucket, name = split_gs(url)
    if not name or name.endswith("/"):
        raise ValueError(f"no object name in {url}")
    # Retried as gsutil cp retried: writing the same shard again under its name is harmless
    (client or gcs_client()).bucket(bucket).blob(name).upload_from_filename(str(path), retry=DEFAULT_RETRY)


def translate_shard(device, codec, tokenizer, ids, texts, shapes, sync_every, contexts=None, eos_ids=EOS_IDS,
                    labels=None):
    if contexts is None:
        prompts = [codec.encode(text) for text in texts]
    else:
        prompts = [codec.encode(text, context, label)
                   for text, context, label in zip(texts, contexts, labels or [None] * len(texts))]
    started = time.time()
    outputs = decode_all(prompts, shapes["slots"], device, shapes["max_decode"], sync_every, eos_ids)
    seconds = time.time() - started
    column = f"text_{codec.target}"
    rows = {"id": [], column: [], "ended": [], "tokens_in": [], "tokens_out": []}
    for i, prompt, (completion, ended) in zip(ids, prompts, outputs):
        rows["id"].append(i)
        rows[column].append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        rows["ended"].append(ended)
        rows["tokens_in"].append(len(prompt))
        rows["tokens_out"].append(len(completion) + int(ended))
    return rows, seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description="A translation model on one host: segments from English to Dutch")
    parser.add_argument("--segments", required=True,
                        help="segments-w<N>.parquet of the plan (local): id, text, tokens, and optionally context")
    parser.add_argument("--out", required=True, help="Bucket directory of this worker's shards of this length class")
    parser.add_argument("--ckpt", default=None,
                        help="The MaxText parameters, <checkpoint>/0/items (default: the model's in this VM's region)")
    parser.add_argument("--tokenizer", required=True, help="The model's tokenizer directory")
    parser.add_argument("--model", default="gemma3-27b", choices=sorted(MODELS), help="A model of translate_models.yml")
    parser.add_argument("--length-class", required=True)
    parser.add_argument("--slots", type=int, default=None, help="Override the class's number of decode slots")
    parser.add_argument("--shard-size", type=int, default=4096, help="Segments per shard (default: 4096)")
    parser.add_argument("--max-shards", type=int, default=None, help="Stop after this many new shards (pilot)")
    parser.add_argument("--sync-every", type=int, default=8, help="Generate steps between transfers (default: 8)")
    parser.add_argument("--source-lang", default="en", choices=sorted(LANGUAGES), help="Language of the segments")
    parser.add_argument("--target-lang", default="nl", choices=sorted(LANGUAGES), help="Language to translate into")
    parser.add_argument("--work-dir", default="/dev/shm/flan_nl",
                        help="Local directory for the shards before upload and MaxText's output (default: FLAN-nl's)")
    parser.add_argument("--lora", default=None,
                        help="Local directory of a LoRA adapter of experiments/en-nl-translator/sft_lora.py "
                             "(adapter.npz, adapter.json), merged into the checkpoint's kernels after loading")
    args = parser.parse_args(argv)
    if args.source_lang == args.target_lang:
        parser.error(f"--source-lang and --target-lang are both {args.source_lang}")
    model = MODELS[args.model]
    classes = model["length_classes"]
    if args.length_class not in classes:
        parser.error(f"--length-class {args.length_class} is not a class of {args.model}: {', '.join(classes)}")

    import jax
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    chips = model["chips"]
    if (jax.device_count(), jax.process_count()) != (chips, 1):
        raise SystemExit(f"needs one process with {chips} chips, has {jax.process_count()} process(es) with "
                         f"{jax.device_count()} devices: set the TPU_* variables of translate.sh")
    shapes = dict(classes[args.length_class], slots=args.slots or model["slots"][args.length_class])
    order = list(classes)
    k = order.index(args.length_class)
    low = classes[order[k - 1]]["max_source"] if k else 0
    table = pq.read_table(args.segments)
    ids, texts, tokens = (table.column(c).to_pylist() for c in ("id", "text", "tokens"))
    contexts = table.column("context").to_pylist() if "context" in table.column_names else None
    labels = table.column("label").to_pylist() if "label" in table.column_names else None
    if max(tokens, default=0) > classes[order[-1]]["max_source"]:
        raise SystemExit(f"{args.segments} has a segment of {max(tokens)} tokens, more than the plan allows")
    chosen = [k for k, n in enumerate(tokens) if low < n <= shapes["max_source"]]
    ids, texts = [ids[k] for k in chosen], [texts[k] for k in chosen]
    context_of = {ids[j]: contexts[k] for j, k in enumerate(chosen)} if contexts is not None else None
    label_of = {ids[j]: labels[k] for j, k in enumerate(chosen)} if labels is not None else None
    # Shards hold consecutive ids, so similar lengths; they are done in a fixed random order, so that the time of the
    # first few is a fair sample of the whole class (the projection of the run's length)
    shards = shards_of(ids, texts, args.shard_size)
    shards = random.Random(0).sample(shards, len(shards))
    done = done_shards(args.out)
    pending = sum(name not in done for name, _, _ in shards)
    print(f"{args.length_class}: {len(ids):,} segments in {len(shards)} shards, {len(shards) - pending} done before; "
          f"shapes {json.dumps(shapes)}", flush=True)
    if not pending:
        return

    if args.ckpt is None:
        args.ckpt = checkpoint_of(model, vm_region())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = ModelPrompts(tokenizer, model, args.source_lang, args.target_lang)
    probe = texts[:3] + texts[-3:]
    codec.check(probe, [context_of[i] for i in ids[:3] + ids[-3:]] if context_of else None,
                [label_of[i] for i in ids[:3] + ids[-3:]] if label_of else None)
    print(f"MAXTEXT {json.dumps({'model': args.model, 'ckpt': args.ckpt, **maxtext_source(), 'patches': patch_maxtext()})}", flush=True)
    started = time.time()
    engine, params = make_engine(args, shapes, chips)
    if args.lora:
        params = load_lora(params, args.lora)
    device = EngineSlots(engine, params, shapes["prefill_lengths"])
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    local = Path(args.work_dir) / "shards" / args.length_class
    local.mkdir(parents=True, exist_ok=True)
    client = gcs_client()
    totals = {"segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

    def store(name, rows, seconds):
        path = local / name
        pq.write_table(pa.table(rows), path, compression="zstd")
        upload(path, f"{args.out}/{name}", client)
        path.unlink()
        n_in, n_out = sum(rows["tokens_in"]), sum(rows["tokens_out"])
        for key, value in (("segments", len(rows["id"])), ("tokens_in", n_in), ("tokens_out", n_out), ("seconds", seconds)):
            totals[key] += value
        print(f"SHARD {json.dumps({'class': args.length_class, 'shard': name, 'segments': len(rows['id']), 'tokens_in': n_in, 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / seconds), 'all_per_s': round((n_in + n_out) / seconds), 'not_ended': rows['ended'].count(False)})}", flush=True)

    def translate(shard_ids, shard_texts):
        shard_contexts = [context_of[i] for i in shard_ids] if context_of else None
        shard_labels = [label_of[i] for i in shard_ids] if label_of else None
        return translate_shard(device, codec, tokenizer, shard_ids, shard_texts, shapes, args.sync_every,
                               shard_contexts, model["eos_ids"], shard_labels)

    translate_pending(shards, done, translate, store, args.max_shards)
    print(f"DONE {json.dumps({'class': args.length_class, **totals, 'out_per_s': round(totals['tokens_out'] / max(totals['seconds'], 1e-9))})}", flush=True)


if __name__ == "__main__":
    # A thread that is no daemon (MaxText's OfflineEngine had one, fsspec's IO loop is another) would keep a failed
    # process alive, holding the TPU and never reaching translate.sh's EXIT line (pilot of 8 Nov). So the process ends
    # here, with the error's traceback and code, whatever the threads do.
    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
