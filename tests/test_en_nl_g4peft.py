"""G4-8-s140 for Hugging Face (experiments/en-nl-translator/g4peft.py, g4hfcheck.py; protocol-g4.md §26): the MaxText
LoRA -> PEFT conversion with small fake tensors against MaxText's own conventions (translate.lora_delta: qwix's update
x.A.B.scale as a kernel of MaxText's shape; MaxText 0.2.5's Gemma 4 conversion hook: kernel = W.T.reshape(shape)), the
module set of Gemma 4 31B-it (410 modules, 244,858,880 parameters), the stops, the merge, the selection of the check's
sources and the card helpers. With torch, transformers and peft (a worker's venv, or any CPU venv) a tiny random
Gemma4ForConditionalGeneration: the converted adapter loads into PEFT with every factor, and base + PEFT adapter gives
the logits of the base model with MaxText's merged kernels; the merged directory gives the same."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "en-nl-translator"))
sys.path.insert(0, str(ROOT / "experiments" / "flan-nl"))

import g4hfcheck as hc  # noqa: E402
import g4peft as gp  # noqa: E402
import translate  # noqa: E402

# google/gemma-4-31B-it @ 842da37, config.json's text_config (the fields the conversion reads)
CONFIG_31B = {"text_config": {
    "hidden_size": 5376, "intermediate_size": 21504, "num_attention_heads": 32, "num_key_value_heads": 16,
    "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 4, "attention_k_eq_v": True,
    "num_hidden_layers": 60,
    "layer_types": ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(60)]}}

TINY_TEXT = {"vocab_size": 64, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 6,
             "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8, "global_head_dim": 12,
             "num_global_key_value_heads": 1, "attention_k_eq_v": True, "hidden_size_per_layer_input": 0,
             "sliding_window": 8, "final_logit_softcapping": 30.0, "max_position_embeddings": 128,
             "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1}
TINY_CONFIG = {"text_config": dict(TINY_TEXT, layer_types=["sliding_attention"] * 5 + ["full_attention"])}
RANK, ALPHA = 4, 2.0
META = {"rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK}


def kernel_shape(name, out_in, t):
    """MaxText's kernel shape of an HF weight (out, in): attention split per head (query/key/value: (embed, heads,
    head_dim); out: (heads, head_dim, embed)), MLP (in, out)."""
    out, inp = out_in
    if name.endswith(("q_proj", "k_proj", "v_proj")):
        hd = head_dim_of(name, t)
        return (inp, out // hd, hd)
    if name.endswith("o_proj"):
        hd = head_dim_of(name, t)
        return (inp // hd, hd, out)
    return (inp, out)


def head_dim_of(name, t):
    layer = int(name.split(".layers.")[1].split(".")[0])
    return gp.layer_dims(t, layer, t["layer_types"][layer] == "full_attention")[0]


MT_TYPE = {v: k for k, v in gp.HF_NAMES.items()}


def mt_name(hf_name):
    layer = int(hf_name.split(".layers.")[1].split(".")[0])
    return f"decoder/layers_{layer}/{MT_TYPE[hf_name.split(f'.layers.{layer}.')[1]]}"


def fake_adapter(config, rng, flat=False):
    """adapter.npz's arrays for a config, in qwix's layouts: lora_a (contracted axes..., rank), lora_b (rank,
    remaining axes...), or both flattened (qwix's dot_general path). Returns (arrays, {hf name: kernel shape})."""
    t = gp.text_config(config)
    arrays, kernels = {}, {}
    for name, out_in in gp.expected_shapes(config).items():
        ks = kernel_shape(name, out_in, t)
        n_in = 1 if not name.endswith("o_proj") else 2
        a_shape, b_shape = (*ks[:n_in], RANK), (RANK, *ks[n_in:])
        if flat:
            a_shape, b_shape = (int(np.prod(ks[:n_in])), RANK), (RANK, int(np.prod(ks[n_in:])))
        key = mt_name(name)
        arrays[f"{key}/kernel_lora_a"] = rng.standard_normal(a_shape).astype(np.float32)
        arrays[f"{key}/kernel_lora_b"] = rng.standard_normal(b_shape).astype(np.float32)
        kernels[name] = ks
    return arrays, kernels


def hf_update(kernel_delta, out_in):
    """MaxText 0.2.5's Gemma 4 hook back to HF (saving_to_hf): kernel.reshape(in, out).T"""
    return np.asarray(kernel_delta).reshape(out_in[1], out_in[0]).T


# ------------------------------------------------------------------------------------------------ module set, names

def test_the_31b_has_410_modules_and_the_trained_parameter_count():
    shapes = gp.expected_shapes(CONFIG_31B)
    assert len(shapes) == 410
    assert sum(32 * (o + i) for o, i in shapes.values()) == 244_858_880
    full = [i for i in range(60) if (i + 1) % 6 == 0]
    assert len(full) == 10
    for i in full:
        assert f"model.language_model.layers.{i}.self_attn.v_proj" not in shapes
        assert shapes[f"model.language_model.layers.{i}.self_attn.q_proj"] == (32 * 512, 5376)
        assert shapes[f"model.language_model.layers.{i}.self_attn.k_proj"] == (4 * 512, 5376)
        assert shapes[f"model.language_model.layers.{i}.self_attn.o_proj"] == (5376, 32 * 512)
    assert shapes["model.language_model.layers.0.self_attn.v_proj"] == (16 * 256, 5376)
    assert shapes["model.language_model.layers.0.mlp.down_proj"] == (5376, 21504)


def test_per_layer_config_gives_the_same_shapes():
    t = dict(TINY_TEXT, layer_types=TINY_CONFIG["text_config"]["layer_types"])
    without = {k: v for k, v in t.items() if k not in ("global_head_dim", "num_global_key_value_heads")}
    with_plc = dict(without, per_layer_config={"5": {"head_dim": 12, "num_key_value_heads": 1}})
    assert gp.expected_shapes({"text_config": with_plc}) == gp.expected_shapes(TINY_CONFIG)


def test_hf_module_names():
    assert gp.hf_module("decoder/layers_12/self_attention/query") == "model.language_model.layers.12.self_attn.q_proj"
    assert gp.hf_module("decoder/layers_0/self_attention/out") == "model.language_model.layers.0.self_attn.o_proj"
    assert gp.hf_module("decoder/layers_59/mlp/wi_0") == "model.language_model.layers.59.mlp.gate_proj"
    assert gp.hf_module("decoder/layers_3/mlp/wi_1") == "model.language_model.layers.3.mlp.up_proj"
    assert gp.hf_module("decoder/layers_3/mlp/wo") == "model.language_model.layers.3.mlp.down_proj"
    with pytest.raises(ValueError):
        gp.hf_module("decoder/layers_3/self_attention/query_norm")


def test_target_modules_regex_reaches_the_text_decoder_only():
    import re

    pattern = re.compile(gp.TARGET_MODULES)
    for name in gp.expected_shapes(CONFIG_31B):
        assert pattern.fullmatch(name)
    for name in ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                 "model.vision_tower.encoder.layers.0.self_attn.q_proj", "model.language_model.layers.0.self_attn.q_norm",
                 "model.language_model.layers.0.self_attn.q_proj.lora_A", "lm_head", "model.embed_vision.embedding_projection"):
        assert not pattern.fullmatch(name)


# -------------------------------------------------------------------------------------- the conversion's arithmetic

@pytest.mark.parametrize("flat", [False, True])
def test_peft_update_is_maxtexts_update_transposed(flat):
    rng = np.random.default_rng(0)
    arrays, kernels = fake_adapter(TINY_CONFIG, rng, flat=flat)
    tensors, cfg, report = gp.convert_arrays(arrays, META, TINY_CONFIG)
    scaling = cfg["lora_alpha"] / cfg["r"]
    assert scaling == META["scale"] and report["modules"] == 6 * 7 - 1 and report["full_without_v_proj"] == 1
    for name, ks in kernels.items():
        key = mt_name(name)
        mt = translate.lora_delta(arrays[f"{key}/kernel_lora_a"], arrays[f"{key}/kernel_lora_b"], ks, META["scale"])
        a = tensors[f"base_model.model.{name}.lora_A.weight"]
        b = tensors[f"base_model.model.{name}.lora_B.weight"]
        assert a.dtype == np.float32 and a.shape[0] == RANK and b.shape[1] == RANK
        np.testing.assert_allclose(b @ a * scaling, hf_update(mt, gp.expected_shapes(TINY_CONFIG)[name]),
                                   rtol=1e-5, atol=1e-5)


def test_forward_equivalence_on_one_attention_projection():
    """x through MaxText's merged kernel (einsum over the kernel's leading axis) equals x through the HF weight with
    PEFT's update, per head."""
    rng = np.random.default_rng(1)
    e, h, d = 16, 2, 8
    kernel = rng.standard_normal((e, h, d)).astype(np.float32)
    a, b = rng.standard_normal((e, RANK)).astype(np.float32), rng.standard_normal((RANK, h, d)).astype(np.float32)
    x = rng.standard_normal((3, e)).astype(np.float32)
    mt = np.einsum("te,ehd->thd", x, kernel + translate.lora_delta(a, b, kernel.shape, 0.5))
    w = kernel.reshape(e, h * d).T  # the HF weight the hook made the kernel from
    lora_a, lora_b = gp.peft_factors(a, b, RANK)
    hf = (x @ w.T + (x @ lora_a.T) @ lora_b.T * 0.5).reshape(3, h, d)
    np.testing.assert_allclose(hf, mt, rtol=1e-5, atol=1e-4)


# ------------------------------------------------------------------------------------------------------- the stops

def test_stops():
    rng = np.random.default_rng(2)
    arrays, _ = fake_adapter(TINY_CONFIG, rng)
    with pytest.raises(SystemExit, match="scale"):
        gp.convert_arrays(arrays, dict(META, scale=0.3), TINY_CONFIG)
    drop = {k: v for k, v in arrays.items() if not k.startswith("decoder/layers_2/mlp/wo/")}
    with pytest.raises(SystemExit, match="1 missing"):
        gp.convert_arrays(drop, META, TINY_CONFIG)
    extra = dict(arrays, **{"decoder/layers_5/self_attention/value/kernel_lora_a": np.zeros((16, RANK), np.float32),
                            "decoder/layers_5/self_attention/value/kernel_lora_b": np.zeros((RANK, 1, 12), np.float32)})
    with pytest.raises(SystemExit, match="1 not in the model"):
        gp.convert_arrays(extra, META, TINY_CONFIG)
    half = {k: v for k, v in arrays.items() if k != "decoder/layers_0/mlp/wi_0/kernel_lora_b"}
    with pytest.raises(SystemExit, match="without both"):
        gp.convert_arrays(half, META, TINY_CONFIG)
    odd = dict(arrays, **{"decoder/layers_0/self_attention/query_norm/kernel_lora_a": np.zeros((1, RANK))})
    with pytest.raises(SystemExit, match="none of"):
        gp.convert_arrays(odd, META, TINY_CONFIG)
    wrong = dict(arrays, **{"decoder/layers_0/mlp/wi_0/kernel_lora_b": np.zeros((RANK, 25), np.float32)})
    with pytest.raises(SystemExit, match="update"):
        gp.convert_arrays(wrong, META, TINY_CONFIG)
    rank = dict(arrays, **{"decoder/layers_0/mlp/wi_0/kernel_lora_a": np.zeros((16, RANK + 1), np.float32)})
    with pytest.raises(ValueError, match="rank"):
        gp.convert_arrays(rank, META, TINY_CONFIG)
    shapes = {f"{n}.weight": s for n, s in gp.expected_shapes(TINY_CONFIG).items()}
    shapes["model.language_model.layers.1.mlp.up_proj.weight"] = (24, 17)
    with pytest.raises(SystemExit, match="base weight"):
        gp.convert_arrays(arrays, META, TINY_CONFIG, shapes)


def test_prefixed_keys_and_the_config(tmp_path):
    rng = np.random.default_rng(3)
    arrays, _ = fake_adapter(TINY_CONFIG, rng)
    prefixed = {f"base/{k}": v for k, v in arrays.items()}
    a, cfg, _ = gp.convert_arrays(arrays, META, TINY_CONFIG)
    b, _, _ = gp.convert_arrays(prefixed, META, TINY_CONFIG)
    assert a.keys() == b.keys()
    assert cfg["r"] == RANK and cfg["lora_alpha"] == ALPHA and cfg["target_modules"] == gp.TARGET_MODULES
    assert cfg["base_model_name_or_path"] == "google/gemma-4-31B-it" and cfg["revision"].startswith("842da37")
    assert cfg["lora_dropout"] == 0.0 and cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM"


def test_convert_command_writes_peft_files(tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")
    rng = np.random.default_rng(4)
    arrays, _ = fake_adapter(TINY_CONFIG, rng)
    np.savez(tmp_path / "adapter.npz", **arrays)
    (tmp_path / "adapter.json").write_text(json.dumps(META))
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    gp.main(["convert", "--npz", str(tmp_path / "adapter.npz"), "--meta", str(tmp_path / "adapter.json"),
             "--config", str(tmp_path / "config.json"), "--out", str(tmp_path / "out")])
    loaded = safetensors.load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    assert len(loaded) == 2 * 41
    cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert cfg["r"] == RANK
    report = json.loads((tmp_path / "out" / "convert.json").read_text())
    assert report["modules"] == 41 and len(report["npz_sha256"]) == 64


# ------------------------------------------------------------------------------------------ select and card helpers

def fake_pairs(n=40, sets=("flores", "ntrex", "wmt24pp", "flan", "europarl")):
    rows = {}
    for s in sets:
        for k in range(n):
            i = f"{s}-{k:03d}"
            rows[i] = {"id": i, "set": s, "source": f"src {i}", "hypothesis": f"hyp {i}", "ended": True,
                       "empty": False, "decode_loop": False, "english_left": False, "tokens_in": 100,
                       "tokens_out": 50}
    return rows


def test_select_is_seeded_clean_and_without_novels_or_flan():
    arm, base = fake_pairs(), fake_pairs()
    arm["flores-001"]["ended"] = False
    base["ntrex-002"]["decode_loop"] = True
    arm["wmt24pp-003"]["tokens_out"] = 950
    base["flores-004"]["source"] = "other"
    check, smoke = hc.select(arm, base, 10, 8)
    again = hc.select(arm, base, 10, 8)
    assert (check, smoke) == again
    assert set(check) == {"flores", "ntrex", "wmt24pp"} and all(len(v) == 10 for v in check.values())
    chosen = [i for v in check.values() for i in v]
    assert not {"flores-001", "ntrex-002", "wmt24pp-003", "flores-004"} & set(chosen + smoke)
    assert len(smoke) == 8 and all(i.startswith("flores-") for i in smoke) and not set(smoke) & set(check["flores"])
    with pytest.raises(SystemExit, match="need 50"):
        hc.select(arm, base, 50, 8)


def test_summary_and_require():
    rows = [{"tokens": 10, "equal": 10}, {"tokens": 10, "equal": 9}]
    s = hc.summary(rows)
    assert s["agreement"] == 0.95 and s["sources_all_equal"] == 1 and s["min_source_agreement"] == 0.9
    assert hc.parse_require(["adapter:arm>=0.99", "model:arm>=0.99"]) == {"adapter:arm": 0.99, "model:arm": 0.99}
    with pytest.raises(SystemExit):
        hc.parse_require(["adapter:g4>=0.99"])


def test_card_prompt_is_natural_v1(tmp_path):
    card = tmp_path / "card.md"
    prompt = hc.natural_v1_template()
    card.write_text("x\n```python\nPROMPT = (\n" + "".join(f"    {json.dumps(p)}\n" for p in prompt.splitlines(True))
                    + ")\n\ndef translate(text):\n    return PROMPT.format(text=text.strip())\n```\n\n```text\n"
                    + prompt + "\n```\n")
    assert hc.card_prompts(card) == {"python": True, "text": True}
    card.write_text(card.read_text().replace("idiomatic", "idiomatik"))
    assert hc.card_prompts(card) == {"python": False, "text": False}


def test_natural_v1_template_is_translates_prompt():
    assert hc.natural_v1_template().format(text="X") == translate.user_text(translate.PROMPTS["natural-v1"], "X")


def test_scores_need_sacrebleu_and_langid():
    pytest.importorskip("sacrebleu")
    pytest.importorskip("langid")
    rows = [("a", "Het weer was zo mooi dat we besloten buiten te lunchen."), ("b", "")]
    ref = {"a": "Het weer was zo mooi dat we besloten buiten te lunchen.", "b": "Iets anders."}
    s = hc.scores(rows, ref)
    assert s["n"] == 2 and s["nonempty"] == 1 and s["exact"] == 1 and s["rows"][0]["chrf"] == 100.0


# --------------------------------------------------------------------------- a tiny Gemma 4 in transformers + PEFT

def tiny_model(torch):
    transformers = pytest.importorskip("transformers")
    vision = {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
              "num_key_value_heads": 2, "head_dim": 4, "patch_size": 4, "position_embedding_size": 64}
    cfg = transformers.Gemma4Config(text_config=dict(TINY_TEXT), vision_config=vision, image_token_id=60,
                                    boi_token_id=61, eoi_token_id=62, video_token_id=63, boa_token_id=59,
                                    eoa_token_index=58, audio_token_id=57)
    torch.manual_seed(0)
    return transformers.Gemma4ForConditionalGeneration(cfg).eval()


def maxtext_merged(model, arrays, kernels, torch):
    """A copy of the model's state with every adapter module's weight W + hook^-1(lora_delta) (MaxText's merge of the
    same adapter, mapped back to HF), in float32."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    shapes = gp.expected_shapes(TINY_CONFIG)
    for name, ks in kernels.items():
        key = mt_name(name)
        delta = translate.lora_delta(arrays[f"{key}/kernel_lora_a"], arrays[f"{key}/kernel_lora_b"], ks, META["scale"])
        state[name + ".weight"] += torch.as_tensor(hf_update(delta, shapes[name]), dtype=torch.float32)
    return state


def test_tiny_gemma4_peft_and_merge(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model = tiny_model(torch)
    rng = np.random.default_rng(5)
    arrays, kernels = fake_adapter(TINY_CONFIG, rng)
    for k in arrays:  # updates of the size of a trained adapter's, so that float32 sums stay comparable
        arrays[k] *= 0.05
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    model.save_pretrained(base_dir)
    np.savez(tmp_path / "adapter.npz", **arrays)
    (tmp_path / "adapter.json").write_text(json.dumps(META))
    gp.main(["convert", "--npz", str(tmp_path / "adapter.npz"), "--meta", str(tmp_path / "adapter.json"),
             "--config", str(base_dir / "config.json"), "--base-dir", str(base_dir), "--out", str(adapter_dir)])
    ids = torch.tensor([[2, 5, 9, 17, 33, 40, 41, 7, 3, 11]])
    want = tiny_model(torch)
    want.load_state_dict(maxtext_merged(model, arrays, kernels, torch))
    with torch.no_grad():
        ref = want(input_ids=ids).logits
        base_logits = model(input_ids=ids).logits
        pm = peft.PeftModel.from_pretrained(tiny_model(torch), str(adapter_dir)).eval()
        assert hc.peft_load_check(pm, adapter_dir, torch) == {"lora_modules": 41, "equal": True}
        got = pm(input_ids=ids).logits
        with pm.disable_adapter():
            off = pm(input_ids=ids).logits
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(off, base_logits)
    assert (got - base_logits).abs().max() > 1e-3  # the adapter changes the model
    names = [n for n, m in pm.named_modules() if hasattr(m, "lora_A")]
    assert names and all(".language_model.layers." in n for n in names)

    # the merge: bf16 weights, W + scaling * B A in float32, then bf16
    model.to(torch.bfloat16).save_pretrained(tmp_path / "base16")
    gp.main(["merge", "--base-dir", str(tmp_path / "base16"), "--adapter-dir", str(adapter_dir),
             "--out", str(tmp_path / "merged")])
    stats = json.loads((tmp_path / "merged" / "merge.json").read_text())
    assert stats["modules"] == 41 and 0 < stats["elements_changed"] <= 1
    merged = pytest.importorskip("transformers").AutoModelForCausalLM.from_pretrained(tmp_path / "merged",
                                                                                      dtype=torch.float32)
    state16 = {k: v.to(torch.float32) for k, v in model.state_dict().items()}
    want32 = maxtext_merged(model, arrays, kernels, torch)
    for name in kernels:
        w = merged.state_dict()[name + ".weight"]
        exact = (state16[name + ".weight"] + (want32[name + ".weight"] - model.state_dict()[name + ".weight"]
                                               .to(torch.float32))).to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(w, exact, rtol=2 ** -7, atol=1e-6)  # at most one bf16 step apart
    assert (tmp_path / "merged" / "config.json").exists()
    with pytest.raises(SystemExit, match="exists"):
        gp.main(["merge", "--base-dir", str(tmp_path / "base16"), "--adapter-dir", str(adapter_dir),
                 "--out", str(tmp_path / "merged")])
    with pytest.raises(SystemExit, match="differs from MaxText"):
        gp.main(["merge", "--base-dir", str(tmp_path / "base16"), "--adapter-dir", str(adapter_dir),
                 "--out", str(tmp_path / "merged2"), "--maxtext-norm", str(stats["relative_update_norm"] * 1.05)])
