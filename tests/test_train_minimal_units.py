"""Optimizer and metric code of scripts/train_minimal.py, on the parameter tree of the real 22-layer model."""
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mmbjax.configuration_mmbbert import MMBertConfig  # noqa: E402
from mmbjax.heads.masked_lm import ModernBertForMaskedLM  # noqa: E402

LAYERS = 22


@pytest.fixture(scope="module")
def tm():
    # Other test modules already started JAX, after which its module-level initialize() can't run
    with mock.patch("jax.distributed.initialize", side_effect=ValueError):
        import train_minimal

    return train_minimal


@pytest.fixture(scope="module")
def model_and_params():
    # The architecture of the 230M runs; only the widths are small
    config = MMBertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=LAYERS, num_attention_heads=2, intermediate_size=32,
        global_attn_every_n_layers=3, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        compute_dtype="float32", param_dtype="float32",
    )
    model = ModernBertForMaskedLM(config, dtype=jnp.float32, param_dtype=jnp.float32)
    ones = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids=ones, attention_mask=ones, deterministic=True)["params"]
    return model, params


def _updates(tm, model, params, factor, weight_decay=0.0):
    optimizer, _ = tm.create_train_state(
        model, learning_rate=1e-3, num_train_steps=100, warmup_steps=1, weight_decay=weight_decay,
        lr_schedule_type="constant", layer_norm_lr_factor=factor,
    )
    opt_state = optimizer.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(3):
        updates, opt_state = optimizer.update(grads, opt_state, params)
    return traverse_util.flatten_dict(updates)


def _is_norm(path):
    return any(part == "norm" or part.endswith("_norm") for part in path)


def test_layer_norm_lr_factor_applies_to_norms_only(tm, model_and_params):
    model, params = model_and_params
    plain = _updates(tm, model, params, factor=1.0)
    scaled = _updates(tm, model, params, factor=0.1)
    norms = [path for path in plain if _is_norm(path)]
    # 22 mlp_norm, 21 attention_norm (layer 0 has none), embeddings, final and prediction head
    assert len(norms) == 22 + 21 + 3
    for path in plain:
        ratio = float(jnp.mean(scaled[path] / plain[path]))
        assert ratio == pytest.approx(0.1 if _is_norm(path) else 1.0, rel=1e-3), path


def test_weight_decay_still_reaches_the_kernels_with_a_factor(tm, model_and_params):
    model, params = model_and_params
    without = _updates(tm, model, params, factor=0.1, weight_decay=0.0)
    with_decay = _updates(tm, model, params, factor=0.1, weight_decay=0.1)
    kernel = ("model", "encoder", "layers_3", "mlp", "wi", "kernel")
    norm = ("model", "encoder", "layers_3", "mlp_norm", "scale")
    assert not np.allclose(with_decay[kernel], without[kernel])
    assert np.allclose(with_decay[norm], without[norm])


def test_per_layer_gradient_norms_cover_all_22_layers_without_mixing(tm, model_and_params):
    _, params = model_and_params
    flat = traverse_util.flatten_dict(params)
    # A gradient of ones in layer 1 only: "layer_1" as a substring would also catch 10 to 19 and 21
    grads = {path: (jnp.ones_like(v) if "layers_1" in path else jnp.zeros_like(v)) for path, v in flat.items()}
    norms = tm.compute_per_layer_gradient_norms(traverse_util.unflatten_dict(grads))

    assert {f"encoder_{i}_attention" for i in range(LAYERS)} <= set(norms)
    assert {f"encoder_{i}_ffn" for i in range(LAYERS)} <= set(norms)
    assert {"embeddings", "lm_head"} <= set(norms)
    assert "other" not in norms
    for name, value in norms.items():
        if name.startswith("encoder_1_"):
            assert float(value) > 0, name
        else:
            assert float(value) == 0, name

    # Nothing counted twice, nothing left out
    ones = tm.compute_per_layer_gradient_norms(jax.tree_util.tree_map(jnp.ones_like, params))
    total = sum(float(v) ** 2 for v in ones.values())
    assert total == pytest.approx(sum(v.size for v in flat.values()), rel=1e-5)


def test_per_layer_norms_are_requested_on_a_step_that_logs(tm):
    every = 100
    for accumulation in (1, 3):
        requested = [
            optimizer_step
            for micro_step in range(1000 * accumulation)
            for optimizer_step in [micro_step // accumulation]
            if tm.wants_per_layer_norms(optimizer_step, micro_step, accumulation, every)
        ]
        assert requested == [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        # The loop logs when optimizer_step + 1, after this micro-step, is a multiple of log_every
        assert all((step + 1) % 10 == 0 for step in requested)


def test_tracker_choice(tm):
    assert tm.load_tracker("none") is None
    fake = types.ModuleType("trackio")
    fake.__spec__ = importlib.machinery.ModuleSpec("trackio", None)
    sys.modules.setdefault("trackio", fake)
    assert tm.load_tracker("trackio") is sys.modules["trackio"]


def test_duty_cycle_is_read_from_the_table_of_tpu_info(tm):
    # Literal output on the second host of a v4-32, during a run: chips are numbered 4 to 7
    output = """
TPU Runtime Utilization

| Chip | HBM Usage (GiB)       | Duty cycle |
|------|-----------------------|------------|
| 4    | 18.10 GiB / 30.75 GiB | 95.72%     |
| 5    | 18.10 GiB / 30.75 GiB | 95.73%     |

TensorCore Utilization

| Core ID | TensorCore Utilization |
|---------|------------------------|
| 0       | N/A                    |
"""
    assert {int(chip): float(duty) for chip, duty in tm.DutyCycleSampler.ROW.findall(output)} == {4: 95.72, 5: 95.73}


def test_resume_with_another_split_setup_is_refused(tm, tmp_path):
    args = types.SimpleNamespace(
        dataset="d", dataset_config="c", datasets_config=None, max_length=1024, tokenizer="t", shuffle_buffer_size=10,
    )
    identity = tm.data_state_identity(args, 32, ["a.parquet"])
    assert identity["long_documents"] == "split_into_pieces"
    assert identity["piece_shuffle_window"] == tm.PIECE_SHUFFLE_WINDOW

    path = tm.epath.Path(tmp_path) / "state.json"
    loader = mock.Mock(get_state=mock.Mock(return_value={"epoch": 0}))
    tm.save_data_state(path, loader, identity, 25000)

    tm.restore_data_state(path, loader, identity, allow_restart=False)
    loader.set_state.assert_called_once_with({"epoch": 0})

    for key, value in [("piece_shuffle_window", identity["piece_shuffle_window"] + 1), ("long_documents", "truncated")]:
        with pytest.raises(RuntimeError, match="another data setup"):
            tm.restore_data_state(path, loader, {**identity, key: value}, allow_restart=False)
    # A data state from before the split: refused as well, also with --allow-data-restart
    old = {k: v for k, v in identity.items() if k not in ("long_documents", "piece_shuffle_window")}
    tm.save_data_state(path, loader, old, 25000)
    with pytest.raises(RuntimeError, match="another data setup"):
        tm.restore_data_state(path, loader, identity, allow_restart=True)


def test_host_finds_its_data_state_after_a_reboot_changed_the_process_indices(tm, tmp_path):
    args = types.SimpleNamespace(
        dataset="d", dataset_config="c", datasets_config=None, max_length=1024, tokenizer="t", shuffle_buffer_size=10,
    )
    identity = {host: tm.data_state_identity(args, 32, [f"{host}.parquet"]) for host in ("a", "b")}
    directory = tm.epath.Path(tmp_path)
    for index, host in enumerate(("a", "b")):
        loader = mock.Mock(get_state=mock.Mock(return_value={"epoch": 0, "of": host}))
        tm.save_data_state(directory / f"process_{index}.json", loader, identity[host], 25000)

    # Host b is process 0 now
    loader = mock.Mock()
    tm.restore_data_state(directory / "process_0.json", loader, identity["b"], allow_restart=False)
    loader.set_state.assert_called_once_with({"epoch": 0, "of": "b"})

    # Files no state was saved for: refused
    stranger = tm.data_state_identity(args, 32, ["c.parquet"])
    with pytest.raises(RuntimeError, match="another data setup"):
        tm.restore_data_state(directory / "process_0.json", loader, stranger, allow_restart=False)


def _masking_args(tm, *flags):
    with mock.patch.object(sys, "argv", ["train_minimal.py", *flags]):
        return tm.parse_args()


def test_masking_default_keeps_the_route_and_the_data_state_of_the_long_run(tm, tmp_path):
    args = _masking_args(tm, "--use-packing", "--dataset", "uonlp/CulturaX", "--dataset-config", "nl", "--max-length", "1024",
                         "--tokenizer", "yhavinga/dutch-llama-tokenizer")
    assert args.masking == "uniform15"
    identity = tm.data_state_identity(args, 32, ["/home/yeb/culturax_nl/nl/nl_part_00038.parquet"], args.masking)
    # The identity in gs://.../culturax-nl-230m-chunk1024/ckpt_data_state/355000/process_0.json, data files cut short
    assert identity == {
        "data_files": ["/home/yeb/culturax_nl/nl/nl_part_00038.parquet"],
        "dataset": "uonlp/CulturaX",
        "dataset_config": "nl",
        "datasets_config": None,
        "process_count": 1,
        "rows_per_host": 32,
        "max_length": 1024,
        "long_documents": "split_into_pieces",
        "piece_shuffle_window": 5000,
        "special_tokens_in_text": "split",
        "tokenizer": "yhavinga/dutch-llama-tokenizer",
        "shuffle_buffer_size": 10000,
    }
    path = tm.epath.Path(tmp_path) / "state.json"
    # A state saved before the flag existed: the same bytes as a state saved now
    path.write_text(tm.json.dumps({"step": 355000, "identity": identity, "state": {"epoch": 0}}))
    loader = mock.Mock()
    tm.restore_data_state(path, loader, identity, allow_restart=False)
    loader.set_state.assert_called_once_with({"epoch": 0})


@pytest.mark.parametrize("saved,resumed", [
    ("uniform15", "uniform30"), ("uniform15", "mix"), ("uniform30", "uniform15"), ("uniform30", "mix"), ("mix", "uniform15"), ("mix", "uniform30"),
    ("uniform15", "uniform10"), ("uniform10", "uniform5"), ("mix", "uniform10"),
])
def test_resume_across_masking_modes_is_refused(tm, tmp_path, saved, resumed):
    args = types.SimpleNamespace(
        dataset="d", dataset_config="c", datasets_config=None, max_length=1024, tokenizer="t", shuffle_buffer_size=10,
    )
    path = tm.epath.Path(tmp_path) / "state.json"
    loader = mock.Mock(get_state=mock.Mock(return_value={"epoch": 0}))
    tm.save_data_state(path, loader, tm.data_state_identity(args, 32, None, saved), 5000)
    tm.restore_data_state(path, loader, tm.data_state_identity(args, 32, None, saved), allow_restart=False)
    # Also with data files, where restore_data_state looks for another host's state, and with --allow-data-restart
    for files, allow in [(None, False), (["a.parquet"], False), (None, True)]:
        tm.save_data_state(path, loader, tm.data_state_identity(args, 32, files, saved), 5000)
        with pytest.raises(RuntimeError, match="another data setup"):
            tm.restore_data_state(path, loader, tm.data_state_identity(args, 32, files, resumed), allow_restart=allow)


def test_masking_flag_refuses_routes_that_would_ignore_it(tm):
    for flags in (["--masking", "mix"], ["--masking", "uniform30"],
                  ["--masking", "mix", "--use-packing", "--masking-mode", "instruction_response"],
                  ["--masking", "mix", "--use-packing", "--dataset-format", "inputs_targets"],
                  ["--masking", "uniform30", "--use-packing", "--datasets-config", "c.json"]):
        with pytest.raises(SystemExit):
            _masking_args(tm, *flags)
    assert _masking_args(tm, "--masking", "mix", "--use-packing").masking == "mix"
    # The older flag that starts with the same word still works, in full
    assert _masking_args(tm).masking_mode == "standard"


def _eval_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    vocab = {token: i for i, token in enumerate(specials + [f"w{i}" for i in range(200)])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", mask_token="<mask>")


def test_eval_batches_are_the_same_15_percent_masks_in_every_mode(tm, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(0)
    path = tmp_path / "held_out.parquet"
    pq.write_table(pa.table({"text": [" ".join(f"w{w}" for w in rng.randint(0, 200, size=rng.randint(3, 60))) for _ in range(600)]}), path)
    tokenizer = _eval_tokenizer()

    batches = {}
    for seed, masking in enumerate(("uniform15", "uniform30", "mix")):
        # What main() does before training; another RNG state here mustn't matter either
        np.random.seed(seed)
        train_collator = tm.create_masking_collator(masking, tokenizer)
        eval_collator = tm.create_eval_collator(masking, train_collator, tokenizer)
        assert type(eval_collator) is tm.PackedMLMDataCollator and eval_collator.mlm_probability == 0.15
        batches[masking] = tm.load_eval_batches(str(path), tokenizer, eval_collator, 64, 4, 5)

    for masking in ("uniform30", "mix"):
        for ours, theirs in zip(batches[masking], batches["uniform15"], strict=True):
            assert ours.keys() == theirs.keys()
            for key in ours:
                assert np.array_equal(ours[key], theirs[key]), (masking, key)

    labels = np.stack([batch["labels"] for batch in batches["mix"]])
    real = np.stack([batch["attention_mask"] for batch in batches["mix"]]).sum()
    assert 0.08 < np.sum(labels != -100) / real < 0.2


@pytest.mark.parametrize("masking,rate", [("uniform5", 0.05), ("uniform10", 0.10)])
def test_anneal_rates_mask_less_with_the_same_handling(tm, masking, rate):
    tokenizer = _eval_tokenizer()
    collator = tm.create_masking_collator(masking, tokenizer)
    assert type(collator) is tm.PackedMLMDataCollator
    input_ids = np.random.RandomState(0).randint(5, 200, size=(64, 256)).astype(np.int32)
    np.random.seed(0)
    batch = collator({"input_ids": input_ids, "input_ids_segment_ids": np.zeros_like(input_ids), "input_ids_positions": np.zeros_like(input_ids)})
    masked = batch["labels"] != -100
    assert masked.mean() == pytest.approx(rate, abs=0.01)
    assert np.mean(batch["input_ids"][masked] == tokenizer.mask_token_id) == pytest.approx(0.8, abs=0.03)
    # The held-out masks stay those of 15% single tokens
    eval_collator = tm.create_eval_collator(masking, collator, tokenizer)
    assert type(eval_collator) is tm.PackedMLMDataCollator and eval_collator.mlm_probability == 0.15
    assert _masking_args(tm, "--masking", masking, "--use-packing").masking == masking


def test_uniform30_masks_twice_as_much_with_the_same_handling(tm):
    tokenizer = _eval_tokenizer()
    collator = tm.create_masking_collator("uniform30", tokenizer)
    assert type(collator) is tm.PackedMLMDataCollator
    input_ids = np.random.RandomState(0).randint(5, 200, size=(64, 256)).astype(np.int32)
    np.random.seed(0)
    batch = collator({"input_ids": input_ids, "input_ids_segment_ids": np.zeros_like(input_ids), "input_ids_positions": np.zeros_like(input_ids)})
    masked = batch["labels"] != -100
    assert masked.mean() == pytest.approx(0.30, abs=0.01)
    assert np.mean(batch["input_ids"][masked] == tokenizer.mask_token_id) == pytest.approx(0.8, abs=0.02)


DECAY_OPTIONS = {"--masking": "decay", "--use-packing": None, "--web-mask-rate": "0.05", "--answer-full-max": "16",
                 "--data-files-list": "files.txt"}


def _flags(options):
    return [part for flag, value in options.items() for part in ([flag] if value is None else [flag, value])]


def test_decay_flags_need_each_other_and_the_mixed_files(tm):
    args = _masking_args(tm, *_flags(DECAY_OPTIONS))
    assert (args.masking, args.web_mask_rate, args.answer_full_max) == ("decay", 0.05, 16)
    refused = [{key: value for key, value in DECAY_OPTIONS.items() if key != left_out} for left_out in DECAY_OPTIONS if left_out != "--masking"]
    refused += [{**DECAY_OPTIONS, "--masking-mode": "instruction_response"}, {**DECAY_OPTIONS, "--datasets-config": "c.json"},
                {**DECAY_OPTIONS, "--web-mask-rate": "0"}, {**DECAY_OPTIONS, "--answer-full-max": "0"},
                # The decay knobs mean nothing on another masking: refused rather than ignored
                {**DECAY_OPTIONS, "--masking": "uniform5"}, {"--use-packing": None, "--answer-full-max": "16"}]
    for options in refused:
        with pytest.raises(SystemExit):
            _masking_args(tm, *_flags(options))


def _decay_identity_args(tm, rate=0.05, full_max=16):
    return types.SimpleNamespace(dataset="d", dataset_config="c", datasets_config=None, max_length=1024, tokenizer="t",
                                 shuffle_buffer_size=10, web_mask_rate=rate, answer_full_max=full_max)


def test_decay_identity_holds_its_masking_and_its_mix(tm, tmp_path):
    identity = tm.data_state_identity(_decay_identity_args(tm), 32, ["mix-abc.parquet"], "decay", "abc")
    assert {key: identity[key] for key in ("masking", "web_mask_rate", "answer_full_max", "answer_token", "mix")} == {
        "masking": "decay", "web_mask_rate": 0.05, "answer_full_max": 16, "answer_token": "<answer>", "mix": "abc"}
    # Arm W's identity has none of these: its uniform5 state is no state of F, nor F's of W
    uniform5 = tm.data_state_identity(_decay_identity_args(tm), 32, ["mix-abc.parquet"], "uniform5")
    assert uniform5["masking"] == "uniform5" and "mix" not in uniform5 and "web_mask_rate" not in uniform5

    path = tm.epath.Path(tmp_path) / "state.json"
    loader = mock.Mock(get_state=mock.Mock(return_value={"epoch": 0}))
    tm.save_data_state(path, loader, identity, 2000)
    tm.restore_data_state(path, loader, identity, allow_restart=False)
    for other in (tm.data_state_identity(_decay_identity_args(tm, rate=0.1), 32, ["mix-abc.parquet"], "decay", "abc"),
                  tm.data_state_identity(_decay_identity_args(tm, full_max=8), 32, ["mix-abc.parquet"], "decay", "abc"),
                  tm.data_state_identity(_decay_identity_args(tm), 32, ["mix-abc.parquet"], "decay", "def"), uniform5):
        with pytest.raises(RuntimeError, match="another data setup"):
            tm.restore_data_state(path, loader, other, allow_restart=True)


def test_decay_collator_of_the_flag_and_its_held_out_masks(tm):
    from mmbjax.training import add_answer_token

    tokenizer = _eval_tokenizer()
    answer_token_id = add_answer_token(tokenizer)
    collator = tm.create_masking_collator("decay", tokenizer, 0.05, 16, answer_token_id)
    assert type(collator) is tm.DecayCollator
    assert (collator.mlm_probability, collator.answer_full_max, collator.answer_token_id) == (0.05, 16, answer_token_id)
    # Held-out loss on the masks of every other run: 15% single tokens, no roles needed
    eval_collator = tm.create_eval_collator("decay", collator, tokenizer)
    assert type(eval_collator) is tm.PackedMLMDataCollator and eval_collator.mlm_probability == 0.15


def test_decay_mix_build_comes_from_the_files_of_the_builder(tm, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def write(name, build=None):
        path = tmp_path / name
        table = pa.table({"text": ["a"]})
        if build is not None:
            table = table.replace_schema_metadata({tm.DECAY_MIX_METADATA_KEY: build.encode()})
        pq.write_table(table, path)
        return str(path)

    import build_decay_mix

    assert build_decay_mix.METADATA_KEY == tm.DECAY_MIX_METADATA_KEY
    assert tm.decay_mix_build([write("mix.parquet", "0123456789ab")]) == "0123456789ab"
    with pytest.raises(ValueError, match="no file of scripts/build_decay_mix.py"):
        tm.decay_mix_build([write("nl_part_00033.parquet")])
    with pytest.raises(ValueError, match="more than one build"):
        tm.decay_mix_build([write("a.parquet", "a"), write("b.parquet", "b")])
