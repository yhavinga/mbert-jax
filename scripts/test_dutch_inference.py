#!/usr/bin/env python3
"""
Quick Dutch inference test - see if the models can predict reasonable Dutch words.
Carmack-style: Just show me if it works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pickle
import json
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from mmbjax.configuration_mmbbert import MMBertConfig
from mmbjax.heads.masked_lm import ModernBertForMaskedLM


def load_checkpoint(checkpoint_path: Path):
    """Load converted checkpoint."""
    with open(checkpoint_path / "params.pkl", 'rb') as f:
        params = pickle.load(f)

    with open(checkpoint_path / "config.json", 'r') as f:
        config_dict = json.load(f)

    config = MMBertConfig(
        vocab_size=config_dict.get("vocab_size", 256000),
        hidden_size=config_dict.get("hidden_size", 768),
        num_hidden_layers=config_dict.get("num_hidden_layers", 6),
        num_attention_heads=config_dict.get("num_attention_heads", 12),
        intermediate_size=config_dict.get("intermediate_size", 3072),
        max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
    )

    return params, config


def predict_masked_word(model, params, tokenizer, text):
    """Predict the masked word in a sentence."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    attention_mask = jnp.array(inputs["attention_mask"])

    # Find mask position
    mask_token_id = tokenizer.mask_token_id
    mask_positions = jnp.where(input_ids == mask_token_id)

    if len(mask_positions[0]) == 0:
        print(f"No mask token found in: {text}")
        return None

    # Get model predictions
    outputs = model.apply(
        {"params": params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True
    )

    # Get predictions for masked position
    mask_idx = mask_positions[1][0]  # First mask position
    logits = outputs.logits[0, mask_idx]

    # Get top 5 predictions
    top_k = 5
    top_indices = jnp.argsort(logits)[-top_k:][::-1]
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    top_probs = jax.nn.softmax(logits)[top_indices]

    return list(zip(top_tokens, top_probs))


def main():
    print("=" * 60)
    print("DUTCH INFERENCE TEST")
    print("=" * 60)

    # Test sentences - common Dutch patterns (use <mask> not [MASK])
    test_sentences = [
        "Amsterdam is de <mask> van Nederland.",  # hoofdstad (capital)
        "De kat zit op de <mask>.",              # mat/bank/stoel
                                                  # (mat/couch/chair)
        "Het weer is vandaag erg <mask>.",       # mooi/slecht/koud
                                                  # (nice/bad/cold)
        "Ik drink graag een kopje <mask>.",      # koffie/thee (coffee/tea)
        "Nederland is bekend om zijn <mask>.",  # tulpen/kaas/molens
                                                # (tulips/cheese/windmills)
        "De trein vertrekt om <mask> uur.",      # Expected: number
        "Morgen ga ik naar de <mask>.",         # winkel/stad/school
                                                 # (shop/city/school)
        "Dit boek is heel <mask> geschreven.",  # goed/mooi/slecht
                                                # (well/beautifully/badly)
    ]

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    # Test both checkpoints
    checkpoints = {
        "Red (1M - Stable)": Path("experiments/converted/red_1m"),
        "Yellow (2M - Exploded)": Path("experiments/converted/yellow_2m")
    }

    for name, checkpoint_path in checkpoints.items():
        print(f"\n{'=' * 40}")
        print(f"{name}")
        print(f"{'=' * 40}")

        # Load checkpoint
        print(f"Loading {checkpoint_path}...")
        params, config = load_checkpoint(checkpoint_path)

        # Create model
        model = ModernBertForMaskedLM(config)

        # Test each sentence
        for sentence in test_sentences:
            print(f"\nInput: {sentence}")
            predictions = predict_masked_word(model, params, tokenizer, sentence)

            if predictions:
                print("Predictions:")
                for token, prob in predictions:
                    # Clean up token display
                    token = token.strip()
                    if token.startswith("##"):
                        token = token[2:]  # Remove subword marker
                    print(f"  {token:15} ({prob:.2%})")

    # Direct comparison on key sentence
    print("\n" + "=" * 60)
    print("DIRECT COMPARISON: 'Amsterdam is de <mask> van Nederland.'")
    print("=" * 60)

    test_sentence = "Amsterdam is de <mask> van Nederland."

    for name, checkpoint_path in checkpoints.items():
        params, config = load_checkpoint(checkpoint_path)
        model = ModernBertForMaskedLM(config)
        predictions = predict_masked_word(model, params, tokenizer, test_sentence)

        print(f"\n{name}:")
        if predictions:
            top_prediction = predictions[0][0].strip()
            if top_prediction.startswith("##"):
                top_prediction = top_prediction[2:]
            confidence = predictions[0][1]
            print(f"  Top prediction: '{top_prediction}' ({confidence:.1%} confidence)")

            # Check if it got "hoofdstad" (capital)
            for token, prob in predictions:
                token = token.strip()
                if "hoofd" in token.lower() or "stad" in token.lower():
                    print(f"  ✓ Found related word: '{token}' at {prob:.1%}")
                    break

    print("\n" + "=" * 60)
    print("QUALITATIVE ASSESSMENT")
    print("=" * 60)
    print("Check if predictions make sense in Dutch context.")
    print("Expected good predictions:")
    print("  - 'hoofdstad' for Amsterdam")
    print("  - 'mat/bank/stoel' for where cat sits")
    print("  - 'koffie/thee' for what to drink")
    print("  - Numbers for train time")


if __name__ == "__main__":
    main()