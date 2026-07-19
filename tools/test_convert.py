#!/usr/bin/env python3
"""Unit tests for tools/convert.py rename() and classify() functions.

Run:  python3 tools/test_convert.py
Exits 0 on success, non-zero on failure. No external deps beyond numpy
(which the converter already requires); does NOT load safetensors/torch.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import rename, classify, flatten_config, M3_EOS_TOKEN_ID

FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    if got != expected:
        FAILURES.append(f"{label}: got {got!r}, expected {expected!r}")


# ---------------------------------------------------------------------------
# rename() — must strip `language_model.` and rewrite `block_sparse_moe` → `mlp`
# ---------------------------------------------------------------------------

def test_rename_strips_language_model_prefix():
    """VAL-FOUND-002: language_model. prefix stripped from all tensor names."""
    check("strip embed",
          rename("language_model.model.embed_tokens.weight"),
          "model.embed_tokens.weight")
    check("strip lm_head",
          rename("language_model.lm_head.weight"),
          "lm_head.weight")
    check("strip norm",
          rename("language_model.model.norm.weight"),
          "model.norm.weight")
    check("strip layernorm",
          rename("language_model.model.layers.0.input_layernorm.weight"),
          "model.layers.0.input_layernorm.weight")
    check("strip q_proj",
          rename("language_model.model.layers.5.self_attn.q_proj.weight"),
          "model.layers.5.self_attn.q_proj.weight")


def test_rename_rewrites_block_sparse_moe_to_mlp():
    """VAL-FOUND-003: block_sparse_moe → mlp in tensor names."""
    # Router gate
    check("router gate",
          rename("language_model.model.layers.3.block_sparse_moe.gate.weight"),
          "model.layers.3.mlp.gate.weight")
    # Shared experts
    check("shared gate",
          rename("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight"),
          "model.layers.3.mlp.shared_experts.gate_proj.weight")
    check("shared up",
          rename("language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight"),
          "model.layers.3.mlp.shared_experts.up_proj.weight")
    check("shared down",
          rename("language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj.weight"),
          "model.layers.3.mlp.shared_experts.down_proj.weight")
    # Routed experts
    check("expert gate_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.gate_proj.weight"),
          "model.layers.3.mlp.experts.0.gate_proj.weight")
    check("expert up_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.up_proj.weight"),
          "model.layers.3.mlp.experts.0.up_proj.weight")
    check("expert down_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.down_proj.weight"),
          "model.layers.3.mlp.experts.0.down_proj.weight")
    # Also accept the w1/w2/w3 naming variant (some HF snapshots use it).
    # HF MiniMax-M3 uses w1/w2/w3 → engine expects gate_proj/down_proj/up_proj.
    check("expert w1 -> gate_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight"),
          "model.layers.3.mlp.experts.0.gate_proj.weight")
    check("expert w3 -> up_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.w3.weight"),
          "model.layers.3.mlp.experts.0.up_proj.weight")
    check("expert w2 -> down_proj",
          rename("language_model.model.layers.3.block_sparse_moe.experts.0.w2.weight"),
          "model.layers.3.mlp.experts.0.down_proj.weight")


def test_rename_e_score_correction_bias_under_gate():
    """e_score_correction_bias moves under mlp.gate.* (engine expectation)."""
    check("e_score_correction_bias",
          rename("language_model.model.layers.3.block_sparse_moe.e_score_correction_bias"),
          "model.layers.3.mlp.gate.e_score_correction_bias")


def test_rename_idempotent():
    """rename() on an already-canonical name returns it unchanged."""
    for n in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ):
        check(f"idempotent {n}", rename(n), n)


def test_rename_shared_mlp_to_shared_experts():
    """Older HF naming variant shared_mlp → shared_experts."""
    check("shared_mlp -> shared_experts",
          rename("language_model.model.layers.3.block_sparse_moe.shared_mlp.gate_proj.weight"),
          "model.layers.3.mlp.shared_experts.gate_proj.weight")


# ---------------------------------------------------------------------------
# classify() — prefix-agnostic, correct dtype policy per VAL-FOUND-004
# ---------------------------------------------------------------------------

def test_classify_embed_and_lm_head_are_io():
    """embed_tokens.weight and lm_head.weight → io (int8)."""
    check("embed io", classify("language_model.model.embed_tokens.weight"), "io")
    check("lm_head io", classify("language_model.lm_head.weight"), "io")
    # Already-canonical names too
    check("embed canonical io", classify("model.embed_tokens.weight"), "io")
    check("lm_head canonical io", classify("lm_head.weight"), "io")


def test_classify_routers_are_f32():
    """*.mlp.gate.weight → f32."""
    check("router f32", classify("language_model.model.layers.3.block_sparse_moe.gate.weight"), "f32")
    check("router canonical f32", classify("model.layers.3.mlp.gate.weight"), "f32")
    check("router layer 59 f32", classify("model.layers.59.mlp.gate.weight"), "f32")


def test_classify_e_score_correction_bias_is_f32():
    """e_score_correction_bias → f32 (not int4 expert)."""
    check("bias f32",
          classify("language_model.model.layers.3.block_sparse_moe.e_score_correction_bias"), "f32")
    check("bias canonical f32",
          classify("model.layers.3.mlp.gate.e_score_correction_bias"), "f32")


def test_classify_norms_are_f32():
    """All norms (input_layernorm, post_attention_layernorm, model.norm) → f32."""
    check("input_ln f32",
          classify("language_model.model.layers.0.input_layernorm.weight"), "f32")
    check("post_ln f32",
          classify("language_model.model.layers.0.post_attention_layernorm.weight"), "f32")
    check("model.norm f32", classify("language_model.model.norm.weight"), "f32")
    check("model.norm canonical f32", classify("model.norm.weight"), "f32")


def test_classify_per_head_qk_norms_are_f32():
    """q_norm / k_norm (per-head norms) → f32."""
    check("q_norm f32",
          classify("language_model.model.layers.5.self_attn.q_norm.weight"), "f32")
    check("k_norm f32",
          classify("language_model.model.layers.5.self_attn.k_norm.weight"), "f32")


def test_classify_index_norms_are_f32():
    """index_q_norm / index_k_norm → f32 (per VAL-FOUND-004 table)."""
    check("index_q_norm f32",
          classify("language_model.model.layers.5.self_attn.index_q_norm.weight"), "f32")
    check("index_k_norm f32",
          classify("language_model.model.layers.5.self_attn.index_k_norm.weight"), "f32")


def test_classify_experts_are_expert():
    """*.mlp.experts.* → expert (int4)."""
    check("expert gate_proj expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.gate_proj.weight"), "expert")
    check("expert up_proj expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.up_proj.weight"), "expert")
    check("expert down_proj expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.down_proj.weight"), "expert")
    check("expert w1 expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight"), "expert")
    check("expert w2 expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.w2.weight"), "expert")
    check("expert w3 expert",
          classify("language_model.model.layers.3.block_sparse_moe.experts.0.w3.weight"), "expert")
    check("expert 127 expert",
          classify("language_model.model.layers.59.block_sparse_moe.experts.127.down_proj.weight"), "expert")


def test_classify_shared_experts_are_expert():
    """*.mlp.shared_experts.* → expert (int4)."""
    check("shared gate expert",
          classify("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight"), "expert")
    check("shared up expert",
          classify("language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight"), "expert")
    check("shared down expert",
          classify("language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj.weight"), "expert")


def test_classify_attention_projections_are_attn():
    """*.self_attn.{q,k,v,o}_proj.weight → attn (int4)."""
    for suf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        check(f"attn {suf}",
              classify(f"language_model.model.layers.5.self_attn.{suf}.weight"), "attn")
        check(f"attn canonical {suf}",
              classify(f"model.layers.5.self_attn.{suf}.weight"), "attn")


def test_classify_index_projections_are_attn():
    """index_q_proj / index_k_proj → attn (int4, per VAL-FOUND-004 table)."""
    check("index_q_proj attn",
          classify("language_model.model.layers.5.self_attn.index_q_proj.weight"), "attn")
    check("index_k_proj attn",
          classify("language_model.model.layers.5.self_attn.index_k_proj.weight"), "attn")


def test_classify_dense_mlp_is_dense():
    """Dense layers 0-2 mlp.{gate,up,down}_proj.weight → dense (int4 with dbits=4)."""
    for suf in ("gate_proj", "up_proj", "down_proj"):
        check(f"dense {suf}",
              classify(f"language_model.model.layers.0.mlp.{suf}.weight"), "dense")


# ---------------------------------------------------------------------------
# flatten_config() — eos_token_id null → 200020 fallback (VAL-FOUND-005)
# ---------------------------------------------------------------------------

def test_flatten_config_eos_null_fallback():
    """When eos_token_id is null in the source config, flatten_config uses 200020."""
    raw = {
        "text_config": {
            "hidden_size": 6144, "num_hidden_layers": 60, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 200064,
            "moe_layer_freq": [0, 0, 0, 1, 1],
        },
        "eos_token_id": None,
    }
    cfg = flatten_config(raw)
    check("eos null -> 200020", cfg["eos_token_id"], M3_EOS_TOKEN_ID)
    check("eos == 200020", cfg["eos_token_id"], 200020)


def test_flatten_config_eos_present_preserved():
    """When eos_token_id is set, flatten_config preserves it."""
    raw = {
        "text_config": {
            "hidden_size": 6144, "num_hidden_layers": 60, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 200064,
            "moe_layer_freq": [0, 0, 0, 1, 1],
        },
        "eos_token_id": 200020,
    }
    cfg = flatten_config(raw)
    check("eos present preserved", cfg["eos_token_id"], 200020)


def test_flatten_config_eos_list_takes_first():
    """When eos_token_id is a list, flatten_config takes the first element."""
    raw = {
        "text_config": {
            "hidden_size": 6144, "num_hidden_layers": 60, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 200064,
            "moe_layer_freq": [0, 0, 0, 1, 1],
        },
        "eos_token_id": [200020, 200021],
    }
    cfg = flatten_config(raw)
    check("eos list first", cfg["eos_token_id"], 200020)


def test_flatten_config_eos_empty_list_fallback():
    """When eos_token_id is an empty list, flatten_config falls back to 200020."""
    raw = {
        "text_config": {
            "hidden_size": 6144, "num_hidden_layers": 60, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 200064,
            "moe_layer_freq": [0, 0, 0, 1, 1],
        },
        "eos_token_id": [],
    }
    cfg = flatten_config(raw)
    check("eos empty list -> 200020", cfg["eos_token_id"], 200020)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main() -> int:
    for t in TESTS:
        t()
        if FAILURES:
            break
    if FAILURES:
        print(f"FAIL ({len(FAILURES)} failures):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"PASS: {len(TESTS)} tests, all rename()/classify()/flatten_config() checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
