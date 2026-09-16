"""Efficiency / deployment metrics.

Covers:
  - model_size_mb       : on-disk size of state_dict
  - params_m            : #parameters in millions
  - peak_memory_mb      : peak memory during inference (CPU + GPU)
  - quant_bits          : quantization bit width of the model
  - compression_ratio   : compressed_dim / total_dim
  - load_time_s         : checkpoint load wall time
  - quant_feedback_overhead_bytes : bytes transmitted over-the-air for
                                    one feedback codeword (CR × quant_bits)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict

import torch

from ..core.context import EvalContext
from ..core.registries import MetricRegistry


# ---------------------------------------------------------------------------
# ModelSize
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class ModelSize:
    """On-disk state_dict size in MB (after dumping)."""

    name = "size"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "MB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        size_mb = ctx.model.get_state_dict_mb()
        return {"value": float(size_mb), "unit": "MB"}


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class ParamsM:
    """Total #parameters in millions."""

    name = "params"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "M"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        params = sum(p.numel() for p in ctx.model.parameters() if hasattr(p, "numel"))
        # Duck-typed: if the model has .parameters() we count; otherwise 0
        return {"value": params / 1e6, "raw": int(params), "unit": "M"}


# ---------------------------------------------------------------------------
# PeakMemory
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class PeakMemory:
    """Peak memory during one forward pass (MB).

    CPU uses tracemalloc; CUDA uses torch.cuda.max_memory_allocated().
    """

    name = "memory"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "MB"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        input_shape = self._resolve_shape(ctx)
        if input_shape is None:
            return {"value": None, "note": "no input_shape available (set input_shape in model_meta.json)"}
        ctx.model.eval()
        x = torch.randn((1, *input_shape), device=ctx.device)
        with torch.no_grad():
            _ = ctx.model.forward(x)
        if str(ctx.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(ctx.device) / (1024 ** 2)
        else:
            import tracemalloc
            tracemalloc.start()
            with torch.no_grad():
                _ = ctx.model.forward(x)
            current, peak_py = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = max(current, peak_py) / (1024 ** 2)
        return {"value": float(peak), "unit": "MB"}

    @staticmethod
    def _resolve_shape(ctx: EvalContext):
        try:
            return tuple(ctx.model.get_input_shape())
        except Exception:
            pass
        shape = ctx.meta("input_shape")
        if shape:
            return tuple(shape)
        return None


# ---------------------------------------------------------------------------
# QuantBits
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class QuantBits:
    """Quantization bit width (0 means no quantization)."""

    name = "bitwidth"
    category = "storage"
    higher_is_better = False
    requires = frozenset()

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        q = int(ctx.meta("quant_bits", 0))
        return {"value": q, "effective": q, "unit": "bits"}


# ---------------------------------------------------------------------------
# CompressionRatio
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class CompressionRatio:
    """Feedback code dimension / total dimension.

    Values come entirely from model_meta.json — no model methods required.
    """

    name = "compression"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = ""

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        cr = ctx.meta("compression_ratio")
        compressed_dim = ctx.meta("compressed_dim")
        total_dim = ctx.meta("total_dim")
        reduction = ctx.meta("reduction")

        if cr is None and compressed_dim and total_dim:
            cr = float(compressed_dim) / float(total_dim)

        return {
            "value": float(cr) if cr is not None else None,
            "compressed_dim": int(compressed_dim) if compressed_dim else None,
            "total_dim": int(total_dim) if total_dim else None,
            "reduction": int(reduction) if reduction else None,
        }


# ---------------------------------------------------------------------------
# CSIReductionRate
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset(), higher_is_better=True)
class CSIReductionRate:
    """CSI Feedback overhead reduction rate vs Rel-16/17 Type II codebook baseline.

    Rel-16/17 Type II codebook feedback overhead:
      - Type II (2x2):  4 bits per real entry in the (Nt×Nr) beam matrix
        ≈ 4 * Nt * Nr * (1 + 2*log2(Nt)) bits per subband
      - For 2.6 GHz (Nt=32, Nr=4): 4 * 32 * 4 * (1 + 2*log2(32))
        = 512 * (1 + 10) = 5632 bits ≈ 704 bytes per subband (≈8800 bits total)
      - Per-subband overhead for Type II ≈ 4 * Nt * Nr = 512 bits

    Our model's overhead: compressed_dim * quant_bits / 8 bytes per codeword.
    Reduction rate = (baseline_overhead - model_overhead) / baseline_overhead * 100%

    Higher is better (positive = we reduce overhead vs Type II).
    """

    name = "csi_reduction_rate"
    category = "storage"
    higher_is_better = True
    requires = frozenset(["model.compression_ratio"])
    unit = "%"

    # Type II codebook overhead per subband (in bits), keyed by (Nt, Nr)
    TYPEII_OVERHEAD_BITS: Dict[tuple, float] = {
        # (Nt, Nr): bits per subband
        (32, 4): 512.0,   # 2.6 GHz: 4 * Nt * Nr
        (64, 4): 1024.0,  # 3.5 GHz
        (256, 8): 8192.0, # 7 GHz: 4 * 256 * 8
    }

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        # All values come from model_meta.json — the model contributes nothing here.
        nt = ctx.meta("nt") or self._data_nt(ctx) or 32
        nr = ctx.meta("nr") or 4
        n_subbands = ctx.meta("n_subbands") or self._data_k(ctx)
        compressed_dim = ctx.meta("compressed_dim")
        total_dim = ctx.meta("total_dim") or (2 * nt * n_subbands)
        q = int(ctx.meta("quant_bits", 0))

        if compressed_dim is None:
            cr = ctx.meta("compression_ratio")
            if cr is not None and total_dim:
                compressed_dim = int(round(cr * total_dim))

        baseline_bits = self.TYPEII_OVERHEAD_BITS.get((nt, nr), 512.0)
        if n_subbands:
            baseline_bits *= n_subbands

        bits_per_code = max(q, 4)
        model_bits = (int(compressed_dim) if compressed_dim else 0) * bits_per_code
        if n_subbands:
            model_bits *= n_subbands

        reduction_pct = 0.0
        if baseline_bits > 0:
            reduction_pct = float(100.0 * (baseline_bits - model_bits) / baseline_bits)

        return {
            "value": reduction_pct,
            "unit": "%",
            "baseline_bits": baseline_bits,
            "model_bits": float(model_bits),
            "compressed_dim": int(compressed_dim) if compressed_dim else None,
            "quant_bits": q,
        }

    @staticmethod
    def _data_nt(ctx: EvalContext) -> Optional[int]:
        try:
            return getattr(ctx.data, "get_metadata", lambda: {})().get("nt")
        except Exception:
            return None

    @staticmethod
    def _data_k(ctx: EvalContext) -> Optional[int]:
        try:
            return getattr(ctx.data, "get_metadata", lambda: {})().get("n_subbands")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# LoadTime
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class LoadTime:
    """Time to load state_dict into the model (seconds)."""

    name = "loadtime"
    category = "storage"
    higher_is_better = False
    requires = frozenset()
    unit = "s"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        ckpt_path = self._resolve_checkpoint(ctx)
        if not ckpt_path or not os.path.exists(ckpt_path):
            return {"value": None, "note": "no checkpoint path"}
        t0 = time.perf_counter()
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        # Move model to CPU temporarily for fair timing
        original_device = next(ctx.model.parameters()).device
        ctx.model.cpu()
        try:
            ctx.model.load_state_dict(sd, strict=False)
        except Exception as e:
            return {"value": None, "note": f"load failed: {e}"}
        finally:
            ctx.model.to(original_device)
        elapsed = time.perf_counter() - t0
        return {"value": float(elapsed), "unit": "s"}

    @staticmethod
    def _resolve_checkpoint(ctx: EvalContext) -> Optional[str]:
        """Resolve checkpoint path: explicit config.checkpoint first,
        otherwise fall back to the bundle's default checkpoint."""
        if ctx.config.checkpoint:
            return ctx.config.checkpoint
        bundle_dir = ctx.config.model_bundle
        if not bundle_dir:
            return None
        meta_path = os.path.join(bundle_dir, "model_meta.json")
        default_name = None
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                default_name = meta.get("default_checkpoint")
            except Exception:
                default_name = None
        if default_name:
            candidate = os.path.join(bundle_dir, default_name)
            if os.path.exists(candidate):
                return candidate
        # Otherwise, pick the first .pt/.pth in the bundle directory
        for ext in ("*.pt", "*.pth"):
            import glob
            found = glob.glob(os.path.join(bundle_dir, ext))
            if found:
                return found[0]
        return None


# ---------------------------------------------------------------------------
# QuantFeedbackOverhead
# ---------------------------------------------------------------------------
@MetricRegistry.register("storage", requires=frozenset())
class QuantFeedbackOverhead:
    """Over-the-air feedback overhead in bytes per codeword.

    Definition: compressed_dim * quant_bits / 8 bytes.
    For an unquantized model (quant_bits=0) we use 4 bytes (float32) per
    code element by convention.

    Lower is better.
    """

    name = "overhead"
    category = "storage"
    higher_is_better = False
    requires = frozenset(["model.compression_ratio"])
    unit = "bytes"

    def compute(self, ctx: EvalContext) -> Dict[str, Any]:
        # All values from model_meta.json — model contributes nothing.
        nt = ctx.meta("nt") or 32
        n_subbands = ctx.meta("n_subbands")
        compressed_dim = ctx.meta("compressed_dim")
        total_dim = ctx.meta("total_dim") or (2 * nt * (n_subbands or 13))
        q = int(ctx.meta("quant_bits", 0))

        if compressed_dim is None:
            cr = ctx.meta("compression_ratio")
            if cr is not None and total_dim:
                compressed_dim = int(round(cr * total_dim))

        bits_per_code = max(q, 4)
        bytes_per_codeword = (int(compressed_dim) if compressed_dim else 0) * bits_per_code // 8
        return {
            "value": bytes_per_codeword,
            "compressed_dim": int(compressed_dim) if compressed_dim else None,
            "quant_bits_effective": int(bits_per_code),
            "unit": "bytes/codeword",
        }
