"""First-party ``.pth -> ONNX`` export for CRAFT + the gen2 CRNN recognizers.

This is the primary model-export path (ADR 0025): it loads EasyOCR's own PyTorch
networks and re-exports them as the ONNX artifacts sceptre hosts on the ``xberg-io``
Hugging Face org. Requires the opt-in ``export`` dependency group (torch, easyocr, onnx,
onnxruntime, numpy); without it this exits cleanly with an explanatory message.

The exports satisfy the runtime I/O contract the Rust backends expect (see ADR 0025):

- single f32 input, single f32 output, bound by position (tensor names are cosmetic);
- **no** normalization baked into the graph — the Rust pipeline pre-normalizes;
- **no** softmax on the recognizer — ``recognize/ctc.rs`` applies it;
- recognizer input ``[B, 1, 64, W]`` -> output ``[B, T, num_classes]`` raw CTC logits,
  class 0 = blank, dynamic batch + width;
- CRAFT input ``[1, 3, H, W]`` -> single ``[1, H/2, W/2, 2]`` region/link heat-map,
  dynamic H + W.

gen2's ``AdaptiveAvgPool2d((None, 1))`` cannot export under a dynamic width axis
(non-constant output size), so it is swapped for a numerically identical mean over the
height dimension, which lowers to a clean ``ReduceMean``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Each gen2 recognizer network paired with a representative EasyOCR language list that
# makes ``easyocr.Reader`` build exactly that network. The charset is taken from EasyOCR's
# own config and cross-checked against the built converter, so a wrong mapping fails loudly.
RECOGNIZER_NETWORKS: dict[str, list[str]] = {
    "english_g2": ["en"],
    "latin_g2": ["fr"],
    "zh_sim_g2": ["ch_sim"],
    "japanese_g2": ["ja"],
    "korean_g2": ["ko"],
    "cyrillic_g2": ["ru"],
    "telugu_g2": ["te"],
    "kannada_g2": ["kn"],
}

CRAFT_NETWORK = "craft_mlt_25k"

# torch <-> onnxruntime agreement tolerance; both wrap the same math, so a correct export
# agrees to well under this. A breach means the export is wrong, not merely imprecise.
PARITY_TOLERANCE = 1e-3
OPSET_VERSION = 17


def repo_root() -> Path:
    """Locate the repository root from this file's location."""
    return Path(__file__).resolve().parents[2]


def character_assets_dir() -> Path:
    """Directory holding the committed per-recognizer charset assets."""
    return repo_root() / "crates" / "sceptre" / "assets" / "character"


def default_out_dir() -> Path:
    """Default (git-ignored) directory for the exported ONNX artifacts."""
    return repo_root() / "target" / "model-export"


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's contents (the registry pin)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_torch_wrappers() -> tuple[object, object, object]:
    """Import torch lazily and build the export wrapper modules.

    Returns ``(torch, RecognizerExport, DetectorExport)`` where the two classes wrap an
    EasyOCR network into the single-input / single-output forward the export requires.
    """
    import torch
    from torch import nn

    class MeanPoolH(nn.Module):
        """``AdaptiveAvgPool2d((None, 1))`` over ``[b, w, c, h]`` == mean over ``h``."""

        def forward(self, x: object) -> object:
            return x.mean(dim=3, keepdim=True)

    class RecognizerExport(nn.Module):
        """Recognizer forward reduced to ``image -> logits`` (CTC ignores the text arg)."""

        def __init__(self, model: object) -> None:
            super().__init__()
            model.AdaptiveAvgPool = MeanPoolH()
            self.model = model

        def forward(self, image: object) -> object:
            return self.model(image, None)

    class DetectorExport(nn.Module):
        """CRAFT forward reduced to ``image -> heatmaps`` (drop the feature output)."""

        def __init__(self, model: object) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: object) -> object:
            out = self.model(image)
            return out[0] if isinstance(out, (tuple, list)) else out

    return torch, RecognizerExport, DetectorExport


def _validate_parity(torch: object, wrapper: object, onnx_path: Path, shapes: list[tuple[int, ...]]) -> float:
    """Assert the exported ONNX matches torch under onnxruntime; return the max abs diff."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_diff = 0.0
    for shape in shapes:
        sample = np.random.rand(*shape).astype(np.float32)
        with torch.no_grad():
            reference = wrapper(torch.from_numpy(sample)).numpy()
        actual = session.run(None, {input_name: sample})[0]
        max_diff = max(max_diff, float(np.abs(reference - actual).max()))
    if max_diff > PARITY_TOLERANCE:
        raise RuntimeError(f"{onnx_path.name}: torch/onnxruntime disagree by {max_diff:.3e} > {PARITY_TOLERANCE:.0e}")
    return max_diff


def export_recognizer(net: str, languages: list[str], out_dir: Path, write_assets: bool) -> dict[str, object]:
    """Export one gen2 recognizer to ONNX + its charset asset; return a summary dict."""
    import easyocr
    from easyocr import config

    torch, recognizer_export, _ = _build_torch_wrappers()

    reader = easyocr.Reader(languages, gpu=False, quantize=False, verbose=False)
    converter_chars = "".join(reader.converter.character[1:])  # drop the blank at index 0
    expected_chars = config.recognition_models["gen2"][net]["characters"]
    if converter_chars != expected_chars:
        raise RuntimeError(f"{net}: languages {languages} built a charset that is not {net}'s")

    wrapper = recognizer_export(reader.recognizer.eval()).eval()
    onnx_path = out_dir / f"{net}.onnx"
    dummy = torch.zeros(1, 1, 64, 256, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        dynamo=False,
        verbose=False,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    max_diff = _validate_parity(torch, wrapper, onnx_path, [(1, 1, 64, 128), (2, 1, 64, 256), (3, 1, 64, 96)])

    if write_assets:
        (character_assets_dir() / f"{net}.txt").write_text(expected_chars, encoding="utf-8")

    return {
        "net": net,
        "onnx": onnx_path,
        "sha256": sha256_file(onnx_path),
        "num_classes": len(expected_chars) + 1,
        "parity": max_diff,
    }


def export_craft(out_dir: Path) -> dict[str, object]:
    """Export the CRAFT detector to ONNX; return a summary dict."""
    import easyocr
    from torch import nn

    torch, _, detector_export = _build_torch_wrappers()

    reader = easyocr.Reader(["en"], gpu=False, quantize=False, verbose=False)
    detector = reader.detector
    if isinstance(detector, nn.DataParallel):
        detector = detector.module
    wrapper = detector_export(detector.eval()).eval()

    onnx_path = out_dir / f"{CRAFT_NETWORK}.onnx"
    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        dynamo=False,
        verbose=False,
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={"image": {2: "height", 3: "width"}, "heatmaps": {1: "height", 2: "width"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )
    max_diff = _validate_parity(torch, wrapper, onnx_path, [(1, 3, 256, 256), (1, 3, 320, 480), (1, 3, 512, 288)])

    return {"net": CRAFT_NETWORK, "onnx": onnx_path, "sha256": sha256_file(onnx_path), "parity": max_diff}


def run(out_dir: Path, only: list[str] | None, skip_craft: bool, write_assets: bool) -> int:
    """Export the requested models, validating each; print a sha256 summary."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        print(
            "sceptre_rs_tools.export: the exporter needs the optional 'export' dependency group "
            "(torch, easyocr, onnx, onnxruntime, numpy). Install it with `uv sync --group export`, "
            "then re-run `task py:export`.",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = only or list(RECOGNIZER_NETWORKS)
    summaries: list[dict[str, object]] = []

    for net in targets:
        if net not in RECOGNIZER_NETWORKS:
            print(f"sceptre_rs_tools.export: unknown recognizer '{net}'", file=sys.stderr)
            return 1
        print(f"sceptre_rs_tools.export: exporting {net} ...", file=sys.stderr)
        summary = export_recognizer(net, RECOGNIZER_NETWORKS[net], out_dir, write_assets)
        summaries.append(summary)
        print(f"  ok: {net} classes={summary['num_classes']} parity={summary['parity']:.2e}", file=sys.stderr)

    if not skip_craft and not only:
        print(f"sceptre_rs_tools.export: exporting {CRAFT_NETWORK} ...", file=sys.stderr)
        summaries.append(export_craft(out_dir))

    print(f"\nExported {len(summaries)} model(s) to {out_dir}")
    print(f"{'model':<18} {'sha256':<64} bytes")
    for summary in summaries:
        onnx_path = summary["onnx"]
        print(f"{summary['net']:<18} {summary['sha256']:<64} {onnx_path.stat().st_size}")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export CRAFT + gen2 CRNN recognizers to ONNX.")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir(), help="ONNX output directory.")
    parser.add_argument("--only", nargs="+", metavar="NET", help="Export only these recognizer networks.")
    parser.add_argument("--skip-craft", action="store_true", help="Skip the CRAFT detector export.")
    parser.add_argument("--no-assets", action="store_true", help="Do not (re)write the charset assets.")
    args = parser.parse_args()
    sys.exit(run(args.out_dir, args.only, args.skip_craft, write_assets=not args.no_assets))


if __name__ == "__main__":
    main()
