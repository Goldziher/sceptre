---
priority: high
---

# OCR Pipeline

The pipeline reimplements EasyOCR's "latest version" path: the CRAFT detector plus the gen2 (`*_g2`) CRNN recognizers, run over ONNX. Legacy gen1 models are out of scope.

## Stages

1. **Load** (`imaging`) — decode into RGB (for detection) and grayscale (for recognition), mirroring EasyOCR `reformat_input`.
2. **Detect** (`detect`) — aspect-ratio resize + ImageNet mean/variance normalize → CRAFT → region/link heat-maps at half resolution → threshold + connected components → boxes → group into lines (horizontal vs. rotated). Refs: `imgproc.py`, `craft_utils.py`, `utils.py:group_text_box`.
3. **Recognize** (`recognize`) — crop each region (perspective transform for rotated quads), resize to height 64, normalize `(x-0.5)/0.5`, pad the batch → CRNN → CTC logits `[B, T, num_classes]` → greedy decode (blank = index 0) with the `custom_mean` confidence. Refs: `recognition.py`, `utils.py:CTCLabelConverter`.

## Models

- Source: the `itextresearch/itext-EasyOCR-*` ONNX repos on Hugging Face (Apache-2.0, dynamic-width): `craft_mlt_25k` + `english_g2`, `latin_g2`, `zh_sim_g2`, `japanese_g2`, `korean_g2`, `cyrillic_g2`.
- Downloaded at runtime, cached under `~/.cache/easyocr-rs`, sha256-verified.

## Key tunables (EasyOCR defaults)

- Detection: `text_threshold 0.7`, `link_threshold 0.4`, `low_text 0.4`, `canvas_size 2560`, `mag_ratio 1.0`.
- Recognition: greedy decoder, `imgH 64`, per-language charset with a blank-prefixed CTC class list.

## Parity

- Validate against EasyOCR golden output (regenerated as JSON from the example images) — text equality per line plus a box-IoU threshold.
