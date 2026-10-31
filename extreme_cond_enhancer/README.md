# Extreme-condition enhancer

`LightingEnhancer.enhance(face_bgr)` accepts and returns OpenCV BGR `uint8` images of
the same shape, plus a diagnostics dict (`condition`, brightness/contrast before and
after, `transform` parameters, `quality_score`, `reason`, `fallback_used`,
`latency_ms`). Normal-light crops are returned unchanged; invalid input or any failure
returns the original crop with `fallback_used=True`.

## Branch 1: deterministic transformation (`color_scaler.py`)

Detection uses LAB `L` mean and the fraction of near-black/near-white pixels. For a
`too_dark` or `too_bright` crop, with luma `Y` (edge-preserving smoothed):

| step | formula |
|---|---|
| robust stretch | `Y_s = clip(255 * (Y - P2) / max(P98 - P2, eps), 0, 255)` (skipped if `P98 - P2 < 8`) |
| adaptive gamma | `gamma = clip(ln(t) / ln(median(Y_s)/255), lo, hi)`; dark: `[MIN_GAMMA, 1]`, bright: `[1, MAX_GAMMA]` |
| tone curve | `T = 255 * (Y_s / 255) ** gamma` |
| gain map | `G = clip(T / max(Y, 1), 0, MAX_GAIN)`, `I_out[c] = I[c] * G` |
| clean-up | NL-means denoise (dark only), CLAHE on LAB `L` only |

Because every channel is multiplied by the same gain, the B:G:R ratio (chromaticity) is
preserved, so skin tones are not shifted. Saturated (255) pixels hold no information;
over-exposed faces can only be partially recovered.

## Branch 2: optional reconstruction model (`zero_dce.py`, `reconstruction_model.py`)

[Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension) (about 10k parameters)
predicts per-pixel curves and applies them eight times. It runs on the original crop, and
its output is compared with branch 1 and with the original by the same quality gate.

**Pretrained, no training needed.** Download `Epoch99.pth` yourself from the official
repository (`snapshots_Zero_DCE++/`) and put it, for example, at
`checkpoints/zero_dce_pp/Epoch99.pth` (`checkpoints/` is git-ignored). This project never
downloads weights.

```bash
export LIGHTING_ENHANCER_ENABLE_RECONSTRUCTION=true
export LIGHTING_ENHANCER_MODEL_ARCH=zero_dce_pp          # or torchscript
export LIGHTING_ENHANCER_CHECKPOINT=checkpoints/zero_dce_pp/Epoch99.pth
export LIGHTING_ENHANCER_MODEL_SCALE=1                   # >1 predicts curves at lower res (faster)
```

`torchscript` accepts any TorchScript module that maps RGB `NCHW` floats in `[0, 1]` to the
same. `LIGHTING_ENHANCER_DEVICE` is `auto` (CUDA/MPS when available, otherwise CPU).

**Optional fine-tuning on face crops.** The losses are zero-reference (spatial consistency,
exposure, colour constancy, illumination smoothness), so no paired ground truth is needed.
`--synthesize` turns normally lit crops into random under- or over-exposures:

```bash
python -m extreme_cond_enhancer.train_reconstruction --data dataset/faces \
    --init checkpoints/zero_dce_pp/Epoch99.pth --synthesize --epochs 20 \
    --out checkpoints/zero_dce_pp_faces --export-torchscript
# writes best.pth / last.pth / history.json / zero_dce_pp.torchscript.pt
```

Omit `--init` to train from scratch. Use `--export-only --init X.pth` to convert a
state dict to TorchScript.

## Configuration

All thresholds and model settings live in `config.py`, each overridable by a
`LIGHTING_ENHANCER_*` environment variable. The module docstring lists every variable
and its default.

## Latency and safety

On an Apple-silicon CPU a 112px dark crop takes about 16 ms (NL-means dominates; set
`LIGHTING_ENHANCER_DENOISE_STRENGTH=0` to skip it). Bright crops take about 1-5 ms. Check
`latency_ms` in the metadata on the target machine, especially when a model is enabled.
Enhanced imagery is used only for emotion and recognition features, never for liveness:
anti-spoofing runs on the original frame first.

## Tests

```bash
python -m pytest tests/extreme_cond_enhancer
```
