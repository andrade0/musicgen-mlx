# MusicGen MLX

Text-to-music generation on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A clean MLX port of Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) for **inference-only** on Mac M1/M2/M3/M4. Describe the music you want in plain English, get a WAV file back.

<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple" alt="Apple Silicon MLX">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
</p>

> **Early release** — This is a working port but still rough around the edges. Contributions welcome!

## What it does

```bash
musicgen-mlx "a chill lo-fi beat with vinyl crackle and soft piano"
```

MusicGen turns text descriptions into music. This port runs entirely on your Mac's GPU through MLX instead of crawling along on CPU through PyTorch.

## Performance

Benchmarked on Apple M4 Max with MLX 0.21:

| Model | Parameters | 8s audio | Speed |
|:---:|:---:|:---:|:---:|
| musicgen-small | 300M | 6.3s | **1.3x** realtime |
| musicgen-stereo-large | 3.3B | ~24s | 0.3x realtime |

MusicGen is an autoregressive model — it generates audio token by token, so it doesn't parallelize as easily as Demucs. The small model is already faster than realtime. The large model is slower but produces significantly better quality.

## Quick start

```bash
git clone https://github.com/andrade0/musicgen-mlx.git
cd musicgen-mlx
make install
```

Now generate music from anywhere on your Mac:

```bash
musicgen-mlx "a gentle piano melody with soft strings"
```

The first run downloads the model from HuggingFace (~1.2 GB for small, ~6.5 GB for stereo-large). Generated audio opens automatically on macOS.

> If you see a PATH warning during install, add this to your `~/.zshrc`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

### Usage

```
$ musicgen-mlx --help

usage: musicgen-mlx [-h] [-m NAME] [-o FILE] [-d SEC] [--no-open]
                    [--top-k K] [--temperature T] [--cfg-coef C]
                    prompt

Generate music from text descriptions using MusicGen
on Apple Silicon with MLX.

positional arguments:
  prompt                text description of the music to generate

model:
  -m, --model NAME      HuggingFace model name (default: facebook/musicgen-small)

output:
  -o, --output FILE     output WAV path (default: ./musicgen_output.wav)
  -d, --duration SEC    duration in seconds (default: 8)
  --no-open             don't open the file after generation

sampling:
  --top-k K             top-k sampling (default: 250)
  --temperature T       sampling temperature (default: 1.0)
  --cfg-coef C          classifier-free guidance coefficient (default: 3.0)
```

### Examples

```bash
# Quick generation with the small model
musicgen-mlx "deep house track with a hypnotic bassline"

# 30 seconds with the large stereo model (best quality)
musicgen-mlx "epic orchestral soundtrack" -m facebook/musicgen-stereo-large -d 30

# Jazz with high temperature for more variation
musicgen-mlx "jazz piano trio improvising" --temperature 1.2 --top-k 500

# Save to a specific file
musicgen-mlx "ambient drone with rain sounds" -o ambient.wav -d 20

# Medium model (good quality/speed balance)
musicgen-mlx "funk guitar riff with wah pedal" -m facebook/musicgen-medium
```

### Use from Python

```python
import mlx.core as mx
import numpy as np
import soundfile as sf
from audiocraft_mlx.models.musicgen import MusicGen

# Load model (downloads on first use)
mg = MusicGen.get_pretrained("facebook/musicgen-small")
mg.set_generation_params(duration=8.0, top_k=250, temperature=1.0)

# Generate
audio = mg.generate(["a chill lo-fi beat"], progress=True)

# Save
wav = np.array(audio[0])  # [channels, samples]
sf.write("output.wav", wav.T, mg.sample_rate)
```

## How it works

MusicGen is an autoregressive transformer that generates audio tokens conditioned on text:

```
    Text prompt
         │
         ▼
  ┌─────────────┐
  │  T5 Encoder  │  (runs in PyTorch)
  │  (text→embed) │
  └──────┬───────┘
         │ text embeddings
         ▼
  ┌─────────────────┐
  │   Transformer    │  (runs in MLX)
  │   Language Model │
  │  (48 layers for  │
  │   large model)   │
  └──────┬───────────┘
         │ audio tokens
         ▼
  ┌─────────────────┐
  │  EnCodec Decoder │  (runs in MLX)
  │  (tokens→audio)  │
  └──────┬───────────┘
         │
         ▼
    WAV audio file
```

1. **T5 text encoder** converts your prompt into embeddings (runs in PyTorch as a bridge)
2. **Transformer LM** generates discrete audio tokens autoregressively, conditioned on the text embeddings, with classifier-free guidance
3. **EnCodec decoder** converts the tokens back into a waveform

The transformer uses RoPE positional embeddings, KV caching, and 4 interleaved codebook streams.

### MLX-specific design choices

- **Layout convention**: Tensors use `[B, C, T]` format externally; transpose at every Conv1d boundary since MLX expects channels-last
- **T5 bridge**: The text encoder runs in PyTorch, output converted to `mx.array` (pragmatic — avoids porting the entire T5 stack)
- **Weight loading**: Downloads original HuggingFace checkpoints, converts Conv weights (transpose axes), splits MultiheadAttention projections, folds weight norm — all automatically
- **Stereo models**: HuggingFace EnCodec runs in PyTorch for stereo, wrapped with MLX↔torch conversion at boundaries
- **No training code**: Inference-only, all batch norm / dropout / weight init stripped out

## Models

| Model | Parameters | Channels | Quality | Status |
|-------|-----------|----------|---------|--------|
| `facebook/musicgen-small` | 300M | mono | Good | Working |
| `facebook/musicgen-medium` | 1.5B | mono | Better | Working |
| `facebook/musicgen-large` | 3.3B | mono | Best | Working |
| `facebook/musicgen-stereo-small` | 300M | stereo | Good | Working |
| `facebook/musicgen-stereo-medium` | 1.5B | stereo | Better | Working |
| `facebook/musicgen-stereo-large` | 3.3B | stereo | Best | Working |
| `facebook/musicgen-melody` | 1.5B | mono | — | Not yet (needs ChromaStemConditioner) |
| `facebook/musicgen-style` | 1.5B | mono | — | Not yet (needs StyleConditioner) |

## Known limitations

This is an early release. Things that could be improved:

- **T5 runs in PyTorch** — the text encoder hasn't been ported to MLX yet. This adds startup time and memory overhead.
- **Stereo EnCodec in PyTorch** — stereo models use the HuggingFace EnCodec wrapper which runs in PyTorch.
- **No melody conditioning** — the `musicgen-melody` model requires a ChromaStemConditioner (Demucs-based) which isn't wired up yet.
- **No extended generation** — generating beyond 30s with sliding window isn't implemented yet.
- **No KV cache optimization** — the KV cache works but isn't optimized for memory reuse.
- **Speed** — autoregressive generation is inherently sequential. The small model is faster than realtime, but larger models are slower.

Contributions to address any of these are very welcome!

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- Dependencies: `mlx`, `numpy`, `torch` (CPU), `transformers`, `huggingface-hub`, `omegaconf`, `soundfile`, `tqdm`

> **Note on PyTorch**: PyTorch is used for the T5 text encoder, stereo EnCodec, and initial weight loading. It runs on CPU. A future version may reduce or remove this dependency.

## Project structure

```
musicgen-mlx/
├── Makefile                        # make install / make uninstall
├── generate.py                     # CLI entry point
├── audiocraft_mlx/
│   ├── models/
│   │   ├── musicgen.py             # Main MusicGen API
│   │   ├── lm.py                   # Transformer language model
│   │   ├── encodec.py              # EnCodec audio codec
│   │   ├── genmodel.py             # Base generative model
│   │   ├── loaders.py              # HuggingFace model loading
│   │   └── builders.py             # Model builders from config
│   ├── modules/
│   │   ├── transformer.py          # Streaming transformer + KV cache
│   │   ├── conditioners.py         # T5 text conditioning
│   │   ├── conv.py                 # Custom convolutions
│   │   ├── lstm.py                 # LSTM wrapper
│   │   ├── rope.py                 # Rotary positional embeddings
│   │   ├── seanet.py               # SEANet encoder/decoder
│   │   ├── streaming.py            # Streaming base module
│   │   ├── codebooks_patterns.py   # Codebook interleaving
│   │   └── activations.py          # Activation functions
│   ├── quantization/
│   │   ├── vq.py                   # Residual vector quantization
│   │   └── core_vq.py              # Core VQ logic
│   └── utils/
│       ├── weight_convert.py       # PyTorch → MLX weight conversion
│       ├── audio_utils.py          # Audio utilities
│       ├── sampling.py             # Top-k/top-p sampling
│       └── padding.py              # Padding utilities
```

## License

MIT License — same as the [original AudioCraft](https://github.com/facebookresearch/audiocraft).

## Credits

- [AudioCraft](https://github.com/facebookresearch/audiocraft) by Meta Research — the original PyTorch model and pretrained weights
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework that makes this run on Apple Silicon
- MusicGen paper: [Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Defossez (2023)](https://arxiv.org/abs/2306.05284)
