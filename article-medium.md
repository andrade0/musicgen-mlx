# I Ported MusicGen to Apple Silicon — Generate Music from Text on Your MacBook

Meta's MusicGen is one of the most impressive text-to-music models available. Type "a chill lo-fi beat with vinyl crackle and soft piano," and it generates a surprisingly good audio clip. The problem? Running it on a Mac means PyTorch on CPU, which is painfully slow.

I ported MusicGen to Apple's [MLX](https://github.com/ml-explore/mlx) framework. The small model (300M parameters) now generates 8 seconds of audio in about 6 seconds on an M4 Max — faster than realtime. The large stereo model (3.3B parameters) works too, producing high-quality stereo music.

The project is open source and ready to try: [github.com/andrade0/musicgen-mlx](https://github.com/andrade0/musicgen-mlx)

> **Heads up**: This is an early release. It works, but there's plenty of room for improvement. If you're interested in contributing, read on — I'll explain what's done and what's left.

---

## Getting started

You need a Mac with Apple Silicon (M1/M2/M3/M4) and Python 3.10+. I recommend using a dedicated conda environment:

```bash
# Create a clean environment
conda create -n musicgen python=3.11 -y
conda activate musicgen

# Clone and install
git clone https://github.com/andrade0/musicgen-mlx.git
cd musicgen-mlx
make install
```

That's it. Now generate music:

```bash
musicgen-mlx "deep house track with a hypnotic bassline"
```

The model downloads from HuggingFace on first run (~1.2 GB for the small model), and the generated audio opens automatically.

## Choosing a model

MusicGen comes in several sizes. All are available through a simple `-m` flag:

| Model | Parameters | Size | Quality | Speed (M4 Max) |
|:---:|:---:|:---:|:---:|:---:|
| `facebook/musicgen-small` | 300M | ~1.2 GB | Good | 1.3x realtime |
| `facebook/musicgen-medium` | 1.5B | ~3.2 GB | Better | ~0.6x realtime |
| `facebook/musicgen-large` | 3.3B | ~6.5 GB | Best | ~0.3x realtime |
| `facebook/musicgen-stereo-small` | 300M | ~1.2 GB | Good (stereo) | ~1.2x realtime |
| `facebook/musicgen-stereo-medium` | 1.5B | ~3.2 GB | Better (stereo) | ~0.5x realtime |
| `facebook/musicgen-stereo-large` | 3.3B | ~6.5 GB | Best (stereo) | ~0.3x realtime |

The **small** model is the default — fast enough for interactive use. If you want the best quality and don't mind waiting, go for **large** or **stereo-large**. The stereo variants produce left/right channel audio instead of mono, which sounds more natural on headphones.

Two variants are not yet supported: `musicgen-melody` (needs a Demucs-based conditioner to accept a reference melody) and `musicgen-style` (needs an EnCodec-based style conditioner). Both are on the roadmap.

```bash
# Quick test with the default small model
musicgen-mlx "funky disco groove with slap bass"

# Better quality with the medium model
musicgen-mlx "funky disco groove with slap bass" -m facebook/musicgen-medium

# Best quality, mono
musicgen-mlx "cinematic orchestral theme" -m facebook/musicgen-large -d 20

# Stereo output (sounds great on headphones)
musicgen-mlx "ambient pad with wide reverb" -m facebook/musicgen-stereo-small

# Best of everything: stereo + large + 30 seconds
musicgen-mlx "epic orchestral soundtrack with dramatic strings" \
    -m facebook/musicgen-stereo-large -d 30
```

Models are cached locally after the first download (~/.cache/huggingface/), so subsequent runs start instantly.

You can also tweak the sampling parameters — higher temperature means more creative/unpredictable output, lower means more conservative:

```bash
# More experimental output
musicgen-mlx "jazz piano trio improvising" --temperature 1.2 --top-k 500

# More focused, predictable output
musicgen-mlx "classical piano sonata" --temperature 0.8 --top-k 100
```

### Use from Python

```python
from audiocraft_mlx.models.musicgen import MusicGen

mg = MusicGen.get_pretrained("facebook/musicgen-small")
mg.set_generation_params(duration=8.0)
audio = mg.generate(["jazz piano trio improvising"])
```

## How MusicGen works

MusicGen is fundamentally different from diffusion-based music models. It's an autoregressive transformer — similar in spirit to GPT, but for audio instead of text.

The pipeline has three stages:

1. **Text encoding**: Your prompt goes through a T5 encoder, producing a sequence of text embeddings
2. **Token generation**: A large transformer (up to 48 layers for the 3.3B model) generates discrete audio tokens one by one, conditioned on the text embeddings. It uses classifier-free guidance to stay faithful to your prompt.
3. **Audio decoding**: Meta's EnCodec converts the discrete tokens back into a continuous waveform

The clever part is the codebook interleaving. EnCodec produces 4 parallel streams of tokens (codebooks). Rather than generating all 4 independently, MusicGen interleaves them in a delayed pattern — codebook 0 leads, then codebook 1 starts one step later, and so on. This captures the dependencies between codebooks while staying efficient.

## What I ported (and what I didn't)

The full inference pipeline runs on MLX:

- **Transformer language model** — 48 layers of streaming multi-head attention with RoPE, KV caching, and cross-attention for text conditioning
- **EnCodec decoder** — SEANet architecture with residual vector quantization, LSTM, and transposed convolutions
- **Weight conversion** — automatic download of Meta's checkpoints with on-the-fly conversion (Conv weight transposition, MultiheadAttention split, weight norm folding, LSTM reorganization)

What still runs in PyTorch:
- **T5 text encoder** — porting the entire T5 stack to MLX would be a project in itself. For now, it runs in PyTorch on CPU and outputs are converted to `mx.array`. This is pragmatic but adds startup overhead.
- **Stereo EnCodec** — stereo models use HuggingFace's EnCodec, wrapped with MLX↔torch conversion at the boundaries.

## The interesting challenges

### Channels-last everything

MLX convolutions expect channels-last layout (`[B, T, C]`), while PyTorch uses channels-first (`[B, C, T]`). Rather than changing the entire model's convention, I kept `[B, C, T]` as the "public" format and transpose at every conv boundary. On MLX, transposes are free — they're view operations on unified memory.

### No complex tensors

MLX doesn't support complex numbers. The original MusicGen uses complex exponentials for RoPE. My port uses paired sin/cos instead — mathematically equivalent, just a different representation.

### Weight norm folding

PyTorch's `weight_norm` stores two tensors (`weight_g` for magnitude, `weight_v` for direction) and computes the actual weight at runtime. Since we're inference-only, I fold them at load time: `weight = weight_g * weight_v / ||weight_v||`. One less thing to compute per forward pass.

### The underscore gotcha

MLX's `nn.Module` ignores attributes that start with `_` — it won't track them as parameters. The original code stores the HuggingFace EnCodec model as `self._hf_model`. In PyTorch, that's fine. In MLX, the model silently has zero parameters. This one took a while to find.

## Performance

On Apple M4 Max with MLX 0.21:

| Model | Parameters | 8s audio | Speed |
|:---:|:---:|:---:|:---:|
| musicgen-small | 300M | 6.3s | 1.3x realtime |
| musicgen-stereo-large | 3.3B | ~24s | 0.3x realtime |

MusicGen is autoregressive — it generates tokens one at a time, so it doesn't parallelize as well as encoder-only models. The small model is already faster than realtime, which is practical for interactive use. The large model is slower but produces noticeably better music.

For context, the original PyTorch on CPU is significantly slower on the same machine.

## What's not done yet — contributions welcome

This is an early release. Here's what could be improved:

**High impact:**
- **Port T5 to MLX** — Remove the PyTorch dependency for text encoding. This would cut startup time and memory significantly.
- **Extended generation** — Support for generating beyond 30 seconds with a sliding window approach.
- **KV cache optimization** — The cache works but isn't optimized for memory reuse across generation steps.

**Medium impact:**
- **Melody conditioning** — The `musicgen-melody` variant needs a ChromaStemConditioner (based on Demucs) to condition on an input melody. The pieces exist but aren't wired together yet.
- **Style conditioning** — The `musicgen-style` variant needs a StyleConditioner based on EnCodec feature extraction.
- **Batch generation** — Generate multiple prompts in parallel.

**Nice to have:**
- **Metal kernels** — Custom Metal shaders for the attention mechanism could speed things up.
- **Quantized models** — 4-bit or 8-bit quantization for the transformer weights to reduce memory.
- **Audio-to-audio** — Condition on an input audio clip (not just text).

If any of this interests you, check out the repo: [github.com/andrade0/musicgen-mlx](https://github.com/andrade0/musicgen-mlx)

## This is part of a bigger project

I've also ported **Demucs** (Meta's music source separation model) to MLX: [github.com/andrade0/demucs-mlx](https://github.com/andrade0/demucs-mlx). That one separates any song into drums, bass, other, and vocals at 34x realtime — a 7-minute track in 12 seconds.

Together, these two tools let you do interesting things entirely on your Mac:
- Generate music from text, then separate the stems
- Isolate vocals from a song, then generate a new instrumental to go under them
- Sample specific elements from generated music

Both projects are MIT licensed. Apple Silicon is turning into a surprisingly capable platform for audio ML.

---

*MusicGen MLX is MIT licensed. The original MusicGen model and pretrained weights are by Meta Research. MLX is by Apple. MusicGen paper: [Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Defossez (2023)](https://arxiv.org/abs/2306.05284).*
