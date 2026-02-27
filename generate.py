#!/usr/bin/env python3
"""Generate music with MusicGen on Apple Silicon (MLX).

Usage:
    musicgen-mlx "a gentle piano melody with soft strings"
    musicgen-mlx "deep house track with hypnotic groove" --duration 15
    musicgen-mlx "epic orchestral" --model facebook/musicgen-medium -o epic.wav
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="musicgen-mlx",
        description="Generate music from text descriptions using MusicGen "
                    "on Apple Silicon with MLX.",
        epilog="Examples:\n"
               "  musicgen-mlx \"a chill lo-fi beat\"\n"
               "  musicgen-mlx \"epic orchestral soundtrack\" --duration 15\n"
               "  musicgen-mlx \"deep house groove\" -m facebook/musicgen-medium\n"
               "  musicgen-mlx \"jazz piano trio\" --temperature 0.8 --top-k 500\n"
               "  musicgen-mlx \"ambient pad\" -m facebook/musicgen-stereo-large -o ambient.wav\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("prompt",
                        help="text description of the music to generate")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("-m", "--model", default="facebook/musicgen-small",
                             metavar="NAME",
                             help="HuggingFace model name (default: facebook/musicgen-small)")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("-o", "--output", default=None,
                              metavar="FILE",
                              help="output WAV path (default: ./musicgen_output.wav)")
    output_group.add_argument("-d", "--duration", type=float, default=8.0,
                              metavar="SEC",
                              help="duration in seconds (default: 8)")
    output_group.add_argument("--no-open", action="store_true",
                              help="don't open the file after generation")

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument("--top-k", type=int, default=250,
                                metavar="K",
                                help="top-k sampling (default: 250)")
    sampling_group.add_argument("--temperature", type=float, default=1.0,
                                metavar="T",
                                help="sampling temperature (default: 1.0)")
    sampling_group.add_argument("--cfg-coef", type=float, default=3.0,
                                metavar="C",
                                help="classifier-free guidance coefficient (default: 3.0)")

    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from audiocraft_mlx.models.musicgen import MusicGen

    # Load model
    print(f"Loading {args.model}...")
    t0 = time.time()
    mg = MusicGen.get_pretrained(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s "
          f"(sr={mg.sample_rate}, channels={mg.audio_channels})")

    mg.set_generation_params(
        use_sampling=True,
        top_k=args.top_k,
        temperature=args.temperature,
        duration=args.duration,
        cfg_coef=args.cfg_coef,
    )

    # Generate
    print(f"Generating {args.duration}s: \"{args.prompt}\"")
    t0 = time.time()
    audio = mg.generate([args.prompt], progress=True)
    gen_time = time.time() - t0
    print(f"\nGenerated in {gen_time:.1f}s "
          f"(ratio: {args.duration / gen_time:.2f}x realtime)")

    # Save
    wav = np.array(audio[0])  # [C, T]
    output = args.output or "musicgen_output.wav"
    sf.write(output, wav.T, mg.sample_rate)

    channels = "stereo" if wav.shape[0] == 2 else "mono"
    print(f"Saved: {output} ({channels}, {wav.shape[1]/mg.sample_rate:.1f}s)")

    # Open
    if not args.no_open and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", output])


if __name__ == "__main__":
    main()
