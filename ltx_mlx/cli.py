"""LTX-Video MLX CLI.

Usage:
    # 2B model text-to-video (fast, ~7s at 480p)
    ltx-mlx --prompt "a cat sitting on a windowsill, golden hour"

    # Image-to-video
    ltx-mlx --prompt "the cat turns its head" --image cat.png

    # 13B model (sharp, ~60s at 720p)
    ltx-mlx --prompt "a portrait, cinematic 4k" --model 13b --height 720 --width 1280

    # Custom settings
    ltx-mlx --prompt "ocean waves" --steps 8 --seed 123 --frames 41 --output ocean.mp4
"""

import argparse


# Resolution presets
PRESETS = {
    "480p": (480, 768),
    "544p": (544, 960),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


def main():
    parser = argparse.ArgumentParser(
        description="LTX-Video generation on Apple Silicon (pure MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ltx-mlx --prompt "a cat on a windowsill"
  ltx-mlx --prompt "portrait, cinematic" --model 13b --resolution 720p
  ltx-mlx --prompt "ocean waves" --height 544 --width 960 --steps 8 --seed 7
        """,
    )

    parser.add_argument("--prompt", type=str, required=True,
                        help="Text description of desired video")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to conditioning image for image-to-video")
    parser.add_argument("--model", type=str, default="2b", choices=["2b", "13b"],
                        help="Model size (default: 2b)")
    parser.add_argument("--model-dir", type=str, default="models/LTX",
                        help="Path to model directory")
    parser.add_argument("--resolution", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Resolution preset (overrides --height/--width)")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height, must be divisible by 32 (default: 480)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width, must be divisible by 32 (default: 768)")
    parser.add_argument("--frames", type=int, default=97,
                        help="Number of frames (default: 97 = ~4s at 24fps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Denoising steps (default: 4 for 2B, 8 for 13B)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path (default: output_ltx_<res>.mp4)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output framerate (default: 24)")

    args = parser.parse_args()

    # Resolve resolution
    if args.resolution:
        height, width = PRESETS[args.resolution]
    else:
        height = args.height or 480
        width = args.width or 768

    # Resolve checkpoint
    checkpoint = None
    if args.model == "13b":
        checkpoint = "LTX 13B/ltxv-13b-0.9.8-distilled.safetensors"

    # Load conditioning image if provided
    image = None
    if args.image:
        from PIL import Image
        import numpy as np
        image = np.array(Image.open(args.image).convert("RGB"))
        print(f"Conditioning image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    from ltx_mlx.pipeline import LTXPipeline

    pipe = LTXPipeline(args.model_dir, checkpoint=checkpoint)

    frames = pipe.generate(
        prompt=args.prompt,
        num_frames=args.frames,
        height=height,
        width=width,
        num_steps=args.steps,
        seed=args.seed,
        image=image,
    )

    out_path = args.output or f"output_ltx_{height}x{width}.mp4"
    import imageio
    imageio.mimwrite(out_path, frames, fps=args.fps, codec="libx264")
    print(f"\nSaved to {out_path} ({len(frames)} frames, {height}x{width} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
