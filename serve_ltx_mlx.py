"""
LTX-Video MLX inference server for Muse.

Pure MLX — no PyTorch, diffusers, or transformers required.
Drop-in replacement for serve_ltx.py with the same API.

Usage:
    python serve_ltx_mlx.py                       # 2B on port 5087
    python serve_ltx_mlx.py --model 13b           # 13B on port 5088
    python serve_ltx_mlx.py --model 13b --port 5088

Endpoints (same as serve_ltx.py):
    POST /generate   — Start async video generation
    GET  /progress   — Fetch generation progress
    GET  /job/<id>   — Check job status
    GET  /download/<id> — Download completed video
    GET  /health     — Server health check
"""

import argparse
import gc
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import imageio
import mlx.core as mx
import numpy as np
from flask import Flask, jsonify, request, send_file

# Use local ltx-mlx package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ltx-mlx"))

from ltx_mlx.pipeline import LTXPipeline

app = Flask(__name__)

# Globals
pipe: LTXPipeline = None
MODEL_VARIANT = "2b"
OUTPUT_DIR = tempfile.mkdtemp(prefix="ltx_mlx_")

# Progress tracking (compatible with serve_ltx.py)
_progress = {"step": 0, "total": 0, "running": False}

# Job tracking
_jobs = {}
_gen_lock = threading.Lock()
_MAX_JOBS = 50


def _prune_jobs():
    if len(_jobs) <= _MAX_JOBS:
        return
    completed = [(k, v) for k, v in _jobs.items() if v.get("status") != "running"]
    to_remove = len(_jobs) - _MAX_JOBS
    for k, _ in completed[:to_remove]:
        path = _jobs[k].get("path")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        del _jobs[k]


def generate_video(prompt, negative_prompt, height, width, num_frames, seed, image=None):
    """Generate video and return path to .mp4 file."""
    # Set up progress tracking
    n_steps = 8 if pipe._is_13b else 4
    _progress["step"] = 0
    _progress["total"] = n_steps
    _progress["running"] = True

    try:
        frames = pipe.generate(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            image=image,
        )
    finally:
        _progress["running"] = False
        _progress["step"] = 0
        _progress["total"] = 0

    # Write video
    out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    writer = imageio.get_writer(
        out_path, fps=24, codec="libx264",
        output_params=["-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return out_path


def _run_generation(job_id, prompt, negative_prompt, height, width, num_frames, seed, image=None):
    """Background thread worker."""
    with _gen_lock:
        try:
            t0 = time.time()
            path = generate_video(prompt, negative_prompt, height, width, num_frames, seed, image=image)
            elapsed = time.time() - t0
            vid_id = Path(path).stem
            _jobs[job_id] = {
                "status": "done",
                "id": vid_id,
                "elapsed": round(elapsed, 1),
                "path": path,
            }
            print(f"[generate] job {job_id} done in {elapsed:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _progress["running"] = False
            _progress["step"] = 0
            _progress["total"] = 0
            _jobs[job_id] = {"status": "error", "error": str(e)}
        finally:
            gc.collect()
            mx.clear_cache()
            _prune_jobs()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": f"ltx-video-mlx-{MODEL_VARIANT}"})


@app.route("/progress", methods=["GET"])
def progress():
    return jsonify(_progress)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    height = data.get("height", 480)
    width = data.get("width", 768)
    num_frames = data.get("num_frames", 97)
    seed = data.get("seed", 42)

    # Handle optional base64-encoded image for image-to-video
    image = None
    image_b64 = data.get("image")
    if image_b64:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        img_bytes = base64.b64decode(image_b64)
        pil_img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        image = np.array(pil_img)
        print(f"[generate] image-to-video: {image.shape[1]}x{image.shape[0]}")

    print(f"[generate] prompt: {prompt[:100]!r}")
    print(f"[generate] {width}x{height}, {num_frames} frames, seed={seed}")

    if height % 32 != 0 or width % 32 != 0:
        return jsonify({"error": "height and width must be divisible by 32"}), 400
    if (num_frames - 1) % 8 != 0:
        return jsonify({"error": "num_frames must be (N*8)+1, e.g. 33, 49, 65, 81, 97"}), 400

    if _progress["running"]:
        return jsonify({"error": "A generation is already in progress"}), 429

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running"}
    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, prompt, negative_prompt, height, width, num_frames, seed),
        kwargs={"image": image},
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/download/<vid_id>", methods=["GET"])
def download(vid_id):
    path = os.path.join(OUTPUT_DIR, f"{vid_id}.mp4")
    if not os.path.exists(path):
        return jsonify({"error": "video not found"}), 404
    response = send_file(path, mimetype="video/mp4")
    try:
        os.remove(path)
    except OSError:
        pass
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-Video MLX inference server")
    parser.add_argument("--model", default="2b", choices=["13b", "2b"],
                        help="Model variant (default: 2b)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: 5087 for 2B, 5088 for 13B)")
    parser.add_argument("--model-dir", default="models/LTX",
                        help="Path to model directory")
    args = parser.parse_args()

    MODEL_VARIANT = args.model
    port = args.port or (5088 if args.model == "13b" else 5087)

    # Resolve checkpoint
    checkpoint = None
    if args.model == "13b":
        checkpoint = "LTX 13B/ltxv-13b-0.9.8-distilled.safetensors"

    pipe = LTXPipeline(args.model_dir, checkpoint=checkpoint)

    print(f"\nServer ready on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
