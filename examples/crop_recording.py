"""Crop borders from a silent demo recording without changing workspace settings."""
import argparse
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2


def ffmpeg_executable():
    """Use a system FFmpeg or the optional bundled binary."""
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _crop_ffmpeg(binary, source, output, left, top, width, height, threads):
    # Decode/crop/encode directly in FFmpeg; avoid Python frame transfers and
    # RGB conversion. Decoder and H.264 encoder both use the requested threads.
    # Publish only a completed file, atomically, without overwriting a target.
    with tempfile.TemporaryDirectory(prefix=".crop-", dir=output.parent) as directory:
        temporary = Path(directory) / "cropped.mp4"
        command = [
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-threads", str(threads), "-i", str(source.resolve()),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"crop={width}:{height}:{left}:{top}:exact=1",
            "-filter_threads", "1", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-threads", str(threads), "-fps_mode", "passthrough",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
            str(temporary),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.strip()}")
        frame_counts = [
            int(line.split("=", 1)[1])
            for line in result.stdout.splitlines() if line.startswith("frame=")
        ]
        frames = frame_counts[-1] if frame_counts else 0
        if frames <= 0:
            raise ValueError("Input contains no readable frames")
        os.link(temporary, output)
    return frames, width, height


def crop_recording(
    source, output, *, left=0, right=0, top=0, bottom=0,
    backend="auto", threads=8,
):
    source, output = Path(source), Path(output)
    if any(value < 0 for value in (left,right,top,bottom)):
        raise ValueError("Crop amounts must be non-negative")
    if not any((left,right,top,bottom)):
        raise ValueError("Specify at least one positive crop amount")
    if output.exists():
        raise FileExistsError(output)
    if backend not in {"auto", "ffmpeg", "opencv"}:
        raise ValueError("Backend must be auto, ffmpeg, or opencv")
    if threads < 1:
        raise ValueError("Threads must be positive")
    binary = ffmpeg_executable() if backend != "opencv" else None
    if backend == "ffmpeg" and binary is None:
        raise RuntimeError(
            "FFmpeg unavailable; install the video extra: pip install -e '.[video]'"
        )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    writer = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        out_width, out_height = width-left-right, height-top-bottom
        if out_width < 2 or out_height < 2:
            raise ValueError("Crop removes the entire image")
        if out_width % 2 or out_height % 2:
            raise ValueError("Choose crop amounts that leave an even width and height")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input has no valid frame rate")
        if output.suffix.lower() != ".mp4":
            raise ValueError("Output must be an .mp4 file")
        output.parent.mkdir(parents=True,exist_ok=True)
        if binary is not None:
            capture.release()
            return _crop_ffmpeg(
                binary, source, output, left, top, out_width, out_height, threads
            )
        output.open("xb").close()
        writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(out_width,out_height))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 encoder")
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame[top:height-bottom,left:width-right].copy())
            frames += 1
        if frames == 0:
            raise ValueError("Input contains no readable frames")
        return frames,out_width,out_height
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    for side in ("left","right","top","bottom"):
        parser.add_argument(f"--{side}",type=int,default=0,help=f"Pixels to remove from {side}")
    parser.add_argument(
        "--threads", type=int, default=min(8, os.cpu_count() or 1),
        help="FFmpeg decoder/encoder threads (default: up to 8)",
    )
    parser.add_argument(
        "--backend", choices=("auto", "ffmpeg", "opencv"), default="auto",
        help="Auto uses multithreaded FFmpeg when available, otherwise OpenCV",
    )
    args=parser.parse_args()
    started = time.perf_counter()
    try:
        frames,width,height=crop_recording(
            args.input,args.output,left=args.left,right=args.right,top=args.top,bottom=args.bottom,
            backend=args.backend,threads=args.threads)
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(1,f"Crop failed: {error}\n")
    elapsed = time.perf_counter() - started
    backend = "FFmpeg" if args.backend != "opencv" and ffmpeg_executable() else "OpenCV"
    print(f"Saved {frames} frames at {width} x {height}: {args.output}")
    print(f"{backend}: {elapsed:.2f}s ({frames / elapsed:.0f} frames/s)")


if __name__=="__main__":
    main()
