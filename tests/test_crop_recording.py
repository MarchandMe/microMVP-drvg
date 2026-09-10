"""Crop only the requested video borders; keep the source untouched."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("crop_recording",Path(__file__).parents[1]/"examples/crop_recording.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize("backend", ["opencv", "ffmpeg"])
def test_crop_silent_video_dimensions_and_no_overwrite(tmp_path, backend):
    if backend == "ffmpeg" and module.ffmpeg_executable() is None:
        pytest.skip("Optional FFmpeg backend is not installed")
    source=tmp_path/"input.mp4";output=tmp_path/"output.mp4"
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),30,(64,48))
    assert writer.isOpened()
    for value in (60, 120, 180):
        frame = np.full((48,64,3), value, dtype=np.uint8)
        frame[:, :4] = 0
        frame[:, -6:] = 0
        writer.write(frame)
    writer.release()
    original=source.read_bytes()
    count,width,height=module.crop_recording(source,output,left=4,right=6,top=2,bottom=4,backend=backend,threads=2)
    assert (count,width,height)==(3,54,42)
    cap=cv2.VideoCapture(str(output))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (42,54)
        values.append(float(frame[20, 20].mean()))
    cap.release()
    # Every frame is preserved, in order, with the requested borders removed.
    assert values == pytest.approx([60, 120, 180], abs=8)
    assert source.read_bytes()==original
    with pytest.raises(FileExistsError):
        module.crop_recording(source,output,left=4)



def test_unavailable_ffmpeg_falls_back_or_reports_explicit_request(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ffmpeg_executable", lambda: None)
    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    assert module.crop_recording(source, tmp_path / "auto.mp4", left=4)[0] == 1
    with pytest.raises(RuntimeError, match="FFmpeg unavailable"):
        module.crop_recording(source, tmp_path / "required.mp4", left=4, backend="ffmpeg")


def test_failed_ffmpeg_does_not_publish_partial_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="encoder failed"),
    )
    output = tmp_path / "output.mp4"
    with pytest.raises(RuntimeError, match="encoder failed"):
        module._crop_ffmpeg("ffmpeg", tmp_path / "input.mp4", output, 4, 0, 60, 48, 2)
    assert not output.exists()
    assert not list(tmp_path.glob(".crop-*"))
