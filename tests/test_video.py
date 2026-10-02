import cv2
import numpy as np

from targetcapture.video import frame_signature, read_frame, seeks_exactly


def _clip(path, n=60):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 90))
    frames = []
    for i in range(n):
        f = np.zeros((90, 160, 3), np.uint8)
        cv2.putText(f, str(i), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        f[:, (i * 7) % 160] = 255
        w.write(f)
        frames.append(f)
    w.release()
    return frames


def test_read_frame_finds_exact_frame_by_signature(tmp_path):
    p = tmp_path / "c.avi"
    _clip(p)
    want = read_frame(str(p), 30)
    # pretend the index is 3 frames off, as on DVR files that seek badly
    got = read_frame(str(p), 27, frame_signature(want))
    assert got is not None and np.abs(got.astype(int) - want.astype(int)).mean() < 0.5
    assert seeks_exactly(str(p))


def test_read_frame_returns_none_when_not_found(tmp_path):
    p = tmp_path / "c.avi"
    _clip(p)
    sig = np.full((27, 48), 200, np.uint8)  # a picture that is not in the video
    assert read_frame(str(p), 10, sig) is None
