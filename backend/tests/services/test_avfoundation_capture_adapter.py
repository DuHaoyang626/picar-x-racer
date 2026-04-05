import os
import threading
import unittest
from typing import Optional, cast
from unittest.mock import patch

import numpy as np
from app.adapters.avfoundation_capture_adapter import AVFoundationCaptureAdapter
from app.exceptions.camera import CameraDeviceError
from app.schemas.camera import CameraSettings
from app.services.camera.avfoundation_service import AVFoundationService


class DummyAVFoundationService:
    def resolve_capture_config(self, device: str, camera_settings: CameraSettings):
        return (
            "MacBook Pro Camera",
            CameraSettings(
                **{
                    **camera_settings.model_dump(),
                    "device": device,
                    "width": 2,
                    "height": 1,
                    "fps": 30.0,
                    "use_gstreamer": False,
                }
            ),
        )

    def resolve_capture_index(self, device: str) -> int:
        return 0


class FakeOpenCVCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = list(frames)
        self._index = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self):
        if not self._opened or self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def release(self) -> None:
        self._opened = False


class ReusingOpenCVCapture:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self._opened = True
        self._read_count = 0

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self):
        if not self._opened:
            return False, None
        self._frame[0, 0, 0] = self._read_count
        self._read_count += 1
        return True, self._frame

    def release(self) -> None:
        self._opened = False


class BlockingOpenCVCapture:
    def __init__(self):
        self._opened = True
        self._calls = 0
        self.read_started = threading.Event()
        self.allow_finish = threading.Event()

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self):
        if not self._opened:
            return False, None
        if self._calls == 0:
            self._calls += 1
            return True, np.zeros((1, 2, 3), dtype=np.uint8)
        self.read_started.set()
        self.allow_finish.wait(timeout=2)
        self._calls += 1
        return True, np.ones((1, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        self._opened = False


class FakeProcess:
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes = b""):
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()

        os.write(stdout_write, stdout_bytes)
        os.close(stdout_write)
        os.write(stderr_write, stderr_bytes)
        os.close(stderr_write)

        self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
        self.stderr = os.fdopen(stderr_read, "rb", buffering=0)
        self._returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout: Optional[float] = None) -> int:
        self._returncode = 0
        return 0

    def kill(self) -> None:
        self._returncode = -9
        if not self.stdout.closed:
            self.stdout.close()
        if not self.stderr.closed:
            self.stderr.close()


class BlockingProcess(FakeProcess):
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes = b""):
        super().__init__(stdout_bytes, stderr_bytes)
        self.released = threading.Event()

    def terminate(self) -> None:
        self.released.set()
        super().terminate()


class TestAVFoundationCaptureAdapter(unittest.TestCase):
    def test_adapter_reads_initial_and_subsequent_frames(self):
        frame1 = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
        frame2 = np.array([[[10, 11, 12], [13, 14, 15]]], dtype=np.uint8)
        fake_capture = FakeOpenCVCapture([frame1, frame2, frame2])

        with patch(
            "app.adapters.avfoundation_capture_adapter.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            adapter = AVFoundationCaptureAdapter(
                "avfoundation:/camera-1",
                CameraSettings(device="avfoundation:/camera-1"),
                cast(AVFoundationService, DummyAVFoundationService()),
            )

        ok1, first_frame = adapter.read()
        ok2, second_frame = adapter.read()

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(first_frame.shape, (1, 2, 3))
        self.assertIsInstance(first_frame, np.ndarray)
        self.assertEqual(first_frame.tolist(), [[[0, 1, 2], [3, 4, 5]]])
        self.assertEqual(second_frame.tolist(), [[[10, 11, 12], [13, 14, 15]]])
        self.assertFalse(adapter.settings.use_gstreamer)

        adapter.release()

    def test_adapter_returns_copied_frames_for_opencv_capture(self):
        shared_frame = np.zeros((1, 2, 3), dtype=np.uint8)
        fake_capture = ReusingOpenCVCapture(shared_frame)

        with patch(
            "app.adapters.avfoundation_capture_adapter.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            adapter = AVFoundationCaptureAdapter(
                "avfoundation:/camera-1",
                CameraSettings(device="avfoundation:/camera-1"),
                cast(AVFoundationService, DummyAVFoundationService()),
            )

        ok1, first_frame = adapter.read()
        ok2, second_frame = adapter.read()

        self.assertTrue(ok1)
        self.assertTrue(ok2)
        self.assertEqual(int(first_frame[0, 0, 0]), 0)
        self.assertEqual(int(second_frame[0, 0, 0]), 1)
        self.assertFalse(np.shares_memory(first_frame, shared_frame))
        self.assertFalse(np.shares_memory(second_frame, shared_frame))

        adapter.release()

    def test_release_waits_for_inflight_opencv_read(self):
        fake_capture = BlockingOpenCVCapture()

        with patch(
            "app.adapters.avfoundation_capture_adapter.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            adapter = AVFoundationCaptureAdapter(
                "avfoundation:/camera-1",
                CameraSettings(device="avfoundation:/camera-1"),
                cast(AVFoundationService, DummyAVFoundationService()),
            )

        ok1, _ = adapter.read()
        self.assertTrue(ok1)

        read_result = {}

        def reader():
            read_result["value"] = adapter.read()

        read_thread = threading.Thread(target=reader)
        release_thread = threading.Thread(target=adapter.release)

        read_thread.start()
        self.assertTrue(fake_capture.read_started.wait(timeout=1))

        release_thread.start()
        release_thread.join(timeout=0.2)
        self.assertTrue(release_thread.is_alive())

        fake_capture.allow_finish.set()

        read_thread.join(timeout=1)
        release_thread.join(timeout=1)

        self.assertFalse(read_thread.is_alive())
        self.assertFalse(release_thread.is_alive())
        self.assertTrue(read_result["value"][0])

    def test_adapter_raises_when_ffmpeg_is_missing(self):
        with patch.object(
            AVFoundationCaptureAdapter,
            "_start_opencv_capture",
            side_effect=CameraDeviceError("OpenCV unavailable"),
        ), patch(
            "app.adapters.avfoundation_capture_adapter.shutil.which",
            return_value=None,
        ):
            with self.assertRaises(CameraDeviceError):
                AVFoundationCaptureAdapter(
                    "avfoundation:/camera-1",
                    CameraSettings(device="avfoundation:/camera-1"),
                    cast(AVFoundationService, DummyAVFoundationService()),
                )

    def test_release_during_read_does_not_crash(self):
        frame1 = bytes([0, 1, 2, 3, 4, 5])
        fake_process = BlockingProcess(frame1)

        with patch.object(
            AVFoundationCaptureAdapter,
            "_start_opencv_capture",
            side_effect=CameraDeviceError("OpenCV unavailable"),
        ), patch(
            "app.adapters.avfoundation_capture_adapter.shutil.which",
            return_value="/opt/homebrew/bin/ffmpeg",
        ), patch(
            "app.adapters.avfoundation_capture_adapter.subprocess.Popen",
            return_value=fake_process,
        ):
            adapter = AVFoundationCaptureAdapter(
                "avfoundation:/camera-1",
                CameraSettings(device="avfoundation:/camera-1"),
                cast(AVFoundationService, DummyAVFoundationService()),
            )

        ok1, _ = adapter.read()
        self.assertTrue(ok1)

        def release_during_read(*args, **kwargs):
            adapter.release()
            return None

        with patch.object(
            AVFoundationCaptureAdapter,
            "_read_exact",
            side_effect=release_during_read,
        ):
            ok2, second_frame = adapter.read()

        self.assertFalse(ok2)
        self.assertEqual(second_frame.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
