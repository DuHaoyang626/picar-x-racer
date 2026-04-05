import unittest
from typing import cast
from unittest.mock import patch

from app.exceptions.camera import CameraDeviceError
from app.schemas.camera import CameraSettings, DiscreteDevice
from app.services.camera.avfoundation_service import AVFoundationService


class FakeFrameRateRange:
    def __init__(self, min_fps: float, max_fps: float):
        self._min_fps = min_fps
        self._max_fps = max_fps

    def minFrameRate(self) -> float:
        return self._min_fps

    def maxFrameRate(self) -> float:
        return self._max_fps


class FakeFormatDescription:
    def __init__(self, width: int, height: int, subtype: int):
        self.width = width
        self.height = height
        self.subtype = subtype


class FakeFormat:
    def __init__(self, width: int, height: int, subtype: int, ranges):
        self._description = FakeFormatDescription(width, height, subtype)
        self._ranges = ranges

    def formatDescription(self) -> FakeFormatDescription:
        return self._description

    def videoSupportedFrameRateRanges(self):
        return self._ranges


class FakeDevice:
    def __init__(self, name: str, unique_id: str, formats):
        self._name = name
        self._unique_id = unique_id
        self._formats = formats

    def localizedName(self) -> str:
        return self._name

    def uniqueID(self) -> str:
        return self._unique_id

    def formats(self):
        return self._formats


class FakeCoreMedia:
    @staticmethod
    def CMVideoFormatDescriptionGetDimensions(description: FakeFormatDescription):
        return description

    @staticmethod
    def CMFormatDescriptionGetMediaSubType(description: FakeFormatDescription) -> int:
        return description.subtype


class TestAVFoundationService(unittest.TestCase):
    def setUp(self):
        self.service = AVFoundationService()
        self.device = FakeDevice(
            "MacBook Pro Camera",
            "camera-1",
            [
                FakeFormat(
                    640,
                    480,
                    0x34323076,
                    [FakeFrameRateRange(15.0, 30.0)],
                ),
                FakeFormat(
                    1280,
                    720,
                    0x34323076,
                    [FakeFrameRateRange(15.0, 30.0)],
                ),
            ],
        )

    def test_list_video_devices_returns_empty_without_frameworks(self):
        with patch.object(AVFoundationService, "_load_frameworks", return_value=None):
            self.assertEqual(self.service.list_video_devices(), [])

    def test_list_video_devices_returns_discrete_profiles(self):
        with patch.object(
            AVFoundationService,
            "_load_frameworks",
            return_value=(object(), FakeCoreMedia),
        ), patch.object(self.service, "_discover_devices", return_value=[self.device]):
            devices = self.service.list_video_devices()

        self.assertEqual(len(devices), 2)
        self.assertIsInstance(devices[0], DiscreteDevice)
        first_device = cast(DiscreteDevice, devices[0])
        self.assertEqual(first_device.device, "avfoundation:/camera-1")
        self.assertEqual(first_device.path, "/camera-1")
        self.assertEqual(first_device.pixel_format, "420v")
        self.assertEqual(first_device.width, 640)
        self.assertEqual(first_device.height, 480)
        self.assertEqual(first_device.fps, 30.0)

    def test_resolve_capture_config_snaps_to_closest_supported_mode(self):
        with patch.object(
            AVFoundationService,
            "_load_frameworks",
            return_value=(object(), FakeCoreMedia),
        ), patch.object(self.service, "_discover_devices", return_value=[self.device]):
            input_name, settings = self.service.resolve_capture_config(
                "avfoundation:/camera-1",
                CameraSettings(
                    device="avfoundation:/camera-1",
                    width=1280,
                    height=720,
                    fps=24.0,
                ),
            )

        self.assertEqual(input_name, "MacBook Pro Camera")
        self.assertEqual(settings.device, "avfoundation:/camera-1")
        self.assertEqual(settings.width, 1280)
        self.assertEqual(settings.height, 720)
        self.assertEqual(settings.fps, 30.0)
        self.assertFalse(settings.use_gstreamer)

    def test_resolve_capture_config_raises_for_unsupported_mode(self):
        with patch.object(
            AVFoundationService,
            "_load_frameworks",
            return_value=(object(), FakeCoreMedia),
        ), patch.object(self.service, "_discover_devices", return_value=[self.device]):
            with self.assertRaises(CameraDeviceError):
                self.service.resolve_capture_config(
                    "avfoundation:/camera-1",
                    CameraSettings(
                        device="avfoundation:/camera-1",
                        width=800,
                        height=600,
                        fps=30.0,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
