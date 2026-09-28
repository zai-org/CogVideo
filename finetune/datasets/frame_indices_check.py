import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_utils():
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    tv = types.ModuleType("torchvision")
    tvf = types.ModuleType("torchvision.transforms")
    functional = types.ModuleType("torchvision.transforms.functional")
    functional.resize = lambda *args, **kwargs: None
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tvf
    sys.modules["torchvision.transforms.functional"] = functional
    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2
    decord = types.ModuleType("decord")
    decord.bridge = types.SimpleNamespace(set_bridge=lambda *args, **kwargs: None)
    decord.VideoReader = object
    sys.modules["decord"] = decord

    path = Path(__file__).resolve().parent / "utils.py"
    spec = importlib.util.spec_from_file_location("cogvideo_dataset_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrameIndexTest(unittest.TestCase):
    def test_a_short_video_keeps_the_last_frame(self):
        utils = load_utils()
        self.assertEqual(utils.sample_frame_indices(5, 3), [0, 2, 4])
        self.assertEqual(utils.sample_frame_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(utils.sample_frame_indices(5, 5), list(range(5)))
        self.assertEqual(utils.sample_frame_indices(8, 1), [0])


if __name__ == "__main__":
    unittest.main()
