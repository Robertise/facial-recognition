import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import numpy as np

from extreme_cond_enhancer import LightingEnhancer, LightingEnhancerConfig
from extreme_cond_enhancer.color_scaler import adaptive_gamma, enhance_color_scaling, robust_stretch

try:
    import torch
except ImportError:  # reconstruction tests need torch
    torch = None


def _face_like(level: float) -> np.ndarray:
    """Skin-toned crop with a gradient and a darker "eye band", scaled to ``level``."""
    yy, xx = np.mgrid[0:96, 0:80].astype(np.float32)
    base = 0.6 + 0.4 * (xx / 80.0)
    base[30:40, 15:65] *= 0.4
    skin = np.array([0.55, 0.70, 0.95], np.float32)  # BGR
    return np.clip(255.0 * level * base[..., None] * skin, 0, 255).astype(np.uint8)


class LightingEnhancerTests(unittest.TestCase):
    def setUp(self): self.enhancer = LightingEnhancer()
    def _assert_valid(self, image):
        output, metadata = self.enhancer.enhance(image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertIn("quality_score", metadata)
        return output, metadata
    def test_dark(self):
        _, meta = self._assert_valid(np.full((80, 80, 3), 8, np.uint8))
        self.assertEqual(meta["condition"], "too_dark")
    def test_bright(self):
        _, meta = self._assert_valid(np.full((80, 80, 3), 250, np.uint8))
        self.assertEqual(meta["condition"], "too_bright")
    def test_normal_bypasses(self):
        image = np.full((80, 80, 3), 128, np.uint8)
        output, meta = self._assert_valid(image)
        self.assertFalse(meta["applied"])
        self.assertTrue(np.array_equal(output, image))
    def test_grayscale_like_and_random(self):
        self._assert_valid(np.dstack([np.tile(np.arange(80, dtype=np.uint8), (80, 1))] * 3))
        self._assert_valid(np.random.default_rng(5).integers(0, 256, (80, 80, 3), dtype=np.uint8))
    def test_tiny_and_invalid(self):
        self._assert_valid(np.zeros((4, 4, 3), np.uint8))
        output, meta = self.enhancer.enhance(np.zeros((4, 4), np.uint8))
        self.assertEqual(output.shape, (4, 4))
        self.assertTrue(meta["fallback_used"])
    def test_no_checkpoint_is_safe(self):
        self.assertFalse(self.enhancer.reconstruction_model.is_available())
    def test_dark_face_is_brightened_with_chroma_preserved(self):
        image = _face_like(0.12)
        output, meta = self._assert_valid(image)
        self.assertTrue(meta["applied"])
        self.assertGreater(meta["brightness_after"], meta["brightness_before"] + 25)
        self.assertGreater(meta["transform"]["mean_gain"], 1.0)
        self.assertLessEqual(meta["transform"]["gamma"], 1.0)
        # Gain maps scale B, G and R together, so the channel ordering of the skin tone survives.
        mean_bgr = output.reshape(-1, 3).mean(0)
        self.assertTrue(mean_bgr[2] > mean_bgr[1] > mean_bgr[0])

    def test_bright_face_is_darkened(self):
        output, meta = self._assert_valid(_face_like(1.6))
        self.assertEqual(meta["condition"], "too_bright")
        if meta["applied"]:
            self.assertLess(meta["brightness_after"], meta["brightness_before"])
            self.assertGreaterEqual(meta["transform"]["gamma"], 1.0)

    def test_transform_math(self):
        stretched = robust_stretch(np.array([10.0, 20.0, 30.0], np.float32), 10.0, 30.0)
        np.testing.assert_allclose(stretched, [0.0, 127.5, 255.0])
        flat = np.full(5, 40.0, np.float32)
        np.testing.assert_array_equal(robust_stretch(flat, 40.0, 41.0), flat)  # degenerate range guarded
        values = np.full(9, 0.1 * 255.0, np.float32)
        gamma = adaptive_gamma(values, 0.4, 0.0, 1.0)
        self.assertAlmostEqual(0.1 ** gamma, 0.4, places=4)
        self.assertEqual(adaptive_gamma(values, 0.4, 0.6, 1.0), 0.6)  # clipped to the configured bound
        self.assertEqual(adaptive_gamma(np.zeros(4, np.float32), 0.4, 0.4, 1.0), 0.4)  # all-black is safe

    def test_normal_condition_transform_is_identity(self):
        image = _face_like(0.5)
        output, _ = enhance_color_scaling(image, "normal", LightingEnhancerConfig())
        self.assertTrue(np.array_equal(output, image))

    def test_quality_gate_rejects(self):
        strict = LightingEnhancer(LightingEnhancerConfig(min_quality_gain=10.0))
        image = _face_like(0.12)
        output, meta = strict.enhance(image)
        self.assertTrue(np.array_equal(output, image))
        self.assertEqual(meta["reason"], "quality gate rejected enhancement")
        self.assertTrue(meta["fallback_used"])

    def test_reconstruction_candidate_is_used_when_best(self):
        class Perfect:
            def is_available(self): return True
            def enhance(self, image):
                return np.clip(image.astype(np.float32) * (128.0 / max(image.mean(), 1.0)), 0, 255).astype(np.uint8)
        enhancer = LightingEnhancer(LightingEnhancerConfig(clahe_clip_limit=0, denoise_strength=0), Perfect())
        _, meta = self._assert_valid_with(enhancer, _face_like(0.12))
        self.assertIn(meta["reason"], {"local reconstruction model", "deterministic color scaling"})

    def test_reconstruction_failure_falls_back(self):
        class Broken:
            def is_available(self): return True
            def enhance(self, image): raise RuntimeError("boom")
        _, meta = self._assert_valid_with(LightingEnhancer(reconstruction_model=Broken()), _face_like(0.12))
        self.assertFalse(meta["deep_reconstruction_applied"])

    def test_missing_checkpoint_or_bad_arch_is_unavailable(self):
        missing = LightingEnhancer(LightingEnhancerConfig(enable_reconstruction=True, checkpoint_path="/nope.pth"))
        self.assertFalse(missing.reconstruction_model.is_available())
        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            bad = LightingEnhancer(LightingEnhancerConfig(enable_reconstruction=True, checkpoint_path=handle.name,
                                                          model_arch="unknown"))
            self.assertFalse(bad.reconstruction_model.is_available())

    @unittest.skipIf(torch is None, "torch not installed")
    def test_configured_zero_dce_checkpoint_and_torchscript_export(self):
        from extreme_cond_enhancer.zero_dce import ZeroDCEPP, export_torchscript
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "zero_dce_pp.pth"
            torch.save({"module." + k: v for k, v in ZeroDCEPP().state_dict().items()}, state_path)
            script_path = export_torchscript(ZeroDCEPP(), Path(tmp) / "zero_dce_pp.torchscript.pt")
            for arch, path in (("zero_dce_pp", state_path), ("torchscript", script_path)):
                config = LightingEnhancerConfig(enable_reconstruction=True, checkpoint_path=str(path),
                                                model_arch=arch, device="cpu")
                enhancer = LightingEnhancer(config)
                self.assertTrue(enhancer.reconstruction_model.is_available(), arch)
                raw = enhancer.reconstruction_model.enhance(_face_like(0.12))
                self.assertEqual(raw.shape, (96, 80, 3))
                self.assertEqual(raw.dtype, np.uint8)
                self._assert_valid_with(enhancer, _face_like(0.12))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_training_pipeline_smoke(self):
        import cv2
        from extreme_cond_enhancer import train_reconstruction
        from extreme_cond_enhancer.zero_dce import load_zero_dce_pp
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "faces"
            data.mkdir()
            for i in range(4):
                cv2.imwrite(str(data / f"{i}.png"), _face_like(0.4 + 0.1 * i))
            out = Path(tmp) / "run"
            train_reconstruction.main(["--data", str(data), "--out", str(out), "--epochs", "1", "--size", "32",
                                       "--batch-size", "2", "--synthesize", "--device", "cpu", "--export-torchscript"])
            self.assertTrue((out / "best.pth").is_file())
            self.assertTrue((out / "zero_dce_pp.torchscript.pt").is_file())
            load_zero_dce_pp(out / "best.pth")

    def _assert_valid_with(self, enhancer, image):
        output, metadata = enhancer.enhance(image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertIn("latency_ms", metadata)
        return output, metadata

    def test_attendance_enhances_only_after_liveness(self):
        """The integration contract keeps anti-spoofing on the original frame."""
        # Load processing with its heavyweight runtime imports replaced by stubs.
        dependencies = {
            "liveness.liveness_detector": {"LivenessDetector": object},
            "emotion.emotion_detector": {"EmotionDetector": object},
            "backend.head_pose": {"HeadPoseEstimator": object},
            "backend.face_recognizer": {"FaceRecognizer": object},
            "backend.database": {"get_all_embeddings": lambda: []},
        }
        modules = {name: types.ModuleType(name) for name in dependencies}
        for name, attributes in dependencies.items():
            for attribute, value in attributes.items():
                setattr(modules[name], attribute, value)
        path = Path(__file__).resolve().parents[2] / "backend" / "processing.py"
        spec = importlib.util.spec_from_file_location("processing_enhancement_test", path)
        processing = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(processing)
        events = []
        class Live:
            def check(self, frame):
                events.append("liveness")
                return True, 0.99, "Real", [10, 10, 30, 30]
        class Enhancer:
            def enhance(self, face):
                events.append("enhance")
                return face, {"applied": False, "reason": "test"}
        class Emotion:
            def predict(self, face): return "neutral", "🙂", 0.9
        class Recognizer:
            def extract_embedding(self, face): return np.ones(2, dtype=np.float32)
            def find_match(self, emb, db):
                return {"matched": False, "employee_id": None, "name": None, "similarity": 0.0}
        with patch.object(processing, "_get_liveness", return_value=Live()), \
             patch.object(processing, "_get_head_pose", return_value=type("P", (), {"estimate": lambda self, f: None})()), \
             patch.object(processing, "_get_lighting_enhancer", return_value=Enhancer()), \
             patch.object(processing, "_get_emotion", return_value=Emotion()), \
             patch.object(processing, "_get_recognizer", return_value=Recognizer()), \
             patch.object(processing, "get_all_embeddings", return_value=[]):
            result = processing.process_attendance_frame(np.full((64, 64, 3), 80, np.uint8))
        self.assertEqual(events, ["liveness", "enhance"])
        self.assertEqual(result["lighting_enhancement"]["reason"], "test")


if __name__ == "__main__": unittest.main()
