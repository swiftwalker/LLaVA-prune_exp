import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llava.model.multimodal_encoder.clip_encoder import (
    _direct_hf_mirror_env,
    _resolve_vision_tower_source,
)


class ClipEncoderHelpersTests(unittest.TestCase):
    def test_resolve_vision_tower_source_prefers_existing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, local_only = _resolve_vision_tower_source(tmpdir)
        self.assertTrue(local_only)
        self.assertEqual(source, tmpdir)

    def test_resolve_vision_tower_source_uses_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir)
            for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
                (snapshot / filename).write_text("x", encoding="utf-8")

            def fake_try_to_load_from_cache(repo_id, filename):
                self.assertEqual(repo_id, "openai/clip-vit-large-patch14-336")
                return snapshot / filename

            with mock.patch(
                "llava.model.multimodal_encoder.clip_encoder.try_to_load_from_cache",
                side_effect=fake_try_to_load_from_cache,
            ):
                source, local_only = _resolve_vision_tower_source("openai/clip-vit-large-patch14-336")

        self.assertTrue(local_only)
        self.assertEqual(source, str(snapshot))

    def test_direct_hf_mirror_env_disables_proxy_and_restores(self):
        saved_http = os.environ.get("HTTP_PROXY")
        saved_https = os.environ.get("HTTPS_PROXY")
        saved_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:7990"
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7990"
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            with _direct_hf_mirror_env():
                self.assertNotIn("HTTP_PROXY", os.environ)
                self.assertNotIn("HTTPS_PROXY", os.environ)
                self.assertEqual(os.environ["HF_ENDPOINT"], "https://hf-mirror.com")
                self.assertIn("hf-mirror.com", os.environ["NO_PROXY"])

            self.assertEqual(os.environ.get("HTTP_PROXY"), "http://127.0.0.1:7990")
            self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://127.0.0.1:7990")
            self.assertEqual(os.environ.get("HF_ENDPOINT"), "https://hf-mirror.com")
        finally:
            for key, value in (
                ("HTTP_PROXY", saved_http),
                ("HTTPS_PROXY", saved_https),
                ("HF_ENDPOINT", saved_endpoint),
            ):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
