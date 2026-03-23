import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

try:
    from huggingface_hub import try_to_load_from_cache
except Exception:  # pragma: no cover - optional import safety
    try_to_load_from_cache = None


_CLIP_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
)
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_HF_ENDPOINT_KEY = "HF_ENDPOINT"
_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _restore_environment(saved_env: dict) -> None:
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def _direct_hf_mirror_env():
    """Temporarily disable proxy env vars while keeping HF mirror enabled."""
    saved_env = {
        key: os.environ.get(key)
        for key in (*_PROXY_ENV_KEYS, "NO_PROXY", "no_proxy", _HF_ENDPOINT_KEY)
    }
    try:
        for key in _PROXY_ENV_KEYS:
            os.environ.pop(key, None)

        no_proxy_values = []
        for key in ("NO_PROXY", "no_proxy"):
            raw = saved_env.get(key)
            if raw:
                no_proxy_values.extend(item for item in raw.split(",") if item)
        if "hf-mirror.com" not in no_proxy_values:
            no_proxy_values.append("hf-mirror.com")
        joined_no_proxy = ",".join(no_proxy_values)
        os.environ["NO_PROXY"] = joined_no_proxy
        os.environ["no_proxy"] = joined_no_proxy
        os.environ[_HF_ENDPOINT_KEY] = saved_env.get(_HF_ENDPOINT_KEY) or _HF_MIRROR_ENDPOINT
        yield
    finally:
        _restore_environment(saved_env)


def _resolve_cached_hf_snapshot(
    repo_id: str,
    required_files: Tuple[str, ...] = _CLIP_REQUIRED_FILES,
) -> Optional[str]:
    if try_to_load_from_cache is None:
        return None

    resolved_files = []
    for filename in required_files:
        cached_path = try_to_load_from_cache(repo_id, filename)
        if cached_path is None or ".no_exist" in str(cached_path):
            return None
        resolved_files.append(Path(cached_path).resolve())

    snapshot_dir = resolved_files[0].parent
    if all((snapshot_dir / filename).exists() for filename in required_files):
        return str(snapshot_dir)
    return None


def _resolve_vision_tower_source(vision_tower: str) -> Tuple[str, bool]:
    if os.path.exists(vision_tower):
        return vision_tower, True

    local_snapshot = _resolve_cached_hf_snapshot(vision_tower)
    if local_snapshot is not None:
        return local_snapshot, True

    return vision_tower, False


def _load_pretrained_component(loader_cls, source: str, *, local_files_only: bool, **kwargs):
    load_kwargs = dict(kwargs)
    if local_files_only:
        load_kwargs["local_files_only"] = True
        return loader_cls.from_pretrained(source, **load_kwargs)

    with _direct_hf_mirror_env():
        return loader_cls.from_pretrained(source, **load_kwargs)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.vision_tower_source, self.vision_tower_local_only = _resolve_vision_tower_source(vision_tower)
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if self.vision_tower_source != self.vision_tower_name:
            print(f"[vision-tower] Using local cached CLIP assets: {self.vision_tower_source}")

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = _load_pretrained_component(
                CLIPVisionConfig,
                self.vision_tower_source,
                local_files_only=self.vision_tower_local_only,
            )

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = _load_pretrained_component(
            CLIPImageProcessor,
            self.vision_tower_source,
            local_files_only=self.vision_tower_local_only,
        )
        self.vision_tower = _load_pretrained_component(
            CLIPVisionModel,
            self.vision_tower_source,
            local_files_only=self.vision_tower_local_only,
            device_map=device_map,
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = _load_pretrained_component(
            CLIPImageProcessor,
            self.vision_tower_source,
            local_files_only=self.vision_tower_local_only,
        )
        self.vision_tower = _load_pretrained_component(
            CLIPVisionModel,
            self.vision_tower_source,
            local_files_only=self.vision_tower_local_only,
            device_map=device_map,
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
