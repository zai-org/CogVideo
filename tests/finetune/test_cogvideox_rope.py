import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

from finetune.cogvideox_rope import prepare_cogvideox_rotary_positional_embeddings


class CogVideoXRotaryEmbeddingTests(unittest.TestCase):
    def test_cogvideox_1_0_non_base_resolution_matches_inference(self):
        config = SimpleNamespace(
            patch_size=2,
            patch_size_t=None,
            sample_height=60,
            sample_width=90,
            attention_head_dim=64,
        )
        height, width, num_frames = 576, 768, 5

        actual = prepare_cogvideox_rotary_positional_embeddings(
            height,
            width,
            num_frames,
            config,
            vae_scale_factor_spatial=8,
            device=torch.device("cpu"),
        )

        grid_height = height // (8 * config.patch_size)
        grid_width = width // (8 * config.patch_size)
        base_size_width = config.sample_width // config.patch_size
        base_size_height = config.sample_height // config.patch_size
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        expected = get_3d_rotary_pos_embed(
            embed_dim=config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=torch.device("cpu"),
        )

        legacy = get_3d_rotary_pos_embed(
            embed_dim=config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=torch.device("cpu"),
        )

        for actual_tensor, expected_tensor, legacy_tensor in zip(actual, expected, legacy):
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))
            self.assertFalse(torch.equal(actual_tensor, legacy_tensor))

    def test_cogvideox_1_5_uses_pretrained_grid_and_temporal_patching(self):
        config = SimpleNamespace(
            patch_size=2,
            patch_size_t=2,
            sample_height=96,
            sample_width=170,
            attention_head_dim=64,
        )
        sentinel = (object(), object())

        with patch(
            "finetune.cogvideox_rope.get_3d_rotary_pos_embed", return_value=sentinel
        ) as get_rope:
            actual = prepare_cogvideox_rotary_positional_embeddings(
                720,
                1280,
                81,
                config,
                vae_scale_factor_spatial=8,
                device=torch.device("cpu"),
            )

        self.assertIs(actual, sentinel)
        get_rope.assert_called_once_with(
            embed_dim=64,
            crops_coords=None,
            grid_size=(45, 80),
            temporal_size=41,
            grid_type="slice",
            max_size=(48, 85),
            device=torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
