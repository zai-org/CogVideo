from typing import Any, Tuple

import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid


def prepare_cogvideox_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Any,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the same rotary embeddings used by the CogVideoX inference pipelines."""
    patch_size = transformer_config.patch_size
    patch_size_t = transformer_config.patch_size_t
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)

    base_size_width = transformer_config.sample_width // patch_size
    base_size_height = transformer_config.sample_height // patch_size

    if patch_size_t is None:
        # CogVideoX 1.0 interpolates the pretrained spatial grid for other resolutions.
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )

    # CogVideoX 1.5 slices its pretrained spatial grid after temporal patching.
    base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t
    return get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(base_size_height, base_size_width),
        device=device,
    )
