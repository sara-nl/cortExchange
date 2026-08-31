"""
Resampling for the V3 preprocessing path.

Vendored rather than imported from astroNNomy, because this is the part that distinguishes the
architecture versions: V2 resamples the way it always did, V3 the way below. Pulling either from
the package would make an astroNNomy upgrade silently change what a pinned version predicts.
"""

import torch
from torch.nn.functional import interpolate


def resize_and_noise(x, size, noise_sigma=0.0):
    """
    Resample to `size`, then optionally add Gaussian noise.

    antialias=True is the low-pass prefilter that keeps decimation from aliasing high-frequency
    power into the image; it is a no-op when upsampling. noise_sigma defaults to 0, so inference
    is deterministic - it exists only to reproduce the jitter older versions applied.
    """
    *_, h, w = x.shape
    if size != h or size != w:
        if x.dtype in (torch.float32, torch.float64):
            x = interpolate(
                x, size=(size, size), mode="bilinear", antialias=True, align_corners=False
            )
        else:
            # bilinear+antialias is not implemented for the reduced-precision types
            x = interpolate(
                x.to(torch.float32),
                size=(size, size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            ).to(x.dtype)

    if noise_sigma:
        # generate in float32: torch has no vectorized normal kernel for bfloat16/float16
        noise = torch.randn(x.shape, device=x.device, dtype=torch.float32).to(x.dtype)
        x = x + noise * noise_sigma

    return x
