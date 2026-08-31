"""
FITS -> [0, 1] preprocessing, kept identical to astroNNomy's pre_processing_for_ml so that
inference sees exactly what training saw.
"""

import numpy as np
from matplotlib.colors import SymLogNorm


def get_rms(data: np.ndarray, maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS

    :param data: numpy array
    :param maskSup: mask threshold

    :return: rms --> rms of image
    """

    mIn = np.ndarray.flatten(data)
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.0
    med = np.median(m)

    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff:
            break
        rmsold = rms

    return rms  # jy/beam


def normalize_fits(image_data: np.ndarray):
    image_data = image_data.squeeze()

    # Pre-processing
    rms = get_rms(image_data)
    norm_f = SymLogNorm(
        linthresh=rms * 2, linscale=2, vmin=-rms, vmax=rms * 50000, base=10
    )

    image_data = norm_f(image_data)

    image_data = image_data - image_data.min()
    image_data = np.clip(image_data, a_min=0, a_max=1)

    return image_data[..., None]
