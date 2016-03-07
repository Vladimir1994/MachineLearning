from __future__ import division

import numpy as np


dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


def _assert_compatible(im1, im2):
    if not im1.dtype == im2.dtype:
        raise ValueError('Input images must have the same dtype.')
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    if im1.dtype != float_type:
        im1 = im1.astype(float_type)
    if im2.dtype != float_type:
        im2 = im2.astype(float_type)
    return im1, im2


def compare_mse(im1, im2):
    _assert_compatible(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, dynamic_range=None):
    _assert_compatible(im_true, im_test)
    if dynamic_range is None:
        dmin, dmax = dtype_range[im_true.dtype.type]
        true_min, true_max = np.min(im_true), np.max(im_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the dynamic_range")
        if true_min >= 0:
            dynamic_range = dmax
        else:
            dynamic_range = dmax - dmin

    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((dynamic_range ** 2) / err)

