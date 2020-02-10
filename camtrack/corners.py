#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_0 = frame_sequence[0]

    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params).squeeze(-2)
    radius = 11

    ids = np.arange(len(p0))
    next_id = len(p0)
    corners = FrameCorners(ids, p0, np.full(len(p0), radius))

    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK((image_0 * 255).astype(np.uint8), (255 * image_1).astype(np.uint8),
                                               p0, cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params)

        idx = (st == 1).reshape(-1)
        p0 = p1[idx]
        ids = ids[idx]

        if len(p0) < feature_params['maxCorners']:
            mask = np.full(image_1.shape, 255, dtype=np.uint8)
            for arr in p0:
                x, y = arr
                cv2.circle(mask, (x, y), feature_params['minDistance'], 0, -1)
            new_centers = cv2.goodFeaturesToTrack(image_1, mask=mask, **feature_params)
            add_length = min(feature_params['maxCorners'] - len(ids), len(new_centers))
            if new_centers is not None:
                ids = np.concatenate([ids, np.arange(next_id, next_id + add_length)])
                p0 = np.concatenate([p0, new_centers[:add_length].squeeze(-2)])
                next_id += add_length

        corners = FrameCorners(ids, p0, np.full(len(p0), radius))

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
