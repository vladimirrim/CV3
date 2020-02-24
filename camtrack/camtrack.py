#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)


class CameraTracker:
    def __init__(self, corner_storage: CornerStorage,
                 intrinsic_mat: np.ndarray, parameters: TriangulationParameters,
                 known_view_1: Tuple[int, Pose],
                 known_view_2: Tuple[int, Pose]):
        self._corner_storage = corner_storage
        self._intrinsic_mat = intrinsic_mat
        self._triangulation_parameters = parameters
        self._length = len(corner_storage)
        self._builder = PointCloudBuilder()
        self._track = [None] * self._length

        self._track[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
        self._track[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
        self._update_cloud(known_view_1[0], known_view_2[0])

    def _update_cloud(self, ind1, ind2):
        frame1, frame2 = self._corner_storage[ind1], self._corner_storage[ind2]
        mat1, mat2 = self._track[ind1], self._track[ind2]
        correspondences = build_correspondences(frame1, frame2)
        if len(correspondences.ids) == 0:
            return 0
        points, ids, _ = triangulate_correspondences(correspondences, mat1, mat2,
                                                     self._intrinsic_mat, self._triangulation_parameters)
        self._builder.add_points(ids, points)
        return len(ids)

    def track(self):
        for cur_index in range(1, self._length):
            print(f'Processing frame {cur_index}/{self._length}')
            corners = self._corner_storage[cur_index]
            ids = []
            object_points = []
            image_points = []
            for id, point in zip(corners.ids, corners.points):
                indices_x, _ = np.where(self._builder.ids == id)
                if len(indices_x) == 0:
                    continue
                ids.append(id)
                object_points.append(self._builder.points[indices_x[0]])
                image_points.append(point)
            if len(object_points) < 5:
                return None
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(object_points), np.array(image_points),
                                                             self._intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP)
            if not retval:
                return None
            view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            print(f'Position based on {len(inliers)} inliers')
            self._track[cur_index] = view_mat
            updatesCount = sum([self._update_cloud(cur_index, next_index) for next_index in range(cur_index)])
            print(f'Points updated:{updatesCount} Current cloud size:{len(self._builder.ids)}')
        return np.array(self._track), self._builder


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray,
                  known_view_1: Tuple[int, Pose],
                  known_view_2: Tuple[int, Pose]) \
        -> Optional[Tuple[List[np.ndarray], PointCloudBuilder]]:
    parameters = TriangulationParameters(max_reprojection_error=1., min_triangulation_angle_deg=2., min_depth=0.1)
    return CameraTracker(corner_storage, intrinsic_mat, parameters, known_view_1, known_view_2).track()


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat,
        known_view_1,
        known_view_2
    )

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
