# @author Pavel Rojtberg
# SPDX-License-Identifier: MIT

import time
from pathlib import Path
import json
import copy

import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import ycbev

seq_path = Path("./06")

undistort_events = False

def slerp(R0, R1, a):
    """
    spherical linear interpolation for rotation matrices
    """
    _slerp = Slerp([0, 1], R.concatenate([R.from_matrix(R0), R.from_matrix(R1)]))
    return _slerp(a).as_matrix()

def lerp_poses(seq_poses, curr_t):
    """
    linearly interpolate poses for current time stamp between neighbouring time stamps
    """
    idx0 = max(curr_t // ycbev.FRAME_TIME_RGB - 1, 0) # prev
    idx1 = idx0 + 1                                   # next

    poses0 = copy.deepcopy(seq_poses.get(str(idx0), [])) # dont modify source
    poses1 = seq_poses.get(str(idx1), [])

    if len(poses0) != len(poses1):
        return poses0 # we cannot "fade" in/ out an object

    a = (curr_t % ycbev.FRAME_TIME_RGB) / ycbev.FRAME_TIME_RGB

    for p0, p1 in zip(poses0, poses1):
        p0["cam_t_m2c"] = np.array(p0["cam_t_m2c"]) * (1 - a) + np.array(p1["cam_t_m2c"]) * a
        p0["cam_R_m2c"] = slerp(p0["cam_R_m2c"], p1["cam_R_m2c"], a)
    return poses0

def main():
    imsize = (1280, 720)
    fps = 30

    t = time.time()
    ts, xs, ys, ps = ycbev.load_event_data(Path(seq_path.stem + "_events.int32.zst"))
    print(f"unpacked data in {time.time() - t:.02f}")

    event_frame = ycbev.EventFrameAligner(seq_path / "../calib_prophesee.json", seq_path / "../calib_stereo_c2ev.json")

    if undistort_events:
        Knew = cv2.getOptimalNewCameraMatrix(event_frame.K, event_frame.cdist, imsize, 0)[0]
        t = time.time()
        ts, xs, ys, ps = ycbev.undistort_event_data(ts, xs, ys, ps, event_frame.K, event_frame.cdist, imsize, Knew)
        print(f"undistorted data in {time.time() - t:.02f}")
        event_frame.K = Knew
        event_frame.cdist = None

    seq_poses = json.load((seq_path / "scene_gt.json").open("r"))

    obj_xyzs = {}

    ev_surface = ycbev.EvSurfaceConverter((ts, xs, ys, ps), imsize, surface_fps=fps)

    while True:
        ret, im = ev_surface.read()
        if not ret:
            break

        poses = lerp_poses(seq_poses, ev_surface.max_t)
        poses = event_frame.align_rgb_poses(poses)

        for o in poses:
            # bbox
            if o["obj_id"] not in obj_xyzs:
                obj_xyzs[o["obj_id"]] = ycbev.read_ply_xyz(seq_path / f"../ycbv_models/models_eval/obj_{o['obj_id']:06d}.ply")
            
            bbox = ycbev.bbox_from_xyz(event_frame.K, event_frame.cdist, np.array(o["cam_R_m2c"]), np.array(o["cam_t_m2c"]), obj_xyzs[o["obj_id"]])
            cv2.rectangle(im, bbox[0], bbox[1], (0, 0, 255), 2)

            # 10 cm sized axes
            cv2.drawFrameAxes(im, event_frame.K, event_frame.cdist, o["cam_R_m2c"], o["cam_t_m2c"], 100)

        cv2.imshow("frame", im)
        k = cv2.waitKey(33)
        if k == 27: break

if __name__ == '__main__':
    main()