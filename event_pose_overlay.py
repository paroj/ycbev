# @author Pavel Rojtberg
# SPDX-License-Identifier: MIT

import bisect
import time
from pathlib import Path
import json
import copy

import cv2
import numpy as np
import zstandard

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

seq_path = Path("./06")

rgb_delta_t = 1000000//30

class EventFrameProcessor:
    def __init__(self, cameracalib, stereocalib):
        ## stereo calibration
        T_c2ev = np.eye(4)
        calib_stereo = json.load(open(stereocalib))
        T_c2ev[:3, :3] = calib_stereo["R_c2ev"]
        T_c2ev[:3, 3] = np.array(calib_stereo["t_c2ev"]).ravel()

        ## intrinsic calibration
        calib = json.load(open(cameracalib))
        self.K = np.float32(calib["camera_matrix"])
        self.cdist = np.float32(calib["distortion_coefficients"])

        self.T_c2ev = T_c2ev

    def transform_poses(self, poses):
        ret = copy.deepcopy(poses) # dont modify source
        for pose in ret:
            T_m2c = np.eye(4)
            T_m2c[:3, :3] = pose["cam_R_m2c"]
            T_m2c[:3, 3] = pose["cam_t_m2c"]
            T_m2c = self.T_c2ev.dot(T_m2c)
            pose["cam_R_m2c"] = T_m2c[:3, :3]
            pose["cam_t_m2c"] = T_m2c[:3, 3]
        return ret

def load_event_data(path):
    t = time.time()
    dctx = zstandard.ZstdDecompressor() 
    bytes = dctx.decompress(path.read_bytes())
    data = np.frombuffer(bytes, np.int32).reshape(-1, 2)

    XMASK = 0x3FFF
    YMASK = 0x3FFF << 14
    PMASK = 0xF << 28

    xs = data[:, 1] & XMASK
    ys = (data[:, 1] & YMASK) >> 14
    ps = (data[:, 1] & PMASK) >> 28
    print("unpacked data in", time.time() - t)

    return data[:, 0], xs, ys, ps

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
    idx0 = max(curr_t // rgb_delta_t - 1, 0) # prev
    idx1 = idx0 + 1                          # next
    poses0 = copy.deepcopy(seq_poses[str(idx0)]) # dont modify source
    poses1 = seq_poses[str(idx1)]

    if len(poses0) != len(poses1):
        return poses0 # we cannot "fade" in/ out an object

    a = (curr_t % rgb_delta_t) / rgb_delta_t

    for p0, p1 in zip(poses0, poses1):
        p0["cam_t_m2c"] = np.array(p0["cam_t_m2c"]) * (1 - a) + np.array(p1["cam_t_m2c"]) * a
        p0["cam_R_m2c"] = slerp(p0["cam_R_m2c"], p1["cam_R_m2c"], a)
    return poses0

def main():
    event_fps = 30  # sample event data to this frame rate
    delta_t=1000000//event_fps
    start_idx = 0
    max_t = delta_t

    height, width = 720, 1280

    ts, xs, ys, ps = load_event_data(Path(seq_path.stem + "_events.int32.zst"))

    event_frame = EventFrameProcessor(seq_path / "../calib_prophesee.json", seq_path / "../calib_stereo_c2ev.json")

    seq_poses = json.load((seq_path / "scene_gt.json").open("r"))

    while True:
        im = np.zeros((height, width, 3), dtype=np.uint8)
        end_idx = bisect.bisect(ts, max_t, lo=start_idx)

        im[ys[start_idx:end_idx], xs[start_idx:end_idx], ps[start_idx:end_idx]] = 200

        poses = lerp_poses(seq_poses, max_t)
        poses = event_frame.transform_poses(poses)

        for o in poses:
            # 10 cm sized axes
            cv2.drawFrameAxes(im, event_frame.K, event_frame.cdist, o["cam_R_m2c"], o["cam_t_m2c"], 100)

        max_t += delta_t
        start_idx = end_idx

        cv2.imshow("frame", im)
        k = cv2.waitKey(33)
        if k == 27: break

if __name__ == '__main__':
    main()