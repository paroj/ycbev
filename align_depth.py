# @author Pavel Rojtberg
# SPDX-License-Identifier: MIT

import numpy as np
from pathlib import Path
import json

import cv2

seq_path = Path("./06")

def main():
    calib = json.load((seq_path / "../calib_realsense.json").open())
    K = np.float32(calib["camera_matrix"])
    Kd = np.float32(calib["depth_camera_matrix"])
    cdist = np.float32(calib["distortion_coefficients"])
    Rt = np.float32(calib["Rt_d2col"])

    seq_poses = json.load((seq_path / "scene_gt.json").open("r"))

    for dpath in sorted(seq_path.glob("depth/*.png")):
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        d = cv2.rgbd.registerDepth(Kd, K, cdist, Rt, d , (1280, 720), depthDilation=True)
        d = cv2.convertScaleAbs(d, alpha=0.1)
        d = cv2.applyColorMap(d, cv2.COLORMAP_JET)

        i = int(dpath.stem) # dont use enumerate for this, as there are gaps in the sequence
        poses = seq_poses[str(i)]

        for o in poses:
            cv2.drawFrameAxes(d, K, cdist, np.array(o["cam_R_m2c"]), np.array(o["cam_t_m2c"]), 100)

        cv2.imshow("aligned_depth", d)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()