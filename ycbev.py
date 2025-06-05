# @author Pavel Rojtberg
# SPDX-License-Identifier: MIT

import copy
import json

import numpy as np
import cv2

import zstandard

FRAME_TIME_RGB = 1000000//30 # μs

def load_event_data(path):
    """
    Load event data from a zstd compressed file
    @return ts, xs, ys, ps: timestamps, x, y, polarity
    """
    dctx = zstandard.ZstdDecompressor() 
    bytes_ = dctx.decompress(path.read_bytes())
    data = np.frombuffer(bytes_, np.int32).reshape(-1, 2)

    XMASK = 0x3FFF
    YMASK = 0x3FFF << 14
    PMASK = 0xF << 28

    xs = data[:, 1] & XMASK
    ys = (data[:, 1] & YMASK) >> 14
    ps = (data[:, 1] & PMASK) >> 28

    return data[:, 0], xs, ys, ps

def write_event_data(path, ts, xs, ys, ps):
    """
    Write event data to a zstd compressed file
    @param ts: timestamps in microseconds
    @param xs: x coordinates
    @param ys: y coordinates
    @param ps: polarities (0 or 1)
    """
    B14MASK = 0x3FFF

    data = np.empty((len(ts), 2), dtype=np.int32)
    data[:, 0] = ts
    data[:, 1] = (xs & B14MASK) | ((ys & B14MASK) << 14) | ((ps & 0xF) << 28)

    dctx = zstandard.ZstdCompressor(threads=-1)
    with open(path, "wb") as f:
        f.write(dctx.compress(data.tobytes()))

def undistort_event_data(ts, xs, ys, ps, K, cdist, imsize, Knew):
    """
    transforms event coordinates to compensate for lens distortion and crop to image size
    
    @note if you are using event-histograms, undistorting the histogram image using cv2.undistort is preferable
    """
    pt2d = np.vstack((xs, ys))
    pt2d = cv2.undistortPoints(pt2d.astype(np.float32), K, cdist, None, None, Knew)
    pt2d = pt2d.reshape(-1,2).astype(np.uint32)
    mask = (pt2d[:, 0] < imsize[0]) & (pt2d[:, 1] < imsize[1])
    xs, ys = pt2d[mask].T.astype(np.int32)
    return ts[mask], xs, ys, ps[mask]

class EventFrameAligner:
    def __init__(self, calib_ev, calib_rig):
        ## rig calibration
        T_c2ev = np.eye(4)
        calib_stereo = json.load(open(calib_rig))
        T_c2ev[:3, :3] = calib_stereo["R_c2ev"]
        T_c2ev[:3, 3] = np.array(calib_stereo["t_c2ev"]).ravel()

        ## intrinsic calibration
        calib = json.load(open(calib_ev))
        self.K = np.float32(calib["camera_matrix"])
        self.cdist = np.float32(calib["distortion_coefficients"])

        self.T_c2ev = T_c2ev

    def align_rgb_poses(self, poses):
        ret = copy.deepcopy(poses) # dont modify source
        for pose in ret:
            T_m2c = np.eye(4)
            T_m2c[:3, :3] = pose["cam_R_m2c"]
            T_m2c[:3, 3] = pose["cam_t_m2c"]
            T_m2c = self.T_c2ev.dot(T_m2c)
            pose["cam_R_m2c"] = T_m2c[:3, :3]
            pose["cam_t_m2c"] = T_m2c[:3, 3]
        return ret

def align_depth_to_rgb(depth, calib_rgbd):
    """
    remap depth to rgb image coordinates
    """
    K = np.float32(calib_rgbd["camera_matrix"])
    Kd = np.float32(calib_rgbd["depth_camera_matrix"])
    cdist = np.float32(calib_rgbd["distortion_coefficients"])
    Rt = np.float32(calib_rgbd["Rt_d2col"])
    return cv2.rgbd.registerDepth(Kd, K, cdist, Rt, depth, calib_rgbd["img_size"], depthDilation=True)

def read_ply_xyz(filename):
    f = open(filename, 'rb')

    if f.readline().strip() != b"ply":
        raise ValueError("Not a ply file")
    
    f.readline() # endianess
    f.readline() # comment
    num_vertices = int(f.readline().strip().split()[2])

    # expect format as below
    NUM_PROPS = 5
    for _ in range(NUM_PROPS):
        f.readline()
    
    if f.readline().strip() != b"end_header":
        raise ValueError("Unexpected ply file")

    xyz = np.frombuffer(f.read(num_vertices*3*4), dtype=np.float32).reshape(num_vertices, 3)
    return xyz

def bbox_from_xyz(K, cdist, R, t, xyz):
    pts_2d = cv2.projectPoints(xyz, R, t, K, cdist)[0].reshape(-1, 2).astype(np.int32)
    return pts_2d.min(axis=0), pts_2d.max(axis=0)