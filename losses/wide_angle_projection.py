import cv2 as cv
import numpy as np
import torch


class WideAngleCamModel:
    def __init__(self):
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0
        self.k5 = 0.0

        self.mu = 0.0
        self.mv = 0.0
        self.u0 = 0.0
        self.v0 = 0.0

        self.m_inv_K11 = 0.0
        self.m_inv_K13 = 0.0
        self.m_inv_K22 = 0.0
        self.m_inv_K23 = 0.0

    def readParameters(self, calib_file):
        fs = cv.FileStorage()
        fs.open(filename=calib_file, flags=cv.FILE_STORAGE_READ)
        camera_param = fs.getNode('projection_parameters')

        self.k2 = camera_param.getNode('k2').real()
        self.k3 = camera_param.getNode('k3').real()
        self.k4 = camera_param.getNode('k4').real()
        self.k5 = camera_param.getNode('k5').real()

        self.mu = camera_param.getNode('mu').real()
        self.mv = camera_param.getNode('mv').real()
        self.u0 = camera_param.getNode('u0').real()
        self.v0 = camera_param.getNode('v0').real()

        self.m_inv_K11 = 1.0 / self.mu
        self.m_inv_K13 = -self.u0 / self.mu
        self.m_inv_K22 = 1.0 / self.mv
        self.m_inv_K23 = -self.v0 / self.mv

        self.K = np.array([self.mu, 0.0, self.u0,
                           0.0, self.mv, self.v0,
                           0.0, 0.0, 1.0]).reshape((3, 3))

        self.D = np.array([self.k2, self.k3, self.k4, self.k5])

        print('-- k2: ' + str(self.k2))
        print('-- k3: ' + str(self.k3))
        print('-- k4: ' + str(self.k4))
        print('-- k5: ' + str(self.k5))

        print('-- mu: ' + str(self.mu))
        print('-- mv: ' + str(self.mv))
        print('-- u0: ' + str(self.u0))
        print('-- v0: ' + str(self.v0))

        print('-- inv k11: ' + str(self.m_inv_K11))
        print('-- inv k13: ' + str(self.m_inv_K13))
        print('-- inv k22: ' + str(self.m_inv_K22))
        print('-- inv k23: ' + str(self.m_inv_K23))

    def liftImageToRay(self, depth, pixel_coords):
        b, h, w = depth.size()
        batch_pixel_coords = torch.zeros((b, 3, h, w), dtype=torch.double)

        tail_coords = torch.ones((h, w), dtype=torch.double)
        np_pixel_coords = pixel_coords.cpu().detach().numpy()

        for i in range(b):
            np_undist_pts_x = cv.fisheye.undistortPoints(np_pixel_coords, self.K, self.D)[..., np.newaxis][..., 0, 0].reshape((1024, 1024))
            undistorted_pts_x = torch.from_numpy(cv.fisheye.undistortPoints(np_pixel_coords, self.K, self.D)[..., np.newaxis][..., 0, 0].reshape((1024, 1024)))
            undistorted_pts_y = torch.from_numpy(cv.fisheye.undistortPoints(np_pixel_coords, self.K, self.D)[..., np.newaxis][..., 1, 0].reshape((1024, 1024)))

            batch_pixel_coords[i, ...] = torch.stack((undistorted_pts_x, undistorted_pts_y, tail_coords), dim=0)
            print()

        return batch_pixel_coords


import os

current_file_path = os.getcwd()
print(current_file_path + '/../configs/calib/cam_0.yaml')

calib_path = current_file_path + '/../configs/calib/cam_0.yaml'
print(os.path.exists(calib_path))

calib_cam = WideAngleCamModel()
calib_cam.readParameters(calib_path)


def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth).unsqueeze(3)  # [1, H, W, 1], y
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth).unsqueeze(3)  # [1, H, W, 1], x
    # ones = torch.ones(1, h, w).type_as(depth)
    pixel_coords = torch.concat((j_range, i_range), dim=3).reshape((-1, 2)).unsqueeze(0)  # [1, 3, H, W]
    return pixel_coords


depth = torch.from_numpy(np.zeros((3, 1024, 1024), dtype=np.double))
pixel_coords = set_id_grid(depth)
batch_cam_coords = calib_cam.liftImageToRay(depth, pixel_coords)
print()
