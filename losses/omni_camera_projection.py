import os
import numpy as np
import torch
from torch import nn
import cv2


class OmniCam(nn.Module):
    def __init__(self, calib_file):
        super().__init__()
        self.readParam(calib_file, 1024, 1024)

    def readParam(self, calib_file, image_h, image_w):
        cam2world_poly = None
        world2cam = None

        C = 0.0
        D = 0.0
        E = 0.0

        center_x = 0.0
        center_y = 0.0

        with open(calib_file, 'r') as fs:
            file_lines = fs.readlines()

            line_counter = 0

            for one_line in file_lines:
                curr_line = one_line.rstrip()
                if one_line[0] == '#' or curr_line == '':
                    continue

                if line_counter == 0:
                    cam2world_poly = np.array([float(i) for i in one_line.split()[1:]])
                    line_counter += 1
                    continue

                if line_counter == 1:
                    world2cam = np.array([float(i) for i in one_line.split()[1:]])
                    line_counter += 1
                    continue

                if line_counter == 2:
                    center_x = float(one_line.split()[1])
                    center_y = float(one_line.split()[0])
                    line_counter += 1
                    continue

                if line_counter == 3:
                    C = float(one_line.split()[0])
                    D = float(one_line.split()[1])
                    E = float(one_line.split()[2])
                    line_counter += 1
                    continue

        print('-- image to world poly: ')
        print(cam2world_poly)
        print('-- world to image poly: ')
        print(world2cam)

        print('-- C D E')
        print('{:10.9f}, {:10.9f}, {:10.9f}'.format(C, D, E))

        print('-- center_x center_y')
        print('{:10.9f}, {:10.9f}'.format(center_x, center_y))

        self.i2c = nn.Parameter(torch.from_numpy(cam2world_poly), requires_grad=False)
        self.c2i = nn.Parameter(torch.from_numpy(world2cam), requires_grad=False)

        self.cde = nn.Parameter(torch.tensor([C, D, E]), requires_grad=False)
        self.center_xy = nn.Parameter(torch.tensor([center_x, center_y]), requires_grad=False)

        self.affine_mat = nn.Parameter(torch.tensor([[1, -self.cde[1]], [-self.cde[2], self.cde[0]]]), requires_grad=False)

        i_range = torch.arange(0, image_h).view(1, image_h, 1).expand(1, image_h, image_w).type_as(self.center_xy).unsqueeze(3)  # [1, H, W]
        j_range = torch.arange(0, image_w).view(1, 1, image_w).expand(1, image_h, image_w).type_as(self.center_xy).unsqueeze(3)  # [1, H, W]
        self.pixel_coords = nn.Parameter(torch.concat((j_range, i_range), dim=3).reshape((-1, 2)).unsqueeze(0), requires_grad=False)

    def polyval(self, P, x):
        npol = P.shape[0]
        val = torch.zeros_like(x)
        for i in range(npol - 1):
            val = val * x + P[i] * x
        val += P[-1]
        return val

    def generateMask(self):
        p = self.pixel_coords[0, ...].transpose()

        # flip axis
        x_c = p[0, :].reshape((1, -1)) - self.center_xy[0]
        y_c = p[1, :].reshape((1, -1)) - self.center_xy[1]

        p_c = torch.concat((x_c, y_c), dim=0)
        invdet = 1.0 / (self.cde[0] - self.cde[1] * self.cde[2])
        A_inv = invdet * self.affine_mat

        p_a = A_inv.matmul(p_c)
        # flip axis
        x_a = p_a[0, :].reshape((1, -1))
        y_a = p_a[1, :].reshape((1, -1))

        rho = torch.sqrt(torch.mul(x_a, x_a) + torch.mul(y_a, y_a))
        z = -self.polyval(torch.flip(self.i2c, dims=[0]), rho).reshape((1, -1))

        theta = torch.atan2(rho, z)
        return theta

    # def polyval(self, P, x):
    #     P_flip = np.flip(P, axis=0)
    #     return np.polyval(P_flip, x)

    def pixelToRay(self, p):
        # flip axis
        x_c = p[0, :].reshape((1, -1)) - self.center_xy[0]
        y_c = p[1, :].reshape((1, -1)) - self.center_xy[1]

        p_c = torch.concat((x_c, y_c), dim=0)
        invdet = 1.0 / (self.cde[0] - self.cde[1] * self.cde[2])
        A_inv = invdet * self.affine_mat

        p_a = A_inv.matmul(p_c)
        # flip axis
        x_a = p_a[0, :].reshape((1, -1))
        y_a = p_a[1, :].reshape((1, -1))

        rho = torch.sqrt(torch.mul(x_a, x_a) + torch.mul(y_a, y_a))
        z = -self.polyval(torch.flip(self.i2c, dims=[0]), rho).reshape((1, -1))

        ray_norm = torch.sqrt(torch.mul(x_a, x_a) + torch.mul(y_a, y_a) + torch.mul(z, z))
        # theta is angle from the optical axis.
        out = torch.concat((torch.divide(x_c, ray_norm), torch.divide(y_c, ray_norm), torch.divide(z, ray_norm)))
        return out

    def liftImageToRay_for_test(self, depth):
        # b, h, w = depth.size()
        # rays_3d = torch.zeros((b, 3, h, w), dtype=torch.double)

        # tail_coords = torch.ones((h, w), dtype=torch.double)

        # for i in range(b):
        clean_coords = self.pixel_coords[0, ...].transpose(0, 1)
        np_undist_coords = self.pixelToRay(clean_coords)

        # rays_3d[i, ...] = torch.stack((undistorted_pts_x, undistorted_pts_y, tail_coords), dim=0)
        # print()

        return np_undist_coords

    def spaceToImage(self, P):
        norm = torch.sqrt(torch.pow(P[0, :], 2) + torch.pow(P[1, :], 2))
        theta = torch.atan2(-P[2, :], norm)
        rho = self.polyval(torch.flip(self.c2i, dims=[0]), theta)

        # flip axis
        x = P[0, :] / norm * rho
        y = P[1, :] / norm * rho

        x_a = x * self.cde[0] + y * self.cde[1] + self.center_xy[0]
        y_a = x * self.cde[2] + y + self.center_xy[1]

        x_im = x_a.reshape((1, -1))
        y_im = y_a.reshape((1, -1))
        out = torch.concat((x_im, y_im), dim=0)

        return out

    def batchLiftImageToRay(self, depth):
        """
        Args:
            depth: depth image used for create batch rays tensor
            pixel_coords:
        Returns:
            batch_rays_3d: [b, 3, H, W]
        """

        b, h, w = depth.size()
        batch_rays_3d = torch.zeros((b, 3, h, w)).type_as(self.pixel_coords[0, ...])

        # np_pixel_coords = pixel_coords.cpu().detach().numpy()

        for i in range(b):
            clean_coords = self.pixel_coords[0, ...].transpose(0, 1)
            np_undist_coords = self.pixelToRay(clean_coords).reshape((3, h, -1))
            batch_rays_3d[i, ...] = np_undist_coords

        max_depth = torch.max(depth)
        min_depth = torch.min(depth)
        # print(depth.get_device())
        # print('+++')
        # print(batch_rays_3d.get_device())
        # print('---')
        return batch_rays_3d * depth.unsqueeze(1)

    def batchSpaceToImage(self, P):
        b, c, HW = P.shape
        if P.is_cuda:
            batch_image_coords = torch.zeros((b, c, HW)).type_as(P).to(P.get_device())
        else:
            batch_image_coords = torch.zeros((b, c, HW)).type_as(P)

        for i in range(b):
            one_batch_image_coords = self.spaceToImage(P[i, ...])
            if P.is_cuda:
                homo_image_coords = torch.concat((one_batch_image_coords, torch.ones((1, HW)).to(P.get_device())), dim=0)
            else:
                homo_image_coords = torch.concat((one_batch_image_coords, torch.ones((1, HW))), dim=0)
                
            batch_image_coords[i, ...] = homo_image_coords

        return batch_image_coords

# current_file_path = os.getcwd()
# print(current_file_path + '/../configs/calib/cam_omni_0.txt')
#
# calib_path = current_file_path + '/../configs/calib/cam_omni_0.txt'
# print(os.path.exists(calib_path))
#
# calib_cam = OmniCam(calib_path)
#
#
# def set_id_grid(depth):
#     b, h, w = depth.size()
#     i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth).unsqueeze(3)  # [1, H, W, 1], y
#     j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth).unsqueeze(3)  # [1, H, W, 1], x
#     # ones = torch.ones(1, h, w).type_as(depth)
#     pixel_coords = torch.concat((j_range, i_range), dim=3).reshape((-1, 2)).unsqueeze(0)  # [1, 2, H, W]
#     return pixel_coords
#
#
# depth = torch.from_numpy(np.zeros((3, 1024, 1024), dtype=np.double))
# pixel_coords = set_id_grid(depth)
#
# theta = calib_cam.generateMask(pixel_coords).reshape((1024, 1024))
# mask = np.zeros_like(theta, dtype=np.uint8)
# mask[theta < 1.1] = 255
#
# image = cv2.imread(current_file_path + '/../configs/calib/000000501.png')
# image[theta > 1.1] = 0
#
# cv2.imwrite(current_file_path + '/../configs/calib/mask.png', mask)
# cv2.imshow('binary mask', mask)
# cv2.imshow('mask', image)
# cv2.waitKey(0)

#
# batch_rays = calib_cam.batchLiftImageToRay(depth, pixel_coords)
#
# points_3d = batch_rays.reshape((3, 3, -1))
# calib_cam.batchSpaceToImage(points_3d)
#
#
# def check_pixelToRay():
#     rays_3d = calib_cam.liftImageToRay_for_test(depth, pixel_coords)
#     image = cv2.imread(current_file_path + '/../configs/calib/000000501.png')
#
#     cv2.imshow('current image', image)
#
#     fx = 200
#     fy = 200
#     cx = 512
#     cy = 512
#
#     np_pixel_coords = pixel_coords.cpu().detach().numpy()
#     undist_image = np.zeros((1024, 1024, 3), np.uint8)
#
#     for i in range(rays_3d.shape[1]):
#         undist_x = int(fx * rays_3d[:, i][0] + cx)
#         undist_y = int(fy * rays_3d[:, i][1] + cy)
#
#         if 0 < undist_x < 1024 and 0 < undist_y < 1024:
#             coorespond_x = int(np_pixel_coords[0][i][0])
#             coorespond_y = int(np_pixel_coords[0][i][1])
#
#             undist_image[undist_y][undist_x][0] = image[coorespond_y][coorespond_x][0]
#             undist_image[undist_y][undist_x][1] = image[coorespond_y][coorespond_x][1]
#             undist_image[undist_y][undist_x][2] = image[coorespond_y][coorespond_x][2]
#
#     cv2.imshow('undist image', undist_image)
#     cv2.waitKey(0)
#
#
# def check_spaceToImage():
#     image = cv2.imread(current_file_path + '/../configs/calib/000000501.png')
#     cv2.imshow('current image', image)
#
#     # fx = 200
#     # fy = 200
#     # cx = 512
#     # cy = 512
#     #
#     # cam_K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.double).reshape((3, 3))
#     #
#     # np_pixel_coords = pixel_coords.cpu().detach().numpy()
#     # map_x = np.zeros((1024, 1024), np.float32)
#     # map_y = np.zeros((1024, 1024), np.float32)
#     #
#     # homo_image_coords = np.concatenate((np_pixel_coords[0, ...], np.ones((np_pixel_coords.shape[1], 1), dtype=np.double)), axis=1).transpose()
#     # undist_rays = np.linalg.inv(cam_K) @ homo_image_coords
#     undist_rays = calib_cam.liftImageToRay_for_test(depth, pixel_coords)
#
#     image_coords = calib_cam.spaceToImage(undist_rays)
#     map_x = image_coords[0, ...].reshape((1024, 1024)).astype(np.float32)
#     map_y = image_coords[1, ...].reshape((1024, 1024)).astype(np.float32)
#
#     dst = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
#
#     dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     image_gray = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     diff_im = cv2.subtract(dst_gray, image_gray)
#
#     cv2.imshow('undist', dst)
#     cv2.imshow('diff', diff_im)
#
#     cv2.waitKey(0)
#
#     print()
#
#
# check_spaceToImage()
