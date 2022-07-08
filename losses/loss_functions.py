from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .inverse_warp import inverse_warp2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k // 2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
                 (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, calib_cam, poses, poses_inv, origin_mask, hparams):
    diff_img_list = []
    diff_color_list = []
    diff_depth_list = []
    valid_mask_list = []
    generated_img_list = []
    depth_mask_list = []

    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1, ref_img_generated, ref_depth_mask, tgt_point_cloud = compute_pairwise_loss(
            tgt_img,
            ref_img,
            tgt_depth,
            ref_depth,
            pose,
            calib_cam,
            hparams,
            origin_mask
        )

        diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2, tgt_img_generated, tgt_depth_mask, ref_point_cloud = compute_pairwise_loss(
            ref_img,
            tgt_img,
            ref_depth,
            tgt_depth,
            pose_inv,
            calib_cam,
            hparams,
            origin_mask
        )

        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]
        generated_img_list += [ref_img_generated, ref_img]
        depth_mask_list += [ref_depth_mask, tgt_depth_mask]

    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)
    generate_img = torch.cat(generated_img_list, dim=0)
    depth_mask = torch.cat(depth_mask_list, dim=0)

    # using photo loss to select best match in multiple views
    if not hparams.no_min_optimize:
        indices = torch.argmin(diff_color, dim=1, keepdim=True)

        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)

    photo_loss, mask = mean_on_mask(diff_img, valid_mask, origin_mask)
    geometry_loss, _ = mean_on_mask(diff_depth, valid_mask, origin_mask)

    return photo_loss, geometry_loss, generate_img, mask, diff_img, depth_mask


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, calib_cam, hparams, origin_mask):
    mask = origin_mask.expand_as(tgt_depth).to(tgt_depth.get_device())
    ref_img_warped, projected_depth, computed_depth, point_cloud = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, calib_cam, mask, padding_mode='zeros')

    diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth)
    
    # masking zero values
    valid_mask_ref = (ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask = valid_mask_tgt * valid_mask_ref
    
    depth_mask = computed_depth - projected_depth

    diff_color = (tgt_img - ref_img_warped).abs().mean(dim=1, keepdim=True)
    if not hparams.no_auto_mask:
        identity_warp_err = (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

    diff_img = (tgt_img - ref_img_warped).abs()
    if not hparams.no_ssim:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    diff_img = torch.sum(diff_img, dim=1, keepdim=True)

    # reduce photometric loss weight for dynamic regions
    if not hparams.no_dynamic_mask:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    return diff_img, diff_color, diff_depth, valid_mask, ref_img_warped, depth_mask, point_cloud


# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask, origin_mask):
    origin_mask = origin_mask.to(valid_mask.get_device())
    origin_mask = origin_mask.unsqueeze(0)
    mask = valid_mask.expand_as(diff)
    fisheye_mask = origin_mask.expand_as(diff)

    mask = mask * fisheye_mask
    
    sum_mask = mask.sum()

    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(1000).float().to(device)
        
    return mean_value, mask


def compute_smooth_loss(tgt_depth, tgt_img, original_mask):
    def get_smooth_loss(disp, img, mask):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x * 10)
        grad_disp_y *= torch.exp(-grad_img_y * 10)
        
        # grad_disp_x *= mask[:, :, :, 1:]
        # grad_disp_y *= mask[:, :, 1:, :]
        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]
        mean_grad_mask_x = (grad_disp_x * mask_x).sum() / mask_x.sum()
        mean_grad_mask_y = (grad_disp_y * mask_y).sum() / mask_y.sum()

        return mean_grad_mask_x + mean_grad_mask_y
    
    mask = original_mask.expand_as(tgt_depth).to(tgt_depth.get_device())
    loss = get_smooth_loss(tgt_depth, tgt_img, mask)

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()

    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='nearest')

    pred = pred.view(batch_size, h, w)

    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10

    if dataset == 'ddad':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 200

    min_depth = 1e-3
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # align scale
        valid_pred = valid_pred * \
                     torch.median(valid_gt) / torch.median(valid_pred)

        valid_pred = valid_pred.clamp(min_depth, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
        sq_rel += torch.mean(((diff_i) ** 2) / valid_gt)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                                           torch.log(valid_pred)) ** 2))
        log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                                       torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]]
