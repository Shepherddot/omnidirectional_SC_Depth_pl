import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

import datasets.custom_transforms as custom_transforms
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet
from losses.loss_functions import (compute_errors, compute_smooth_loss, photo_and_geometry_loss)
from models.DepthNet import DepthNet
# from models.PoseNet import PoseNet
from visualization import *
from datasets.robocar_oxford import OxfordDataset
from losses.omni_camera_projection import OmniCam
import os


class SC_Depth(LightningModule):
    def __init__(self, hparams, model_path=None, is_train=True):
        super(SC_Depth, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        
        # self.pose_net = PoseNet()
        if is_train:
            self.get_data_ready()
        
        current_file_path = os.getcwd()
        calib_path = current_file_path + '/configs/calib/cam_right.txt'
        self.calib_cam = OmniCam(calib_path)

        binary_mask = cv2.imread(current_file_path + '/configs/calib/mask.png', cv2.IMREAD_GRAYSCALE) / 255
        self.origin_mask = torch.from_numpy(binary_mask).unsqueeze(0)
        
        if model_path:
            self.load_model(model_path=model_path)

    def get_data_ready(self):
        if self.hparams.hparams.dataset_name == 'kitti':
            training_size = [256, 832]
        elif self.hparams.hparams.dataset_name == 'ddad':
            training_size = [384, 640]
        else:
            print('unknown dataset type')

        # data loader
        train_transform = custom_transforms.Compose([
            # custom_transforms.RandomHorizontalFlip(),
            # custom_transforms.RandomScaleCrop(),
            # custom_transforms.RescaleTo(training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        valid_transform = custom_transforms.Compose([
            # custom_transforms.RescaleTo(training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

        self.train_dataset = OxfordDataset(
            self.hparams.hparams.dataset_dir,
            transform=train_transform,
            train=True,
            sequence_length=self.hparams.hparams.sequence_length,
            skip_frames=self.hparams.hparams.skip_frames,
            dataset=self.hparams.hparams.dataset_name
        )

        if self.hparams.hparams.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=valid_transform,
                dataset=self.hparams.hparams.dataset_name
            )
        elif self.hparams.hparams.val_mode == 'photo':
            self.val_dataset = OxfordDataset(
                self.hparams.hparams.dataset_dir,
                transform=valid_transform,
                train=False,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name
            )
        else:
            print("wrong validation mode")

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validatioin'.format(len(self.val_dataset)))

    def inference_depth(self, img):
        pred_depth = self.depth_net(img)
        return pred_depth
    
    def load_model(self, model_path):
        # current_path = os.getcwd() + '/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt'
        pretrained = torch.load(model_path, map_location=torch.device('cpu'))
        depth_parameter = self.depth_net.state_dict()
        got_parameter = dict()
        
        for param in pretrained['state_dict'].items():
            if 'depth_net' in param[0]:
                layer_name = param[0].split('depth_net.')[1]
                got_parameter[layer_name] = param[1]
                    
        # got_parameter.update(param for param in pretrained['state_dict'].items() if param[0].split('depth_net.')[1] in depth_parameter)
        self.depth_net.load_state_dict(got_parameter)

    # def inference_pose(self, img1, img2):
    #     pred_pose = self.pose_net(img1, img2)
    #     return pred_pose

    def forward(self, tgt_img, ref_imgs):
        # in lightning, forward defines the prediction/inference actions for training
        tgt_depth = self.inference_depth(tgt_img)
        ref_depths = [self.inference_depth(im) for im in ref_imgs]

        # poses = [self.inference_pose(tgt_img, im) for im in ref_imgs]
        # poses_inv = [self.inference_pose(im, tgt_img) for im in ref_imgs]

        return tgt_depth, ref_depths

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            # {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        tgt_img, ref_imgs, poses, poses_inv = batch

        # network forward
        tgt_depth, ref_depths = self(tgt_img, ref_imgs)

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight

        loss_1, loss_2, generated_img, mask, diff_img, depth_mask = photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, self.calib_cam, poses, poses_inv, self.origin_mask, self.hparams.hparams)
        loss_3 = compute_smooth_loss(tgt_depth, tgt_img, self.origin_mask)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
        
        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)

        return {'loss': loss, 'depth': tgt_depth, 'mask': mask, 'genereated': generated_img, 'diff_image':diff_img, 'depth_mask': depth_mask, 'tgt_image': tgt_img}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 100 == 1:
            depth_image = outputs['depth']
            mask_image = outputs['mask']
            rgb_img = outputs['genereated']
            diff_image = outputs['diff_image']
            projected_depth_mask = outputs['depth_mask']
            tgt_img = outputs['tgt_image']
            
            vis_img_wrapped = visualize_image(rgb_img[0])
            vis_img_origin = visualize_image(rgb_img[1])
            vis_depth, min_depth, max_depth = visualize_training_depth(depth_image[0, 0], origin_mask=self.origin_mask)  # (3, H, W)
            vis_mask = visualize_training_mask(mask_image[0, 0])  # (3, H, W)
            vis_tgt_img = visualize_image(tgt_img[0])
            
            given_mask = visualize_training_mask(self.origin_mask[0])
            vis_diff_img = visualize_training_mask(diff_image[0, 0])
            
            vis_projected_depth_mask, _, _ = visualize_training_depth(projected_depth_mask[0, 0].abs(), origin_mask=self.origin_mask)
            
            self.logger.experiment.add_image('training/img_depth_0', vis_depth, self.global_step)
            self.logger.experiment.add_image('training/img_mask_0', vis_mask, self.global_step)
            self.logger.experiment.add_image('training/img_origin_0', given_mask, self.global_step)
            self.logger.experiment.add_image('training/img_rgb_wrapped', vis_img_wrapped, self.global_step)
            self.logger.experiment.add_image('training/img_rgb_origin', vis_img_origin, self.global_step)
            self.logger.experiment.add_image('training/diff_image', vis_diff_img, self.global_step)
            self.logger.experiment.add_image('training/projected_depth_mask', vis_projected_depth_mask, self.global_step)
            self.logger.experiment.add_image('training/tgt_img', vis_tgt_img, self.global_step)
            
            self.log('train/min_depth', min_depth)
            self.log('train/max_depth', max_depth)

    def validation_step(self, batch, batch_idx):

        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = self.inference_depth(tgt_img)
            errs = compute_errors(gt_depth, tgt_depth, self.hparams.hparams.dataset_name)

            errs = {'abs_diff': errs[0], 'abs_rel': errs[1], 'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs, poses, poses_inv = batch
            tgt_depth, ref_depths = self(tgt_img, ref_imgs)
            loss_1, loss_2, generated_img, _, _, _ = photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, self.calib_cam, poses, poses_inv, self.origin_mask, self.hparams.hparams)
            errs = {'photo_loss': loss_1.item(), 'geometry_loss': loss_2.item()}
        else:
            print('wrong validation mode')

        if self.global_step < 10:
            return errs

        # plot
        if batch_idx % 3 == 0:
            vis_img = visualize_image(tgt_img[0])  # (3, H, W)
            vis_depth, _, _ = visualize_depth(tgt_depth[0, 0], origin_mask=self.origin_mask)  # (3, H, W)
            vis_generated_img = visualize_image(generated_img[0])
            stack = torch.cat([vis_img, vis_depth, vis_generated_img], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images('val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)

        return errs

    def validation_epoch_end(self, outputs):

        if self.hparams.hparams.val_mode == 'depth':
            mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in outputs]).mean()

            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)

        elif self.hparams.hparams.val_mode == 'photo':
            mean_pl = np.array([x['photo_loss'] for x in outputs]).mean()
            self.log('val_loss', mean_pl, prog_bar=True)
