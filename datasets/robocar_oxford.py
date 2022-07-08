import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
from scipy.spatial.transform import Rotation as R
import cv2


def load_as_float(path):
    return imread(path).astype(np.float32)


def build_se3_transform(xyz_xyzw):
    if len(xyz_xyzw) != 7:
        raise ValueError("Must supply 6 values to build transform")

    xyz_xyzw = [float(i) for i in xyz_xyzw]
    r = R.from_quat([xyz_xyzw[3], xyz_xyzw[4], xyz_xyzw[5], xyz_xyzw[6]])
    translation = np.array([xyz_xyzw[0], xyz_xyzw[1], xyz_xyzw[2]]).reshape((3, 1))

    rotation = r.as_matrix()
    transform = np.identity(4)
    transform[:3, :3] = rotation
    transform[:3, 3][:, np.newaxis] = translation

    return transform


class OxfordDataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='robocar'):
        np.random.seed(0)
        random.seed(0)
        self.root = Path(root)
        scene_list_path = self.root / 'base_pose.txt' if train else self.root / 'val.txt'
        self.scenes = [self.root / 'mono_right' / folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)

        current_file_path = os.getcwd()
        ext_path = current_file_path + '/configs/calib/ext.yaml'
        fs = cv2.FileStorage()
        fs.open(ext_path, cv2.FILE_STORAGE_READ)
        ext = fs.getNode('cam0').mat()

        last_tgt_pose = np.identity(4, np.float32)
        
        image_paths = []
        base_poses = []
        for idx, scene in enumerate(self.scenes):
            current_pose = np.linalg.inv(ext) @ build_se3_transform(scene.split()[1:]) @ ext
            current_image_path = scene.split()[0] + '.png'
            if idx == 0:
                image_paths.append(current_image_path)
                base_poses.append(current_pose)
                last_tgt_pose = np.copy(current_pose)
            else:
                diff_pose = np.linalg.inv(last_tgt_pose) @ current_pose
                trans_part = diff_pose[:3, 3]
                dist = np.linalg.norm(trans_part)
                
                if dist > 0.3:
                    image_paths.append(current_image_path)
                    base_poses.append(current_pose)
                    last_tgt_pose = np.copy(current_pose)

        imgs = sorted(image_paths)
        for i in range(demi_length * self.k, len(imgs) - demi_length * self.k):
            sample = {'tgt': imgs[i], 'ref_imgs': [], 'Trt': [], 'Ttr': []}
            for j in shifts:
                sample['ref_imgs'].append(imgs[i + j])
                
                pose = np.linalg.inv(base_poses[i + j]) @ base_poses[i]
                sample['Trt'].append(pose.astype(np.float32))
                
                pose_inv = np.linalg.inv(base_poses[i]) @ base_poses[i + j]
                sample['Ttr'].append(pose_inv.astype(np.float32))

            sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs = self.transform([tgt_img] + ref_imgs)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        poses_Trt = sample['Trt']
        poses_Ttr = sample['Ttr']

        return tgt_img, ref_imgs, poses_Trt, poses_Ttr

    def __len__(self):
        return len(self.samples)
