import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image


def visualize_inference(depth, origin_mask, cmap=cv2.COLORMAP_HSV):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    
    mask = origin_mask.cpu().numpy()[0] > 0.5
    inv_mask = origin_mask.cpu().numpy()[0] < 0.5
    x[inv_mask] = 0.0
    
    x = np.nan_to_num(x)  # change nan to 0
    # x[x > 50] = 50
    
    mi = np.min(x[mask])  # get minimum depth
    ma = np.max(x[mask])
    
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, mi, ma
    

def visualize_image(image):
    """
    tensor image: (3, H, W)
    """
    x = (image.cpu() * 0.225 + 0.45)
    return x


def visualize_depth(depth, origin_mask, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    
    mask = origin_mask.cpu().numpy()[0] > 0.5
    inv_mask = origin_mask.cpu().numpy()[0] < 0.5
    x[inv_mask] = 0.0
    
    x = np.nan_to_num(x)  # change nan to 0
    # x[x > 50] = 50
    
    mi = np.min(x[mask])  # get minimum depth
    ma = np.max(x[mask])
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, mi, ma


def visualize_training_depth(depth, origin_mask, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.detach().cpu().numpy()
    inv_mask = origin_mask.cpu().numpy()[0] < 0.5
    mask = origin_mask.cpu().numpy()[0] > 0.5
    
    x[inv_mask] = 0.0
    x = np.nan_to_num(x)
        
    mi = np.min(x[mask])  # get minimum depth
    ma = np.max(x[mask])
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x = cv2.applyColorMap(x, cmap)
    x_ = Image.fromarray(x)
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, mi, ma 


def visualize_training_mask(mask, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = mask.detach().cpu().numpy()
    x = np.nan_to_num(x)

    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    x_ = Image.fromarray(x)
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_
