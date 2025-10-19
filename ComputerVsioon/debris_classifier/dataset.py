from torchvision.io import read_image
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from skimage.io import imread
from skimage.measure import regionprops, label
import numpy as np
import os
import h5py
from PIL import Image
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import defaultdict
from itertools import combinations
import random
from torch.utils.data import DataLoader
import pandas as pd

import pandas as pd
from skimage import measure, filters, morphology
from tqdm import tqdm
from torch import nn

print(torch.cuda.is_available())

def circles(image):
    cv2.HoughCircles(image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=10,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30)

class normal_ds(Dataset):
    def __init__(self, imgs, labels, transformation = None):
        super(normal_ds, self).__init__()
        self.imgs = np.array(imgs) 
        self.clf_labels = torch.Tensor(labels) 
        self.transf = transformation
    def __len__(self):
        return self.imgs.shape[0]
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        lab = self.clf_labels[idx]
        if self.transf:
            img = self.transf(image = img)['image']
        return torch.from_numpy(img).unsqueeze(0), lab
    


class customdataset(Dataset):
    def __init__(self, img_paths, labels, transform = None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = read_image(self.img_paths[index])[0]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img.numpy())
        img = img.repeat(3, 1, 1)
        return img, label
    def __len__(self):
        return len(self.img_paths)
        

import numpy as np
import cv2

def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Make sure the input is uint8
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

class rescale_pad(object):
    def __init__(self, target_w, target_h):
        self.target_w = target_w
        self.target_h = target_h
        self.sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
    
    def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Make sure the input is uint8
        if gray_img.dtype != np.uint8:
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_img)
    
    def __call__(self, sample):
        # sample = apply_clahe(sample)
        # sample = cv2.filter2D(sample, -1, self.sharpen_kernel)
        h, w = sample.shape[:2]
        scale = min(self.target_h/h, self.target_w/w)
        new_h, new_w = int(h*scale), int(w*scale)
        resized = cv2.resize(sample, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # Padding
        delta_h = self.target_h - new_h
        delta_w = self.target_w - new_w
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = 0
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded

class rescale_pad_no_pad(object):
    def __init__(self, target_w, target_h):
        self.target_w = target_w
        self.target_h = target_h
        self.sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
    
    def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Make sure the input is uint8
        if gray_img.dtype != np.uint8:
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_img)
    
    def __call__(self, sample):
        # sample = apply_clahe(sample)
        # sample = cv2.filter2D(sample, -1, self.sharpen_kernel)
        h, w = sample.shape[:2]
        scale = min(self.target_h/h, self.target_w/w)
        resized = cv2.resize(sample, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR if scale < 1 else cv2.INTER_CUBIC)
        return resized



composed_rescale_augment = transforms.Compose([rescale_pad(64, 64),
                               transforms.ToPILImage(),  # Needed for torchvision transforms below
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(degrees=10),
                                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                               transforms.ToTensor()])

composed_rescale = transforms.Compose([rescale_pad_no_pad(64, 64),
                               transforms.ToTensor()])


import mahotas

def extract_haralick_features(image):
    # Assumes image is grayscale uint8
    features = mahotas.features.haralick(image).mean(axis=0)  # average across directions
    return features

from skimage.feature import local_binary_pattern
import numpy as np

def extract_lbp_histogram(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2),
                             density=True)
    return hist  # normalized histogram


def extract_morphological_features(img):
    """
    Extracts relevant morphological features from a grayscale image of a single cell.
    
    Parameters:
        img (np.ndarray): Grayscale image of a cell.
    
    Returns:
        features (dict): A dictionary containing morphological feature names and values.
    """
    features = {}

    # Thresholding
    thresh_val = filters.threshold_otsu(img)
    binary = img > thresh_val

    # Morphological cleanup
    binary = morphology.remove_small_objects(binary, min_size=20)
    binary = morphology.binary_closing(binary, morphology.disk(3))

    # Label regions
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img, intensity_image=img)

    if not regions:
        return None  # Skip if no region is detected

    region = max(regions, key=lambda r: r.area)  # Focus on the largest component

    # Extract features
    features["area"] = region.area
    features["perimeter"] = region.perimeter
    features["eccentricity"] = region.eccentricity
    features["mean_intensity"] = region.mean_intensity
    features["solidity"] = region.solidity
    features["circularity"] = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-5)
    features["gradient"] = np.mean(np.gradient(img.astype(float)))

    return features



def classify_cell(features):
    alive_signs = 0

    # if features["area"] > 400:
    #     alive_signs += 1
    if features["perimeter"] > 90:
        alive_signs += 1
    if features["solidity"] > 0.9:
        alive_signs += 1
    if features["eccentricity"] < 0.7:
        alive_signs += 1
    # if features["gradient"] < 3.0:
    #     alive_signs += 1
    if features["mean_intensity"] > 110:
        alive_signs += 1
    if features["circularity"] > 0.8:
        alive_signs += 1
    if alive_signs <= 3:
        return ("dead", alive_signs)
    elif alive_signs == 2:
        return ("ambiguous", alive_signs)
    else:
        return ("alive", alive_signs)

def strict_alive_filter(features):
    return (
       features['solidity'] > 0.85 and
    #    features['circularity'] > 0.75 and
        0.1 < features['eccentricity'] < 0.65 and
        1300 < features['area'] < 4500 and
        features['gradient'] > 0.03
    )



def extract_cells_from_image(image, mask, width_thrshold, height_threshold, cell_label, exp_day):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    cell_images = []
    morphs = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cell = image[minr:maxr, minc:maxc]
        if cell.shape[0] >= width_thrshold and cell.shape[1] >= height_threshold:
            morph_features = extract_morphological_features(cell)
            haralick_feats = extract_haralick_features(cell)
            lbp_feats = extract_lbp_histogram(cell)
            
            morphs.append(np.concatenate([np.array([exp_day]),np.array(list(morph_features.values())), haralick_feats, lbp_feats]))
            if cell_label == True:
                cell_images.append(cell)
            #     if strict_alive_filter(morph_features) == True:
            #         cell_images.append(cell)
            if cell_label == False: 
                if strict_alive_filter(morph_features) == True: ###drop out cells with high chance of being alive among dead cell dataset
                    continue
                else:
                    cell_images.append(cell)
            
        
    return (cell_images, morphs)

class CellDataset(Dataset):
    def __init__(self, filtered_csv, width_threshold, heigth_threshold, cell_label, transform=None):
        self.transform = transform
        self.filetered_metadata = filtered_csv
        self.cell_data = []  # list of (cell_image, optional_info)
        self.morphs_data = []
        self.pat_names = []

        for index, row in self.filetered_metadata.iterrows():
            device = row['device']
            vessel_id = row['vessel_id']
            file_name = row['file_name']
            time_step = row['time_step']
            pat_name = row['patient_name']
            exp_day = int(float(row['exp_day']))
            try:
                    mask_path = f"/scratch-grete/projects/nim00007/data/pdo/data/pdo/segmentations/{device}/vessel_{vessel_id}/{file_name.split('.')[0]}.h5"
                    img_path = f"/scratch-grete/projects/nim00007/data/pdo/data/pdo/images/{device}/vessel_{vessel_id}/{file_name.split('.')[0]}.h5"
                    with h5py.File(mask_path, "r") as f:
                        # print(f.keys())
                        g = f['cellpose/v1']
                        last_tp = list(g.keys())
                        mask = g[time_step][:]
                    with h5py.File(img_path, 'r') as f:
                        g = f['timepoints']
                        last_tp = list(g.keys())
                        img = g[time_step][:]
                        #print(last_tp)
                    cells, morphs = extract_cells_from_image(img, mask, width_threshold, heigth_threshold, cell_label, exp_day)
                    self.pat_names += [pat_name] * len(cells)
                    self.cell_data.extend(cells)
                    self.morphs_data.extend(morphs)
            except Exception as error:
                    print(error)
                    continue
            
        self.cell_label = cell_label

    def __len__(self):
        return len(self.cell_data)

    def __getitem__(self, idx):
        cell = self.cell_data[idx]
        cell_label = self.cell_label
        morph = self.morphs_data[idx]
        pat_name = self.pat_names[idx]

        if self.transform:
            cell = self.transform(cell)

        return cell, morph, cell_label, pat_name
    
    
class CellDataset_longitude(Dataset):
    def __init__(self, filtered_csv, width_threshold, heigth_threshold, cell_label, transform=None):
        self.transform = transform
        self.filetered_metadata = filtered_csv
        self.cell_data = []  # list of (cell_image, optional_info)
        self.morphs_data = []
        self.pat_names = []
        self.exp_days = []
        self.file_names = []

        for index, row in self.filetered_metadata.iterrows():
            device = row['device']
            vessel_id = row['vessel_id']
            file_name = row['file_name']
            time_step = row['time_step']
            pat_name = row['patient_name']
            exp_day = int(float(row['exp_day']))
            try:
                    mask_path = f"/scratch-grete/projects/nim00007/data/pdo/data/pdo/segmentations/{device}/vessel_{vessel_id}/{file_name.split('.')[0]}.h5"
                    img_path = f"/scratch-grete/projects/nim00007/data/pdo/data/pdo/images/{device}/vessel_{vessel_id}/{file_name.split('.')[0]}.h5"
                    with h5py.File(mask_path, "r") as f:
                        # print(f.keys())
                        g = f['cellpose/v1']
                        last_tp = list(g.keys())
                        mask = g[time_step][:]
                    with h5py.File(img_path, 'r') as f:
                        g = f['timepoints']
                        last_tp = list(g.keys())
                        img = g[time_step][:]
                        #print(last_tp)
                    cells, morphs = extract_cells_from_image(img, mask, width_threshold, heigth_threshold, cell_label, exp_day)
                    self.pat_names += [pat_name] * len(cells)
                    self.cell_data.extend(cells)
                    self.morphs_data.extend(morphs)
                    self.exp_days += [exp_day] * len(cells)
                    self.file_names += [[file_name, time_step, vessel_id, device]] * len(cells)
            except Exception as error:
                    print(error)
                    continue
            
        self.cell_label = cell_label

    def __len__(self):
        return len(self.cell_data)

    def __getitem__(self, idx):
        cell = self.cell_data[idx]
        cell_label = self.cell_label
        morph = self.morphs_data[idx]
        pat_name = self.pat_names[idx]
        exp_day = self.exp_days[idx]
        file_name = self.file_names[idx]

        if self.transform:
            transformed_cell = self.transform(cell)
            cell = torch.from_numpy(cell)
        else:
            cell = torch.from_numpy(cell)
            transformed_cell = cell

        return cell, transformed_cell, morph, cell_label, exp_day, pat_name, file_name

class CellImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CellImageEncoder, self).__init__()

        # --- Initial Low-level Texture + Edge Detectors ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 64, 64] -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- Multi-Scale (Inception-like) Block ---
        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=1, padding=0),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=5, padding=2),   # [B, 32, 64, 64]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),   # fuse channels → [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- Residual Block with Attention ---
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, 64, 1, 1]
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Global Pooling and Projection to Embedding ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)   # → [B, 64, 1, 1]
        self.projector = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)  # low-level
        multi_scale_feats = [F.relu(conv(x)) for conv in self.conv_multi_scale]
        x = torch.cat(multi_scale_feats, dim=1)
        x = self.fusion(x)

        # Residual attention block
        residual = self.res_block(x)
        attn = self.attention(x)
        x = x + residual * attn

        # Global pooling and embedding
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 64]
        x = self.projector(x)  # [B, embedding_dim]
        x = F.normalize(x, dim=1)  # normalize for contrastive learning
        return x

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

class enc_cls(nn.Module):
    def __init__(self, enc, clf_header):
        super(enc_cls, self).__init__()
        self.enc = enc
        self.clf_head = clf_header
    def forward(self, x):
        x = self.enc(x)
        x = self.clf_head(x)
        return x    

strong_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.7, alpha=250, sigma=250 * 0.05),
        A.GaussNoise(p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.OneOf([
            A.MedianBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
        ], p=0.5),
    ])



