import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from cellpose import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torch import nn
from torchvision.models import vgg16, VGG16_Weights, efficientnet_b1, EfficientNet_B1_Weights
import torch
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

def compute_class_separability(embeddings, labels, max_pairs=2000):
    """
    Compute class separability score = inter / intra
    Args:
        embeddings (Tensor): [N, D]
        labels (Tensor): [N]
        max_pairs (int): max number of pairs to use per category for memory efficiency
    Returns:
        dict with intra_class_dist, inter_class_dist, separability_score
    """
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    class_to_embeddings = defaultdict(list)

    for emb, label in zip(embeddings, labels):
        class_to_embeddings[int(label.item())].append(emb)

    intra_dists = []
    inter_dists = []

    # Intra-class distances
    for emb_list in class_to_embeddings.values():
        if len(emb_list) < 2:
            continue
        pairs = list(combinations(emb_list, 2))
        random.shuffle(pairs)
        for emb1, emb2 in pairs[:max_pairs]:
            dist = torch.norm(emb1 - emb2).item()
            intra_dists.append(dist)

    # Inter-class distances
    class_items = list(class_to_embeddings.items())
    for i in range(len(class_items)):
        for j in range(i + 1, len(class_items)):
            emb_list_i = class_items[i][1]
            emb_list_j = class_items[j][1]
            sampled_i = random.sample(emb_list_i, min(len(emb_list_i), 10))
            sampled_j = random.sample(emb_list_j, min(len(emb_list_j), 10))
            for emb1 in sampled_i:
                for emb2 in sampled_j:
                    dist = torch.norm(emb1 - emb2).item()
                    inter_dists.append(dist)

    intra_mean = np.mean(intra_dists) if intra_dists else float('inf')
    inter_mean = np.mean(inter_dists) if inter_dists else 0.0
    score = inter_mean / (intra_mean + 1e-8)

    return {
        'intra_class_dist': intra_mean,
        'inter_class_dist': inter_mean,
        'separability_score': score
    }


class CellPoseClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(CellPoseClassifier, self).__init__()
        self.encoder = encoder
        self.freeze_encoder_layers()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def freeze_encoder_layers(self):
        # Freeze encoder blocks (not neck)
        for name, param in self.encoder.named_parameters():
            if 'blocks' in name or 'patch_embed' in name:
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)  # Output from encoder.neck: [B, 256, 8, 8]
        pooled = self.global_pool(features)  # [B, 256, 1, 1]
        out = self.classifier(pooled)
        return out
    
from cellpose import models
model = models.CellposeModel(gpu=True)
model = model.net
cellpose_encoder = model.encoder
model_cellpose_clf = CellPoseClassifier(encoder=cellpose_encoder, num_classes=2)
model_cellpose_clf = model_cellpose_clf.to(torch.float32)
original_pos_embed = model_cellpose_clf.encoder.pos_embed
# Resize positional embeddings to [1, 8, 8, 1024]
pos_embed_resized = torch.nn.functional.interpolate(
    original_pos_embed.permute(0, 3, 1, 2),  # [1, C, H, W]
    size=(8, 8),
    mode='bilinear',
    align_corners=False
).permute(0, 2, 3, 1)  # back to [1, H, W, C]

# Replace in model
model_cellpose_clf.encoder.pos_embed = torch.nn.Parameter(pos_embed_resized)




#resnet = resnet18(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet = resnet50()
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = resnet.fc.in_features
out_ftrs = resnet.fc.out_features
resnet.fc = nn.Linear(num_ftrs, 512)

class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        self.model = model
        self.clf = nn.Sequential(nn.ReLU(), nn.Linear(512, 10))
        #self.sig = 

    def forward(self, x):
        x = self.model(x)
        x = self.clf(x)
        # x = torch.nn.Sigmoid(x)
        return x
        # return x

model = ResNet(resnet)



# vgg = vgg16(VGG16_Weights.IMAGENET1K_V1)
vgg = vgg16()
vgg.classifier[6] = nn.Linear(4096, 16, bias = True)
vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        self.model = model
        self.clf = nn.ModuleList([nn.ReLU(), nn.Linear(512, 128),
                                  nn.ReLU(), nn.Linear(128, 32),
                                  nn.ReLU(), nn.Linear(32, 1)])
        #self.sig = 
        self.clf1 = nn.ModuleList([nn.ReLU(), nn.Linear(16, 1)])

    def forward(self, x):
        x = self.model(x)
        x = self.clf1[0](x)
        x = self.clf1[1](x)
        # for i in self.clf:
        #     x = i(x)
        # x = torch.nn.Sigmoid(x)
        return x
model_vgg19 = VGG(vgg)

model_eff_b = efficientnet_b1(weights= EfficientNet_B1_Weights.IMAGENET1K_V2)
model_eff_b.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
model_eff_b.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
