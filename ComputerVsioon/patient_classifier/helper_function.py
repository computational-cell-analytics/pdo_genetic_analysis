
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from cellpose import models
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import CellDataset_longitude

def custom_collate_fn(batch):
    # 'batch' is a list of tuples like [(rescaled_img1, non_rescaled_img1), (rescaled_img2, non_rescaled_img2), ...]
    # We want to stack only the first element of each tuple.
    rescaled_images = torch.stack([item[1] for item in batch])
    labels = torch.LongTensor([item[3] for item in batch])
    exp_days = torch.LongTensor([item[4] for item in batch])
    pat_names = [item[5] for item in batch]
    file_names = [item[6] for item in batch]
    # print(len(rescaled_images))
    return [rescaled_images, labels, exp_days, pat_names, file_names]


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




def filtering_based_on_debris_model():

    train_ds = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_metadata_train_v1.pt", weights_only=False)
    val_ds = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_metadata_val_v1.pt", weights_only=False)
    test_ds = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_metadata_test_v1.pt", weights_only=False)
    model_clf = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/debris_classifier/cellpose_classifier_40_1_v1.pt", weights_only=False, map_location=torch.device('cpu'))


    device = 'cuda'
    train_df = DataLoader(train_ds, batch_size= 128, shuffle=False, collate_fn=custom_collate_fn)
    val_df = DataLoader(val_ds, batch_size= 128, shuffle=False, collate_fn = custom_collate_fn)
    test_df = DataLoader(test_ds, batch_size= 128, shuffle=False, collate_fn = custom_collate_fn)


    model_clf.to(device)
    model_clf.eval()
    

    train_idx = []
    val_idx = []
    test_idx = []

    for bat in tqdm(train_df):
        imgs, _, _, _, _ = bat
        imgs = imgs.to(device)
        preds = model_clf(imgs.repeat(1, 3, 1, 1))
        with torch.no_grad():
            train_idx.extend(((preds[:,1].cpu() >= 0.5) & (preds[:, 0] <= -1).cpu()))
            
    for bat in tqdm(val_df):
        imgs, _, _, _, _ = bat
        imgs = imgs.to(device)
        preds = model_clf(imgs.repeat(1, 3, 1, 1))
        with torch.no_grad():
            val_idx.extend(((preds[:,1].cpu() >= 0.5) & (preds[:, 0] <= -1).cpu()))

    for bat in tqdm(test_df):
        imgs, _, _, _, _ = bat
        imgs = imgs.to(device)
        preds = model_clf(imgs.repeat(1, 3, 1, 1))
        with torch.no_grad():
            test_idx.extend(((preds[:,1].cpu() >= 0.5) & (preds[:, 0] <= -1).cpu()))


    with open("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/train_40_idx_metadata_v1.txt",'w') as f:
        for id in train_idx:
            f.write(f"{id},")
    with open("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/val_40_idx_metadata_v1.txt",'w') as f:
        for id in val_idx:
            f.write(f"{id},")
    with open("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/test_40_idx_metadata_v1.txt",'w') as f:
        for id in test_idx:
            f.write(f"{id},")
        
        


def extract_flat_features():
    ####use the cell pose encoder to extract cell feature
    device = 'cuda'
    model = utils.model
    model.to(device)
    all_features = []
    all_labels = []
    train_ds = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_cleaned_test.pt", weights_only=False)
    train_df = DataLoader(train_ds, batch_size= 128, shuffle=False)
    with torch.no_grad():
        for imgs, _, _, labels in tqdm(train_df, desc="Extracting features"):
            imgs = imgs.repeat(1, 3, 1, 1).to(device)
            feats = model(imgs)  # output: (B, 256, 8, 8)
            # feats = feats.mean(dim=(2, 3)) ### for cellpose
            # feats = feats.view(feats.size(0), -1)  # flatten to (B, 16384)
            # print(labels)
            # print(type(labels))
            all_features.append(feats.cpu())
            all_labels.extend(list(labels))
            # break
    all_features = torch.cat(all_features).numpy()
    all_labels = np.array(all_labels)
    return all_features, all_labels

def tsne_and_plot(features, labels, perplexity=30):
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["x"] = tsne_result[:, 0]
    df["y"] = tsne_result[:, 1]
    df["label"] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", alpha=0.7)
    plt.title("t-SNE of Cellpose Encoder Features")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/resnet_encoder_allflatten_tsne.png")
    # plt.show()
    return tsne_result

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_tsne_violin(tsne_features, labels):
    """
    tsne_features: numpy array of shape (N, 2)
    labels: array-like of N labels (e.g. patient IDs)
    """
    df = pd.DataFrame({
        'TSNE-1': tsne_features[:, 0],
        'TSNE-2': tsne_features[:, 1],
        'TSNE-3': tsne_features[:, 2],
        'Patient': labels
    })

    plt.figure(figsize=(14, 5))

    # TSNE Dimension 1
    plt.subplot(2, 2, 1)
    sns.violinplot(x='Patient', y='TSNE-1', data=df, inner='box')
    plt.xticks(rotation=45)
    plt.title('t-SNE Dimension 1')

    # TSNE Dimension 2
    plt.subplot(2, 2, 2)
    sns.violinplot(x='Patient', y='TSNE-2', data=df, inner='box')
    plt.xticks(rotation=45)
    plt.title('t-SNE Dimension 2')
    # TSNE Dimension 3
    plt.subplot(2, 2, 3)
    sns.violinplot(x='Patient', y='TSNE-3', data=df, inner='box')
    plt.xticks(rotation=45)
    plt.title('t-SNE Dimension 3')

    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/resnet_encoder_allflatten_tsne_violin.png")

    plt.show()

def filtering_based_on_debris_model_for_drug_afected_ds():

    napab_affected_ds = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_affected.pt", weights_only=False)
    napab_control_ds = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_control.pt", weights_only=False)
    model_clf = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/debris_classifier/cellpose_classifier_40_1_v1.pt", weights_only=False, map_location=torch.device('cpu'))


    device = 'cuda'
    napab_affected_df = DataLoader(napab_affected_ds, batch_size= 128, shuffle=False)
    napab_control_df = DataLoader(napab_control_ds, batch_size= 128, shuffle=False)
    


    model_clf.to(device)

    napab_affected_idx = []
    napab_control_idx = []

    for imgs, _, _, _, _ in napab_affected_df:
        imgs = imgs.to(device)
        preds = model_clf(imgs.repeat(1, 3, 1, 1))
        with torch.no_grad():
            napab_affected_idx.extend(preds[:,1].cpu() >= 0.5)
            
    for imgs,_, _, _, _ in napab_control_df:
        imgs = imgs.to(device)
        preds = model_clf(imgs.repeat(1, 3, 1, 1))
        with torch.no_grad():
            napab_control_idx.extend(preds[:,1].cpu() >= 0.5)


    with open("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_affected_40_idx.txt",'w') as f:
        for id in napab_affected_idx:
            f.write(f"{id},")
    with open("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_control_40_idx.txt",'w') as f:
        for id in napab_control_idx:
            f.write(f"{id},")
        

# features, labels = extract_flat_features()
# print("getting fraturs done")
# tsne_features = tsne_and_plot(features, labels)
# plot_tsne_violin(tsne_features, labels)
filtering_based_on_debris_model()
