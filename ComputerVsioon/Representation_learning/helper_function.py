
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
from dataset import custom_collate_fn





def extract_flat_features(rep_model):
    ####use the cell pose encoder to extract cell feature
    device = 'cuda'
    # model = utils.CellImageEncoder()
    # model = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/represent_model_epoch20.pt", weights_only=False)
    rep_model.to(device)
    rep_model.eval()
    all_features = []
    all_labels = []
    train_ds = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_val_v1.pt", weights_only=False)
    train_df = DataLoader(train_ds, batch_size= 128, shuffle=False, collate_fn = custom_collate_fn)
    with torch.no_grad():
        for imgs, _, _, labels, file_name in tqdm(train_df, desc="Extracting features"):
            imgs = imgs.repeat(1, 1, 1, 1).to(device)
            feats = rep_model(imgs)  # output: (B, 256, 8, 8)
            # feats = feats.view(feats.size(0), -1)  # flatten to (B, 16384)
            # print(labels)
            # print(type(labels))
            all_features.append(feats.cpu())
            # all_labels.extend([label[-1] for label in file_name]) ## for getting device name
            # print(all_labels)
            # break
            all_labels.extend(list(labels))
            # break
    all_features = torch.cat(all_features).numpy()
    all_labels = np.array(all_labels)
    return all_features, all_labels

def tsne_and_plot(features, labels, epoch = 20, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["x"] = tsne_result[:, 0]
    df["y"] = tsne_result[:, 1]
    df["Id"] = [idx for idx in range(len(labels))]
    df["label"] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", alpha=0.7)
    plt.title("t-SNE of Cellpose Encoder Features")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/res18_encoder_new_setup_all_flatten_tsne_scatter_augmented_epoch{epoch}_new_data_val.png")
    plt.show()
    # fig = px.scatter(df, x='x', y='y', color='label', hover_data = ['Id'], title="t-SNE Visualization of Cell Image Embeddings")
    # fig.write_html(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/tsne_plot_augmented_epoch{epoch}.html")
    return tsne_result

def tsne_and_plot_without_epoch(features, labels, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(features)

    df = pd.DataFrame()
    df["x"] = tsne_result[:, 0]
    df["y"] = tsne_result[:, 1]
    df["Id"] = [idx for idx in range(len(labels))]
    df["label"] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", alpha=0.7)
    plt.title("t-SNE of Cellpose Encoder Features")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/napab_affected_non_affected_embeddings.png")
    plt.show()
    # fig = px.scatter(df, x='x', y='y', color='label', hover_data = ['Id'], title="t-SNE Visualization of Cell Image Embeddings")
    # fig.write_html(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/tsne_plot_augmented_epoch{epoch}.html")
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
        # 'TSNE-3': tsne_features[:, 2],
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
    # # TSNE Dimension 3
    # plt.subplot(2, 2, 3)
    # sns.violinplot(x='Patient', y='TSNE-3', data=df, inner='box')
    # plt.xticks(rotation=45)
    # plt.title('t-SNE Dimension 3')

    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/custom_encoder_allflatten_tsne_violin_augmented_epoch20.png")
    plt.show()

import torch
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import umap
from sklearn.preprocessing import StandardScaler

def extract_flat_features_with_time_wise_dataset():
    ####use the cell pose encoder to extract cell feature
    device = 'cuda'
    rep_model = utils.CellImageEncoder()
    rep_model.load_state_dict(torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/model_epoch32_sep1.8459.pt", weights_only=True))
    rep_model.to(device)
    all_features = []
    all_labels = []
    all_days = []
    napab_affected_df = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_affected_cleaned.pt", weights_only=False)
    napab_affected_loader = DataLoader(napab_affected_df, batch_size= 128, shuffle=False)
    napab_control_df = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/napab_control_cleaned.pt", weights_only=False)
    napab_control_loader = DataLoader(napab_control_df, batch_size= 128, shuffle=False)
    with torch.no_grad():
        for imgs, _, labels, exp_day, _ in tqdm(napab_affected_loader, desc="Extracting features"):
            imgs = imgs.repeat(1, 1, 1, 1).to(device)
            feats = rep_model(imgs)  # output: (B, 256, 8, 8)
            # feats = feats.view(feats.size(0), -1)  # flatten to (B, 16384)
            # print(labels)
            # print(type(labels))
            all_features.append(feats.cpu())
            all_labels.extend(list(labels))
            all_days.extend(list(exp_day))
            # break
        for imgs, _, labels, exp_day, _ in tqdm(napab_control_loader, desc="Extracting features"):
            imgs = imgs.repeat(1, 1, 1, 1).to(device)
            feats = rep_model(imgs)  # output: (B, 256, 8, 8)
            # feats = feats.view(feats.size(0), -1)  # flatten to (B, 16384)
            # print(labels)
            # print(type(labels))
            all_features.append(feats.cpu())
            all_labels.extend(list(labels))
            all_days.extend(list(exp_day))
    
    all_features = torch.cat(all_features).numpy()
    all_labels = np.array(all_labels)
    all_days = np.array([int(i) for i in all_days])
    print(all_days[:6])
    reducer = umap.UMAP()
    scaled = StandardScaler().fit_transform(all_features)
    embedding = reducer.fit_transform(scaled)
    df = pd.DataFrame({
        'TSNE-1': embedding[:, 0],
        'TSNE-2': embedding[:, 1],
        # 'TSNE-3': tsne_features[:, 2],
        'exp_day': all_days,
        'Patient': all_labels
    })
    print(embedding.shape)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="TSNE-1", y="TSNE-2", hue="exp_day", palette="viridis", alpha=0.7)
    plt.title("t-SNE of Cellpose Encoder Features")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/napab_affected_non_affected_embeddings_umap.png")
    
    return all_features, all_labels


# all_features, all_labels = extract_flat_features_with_time_wise_dataset()
# tsne_features = tsne_and_plot_without_epoch(all_features, all_labels)
# model = utils.CellImageEncoder()
# model = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/represent_model_epoch20.pt", weights_only=False)
# model.load_state_dict(torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/custom_encoder_new_setup_model_epoch32_sep2.1850_new_data.pt"))

# features, labels = extract_flat_features(model)
# print("getting fraturs done")
# tsne_features = tsne_and_plot(features, labels)
# plot_tsne_violin(tsne_features, labels)
