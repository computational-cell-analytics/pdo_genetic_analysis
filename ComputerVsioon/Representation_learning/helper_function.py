
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







def extract_flat_features(rep_model):
    ####use the cell pose encoder to extract cell feature
    device = 'cuda'
    # model = utils.CellImageEncoder()
    # model = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/represent_model_epoch20.pt", weights_only=False)
    rep_model.to(device)
    all_features = []
    all_labels = []
    train_ds = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_cleaned_test.pt", weights_only=False)
    train_df = DataLoader(train_ds, batch_size= 128, shuffle=False)
    with torch.no_grad():
        for imgs, _, _, labels in tqdm(train_df, desc="Extracting features"):
            imgs = imgs.repeat(1, 1, 1, 1).to(device)
            feats = rep_model(imgs)  # output: (B, 256, 8, 8)
            # feats = feats.view(feats.size(0), -1)  # flatten to (B, 16384)
            # print(labels)
            # print(type(labels))
            all_features.append(feats.cpu())
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
    plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/custom_encoder_all_flatten_tsne_scatter_augmented_epoch{epoch}.png")
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



# model = utils.CellImageEncoder()
# model = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/represent_model_epoch20.pt", weights_only=False)

# features, labels = extract_flat_features(model)
# print("getting fraturs done")
# tsne_features = tsne_and_plot(features, labels)
# plot_tsne_violin(tsne_features, labels)
