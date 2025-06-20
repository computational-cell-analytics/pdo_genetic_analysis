from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import v2, Resize
from torchvision.transforms.functional import resize
import cv2
from glob import glob
import utils as models
import torch.optim as optim
import losses
from torchvision.models import vgg19, VGG19_Weights
from matplotlib import pyplot as plt
from utils import VGG
from data_load import custom_dataset, cv_2_transforms, normalise_resize


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def visualize_latent_distribution_by_label(encoder, dataloader, device, latent_dim=350, n_batches=15):
    encoder.eval()
    z_all = []
    labels_all = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            # if i >= n_batches:
            #     break
            imgs = imgs.to(device)
            z = encoder(imgs).cpu()
            z_all.append(z)
            labels_all.append(labels)

    z_all = torch.cat(z_all, dim=0)  # [N, latent_dim]
    labels_all = torch.cat(labels_all, dim=0)  # [N]
    z_flat = z_all.view(-1).numpy()
    labels_repeat = labels_all.repeat_interleave(latent_dim).numpy()

    plt.figure(figsize=(10, 6))
    sns.histplot(x=z_flat, hue=labels_repeat, bins=100, stat='density', element='step', palette='tab10')
    
    # Overlay standard normal
    x = torch.linspace(-4, 4, 1000).numpy()
    plt.plot(x, norm.pdf(x), 'k--', label='N(0,1)')
    plt.title('Flattened Latent Distribution by Label')
    plt.xlabel('z value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("encod_sush_3_wdecay_weight_warmup_all_losses_latedist.png")



def tsne_latent_plot(encoder, dataloader, device, n_batches=5):
    encoder.eval()
    z_all = []
    labels_all = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            # if i >= n_batches:
            #     break
            imgs = imgs.to(device)
            z = encoder(imgs).cpu()
            z_all.append(z)
            labels_all.append(labels)

    z_all = torch.cat(z_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0).numpy()
    
    z_2d = TSNE(n_components=2, perplexity=30).fit_transform(z_all.numpy())
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels_all, palette='tab10', s=40, alpha=0.8)
    plt.title('t-SNE Projection of Latent Space by Label')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("encod_sush_3_wdecay_weight_warmup_all_losses_tsne.png")
    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

all_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/**/NonDemented/*.jpg", recursive = True)
all_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/**/VeryMildDemented/*.jpg", recursive = True)
all_paths = all_no_paths + all_very_paths
all_paths = np.array(all_paths)
labels = np.array(['Non' in path.split('/')[-2] for path in all_paths])

idx = np.arange(all_paths.shape[0])
np.random.shuffle(idx)
all_paths = all_paths[idx]
labels = labels[idx]
batch_size = 64

all_img_normal_dataset = custom_dataset(all_paths, labels, normalise_resize)
train_loader = DataLoader(all_img_normal_dataset, batch_size = 64, shuffle = True)
encod = torch.load("/user/sina.garazhian/u12203/DISCOWER/encod_Sush_6_wdecay_weight_warmup_clf_mean_cov_subset_v3.pt", map_location = device, weights_only = False)

visualize_latent_distribution_by_label(encod, train_loader, device)
tsne_latent_plot(encod, train_loader, device)
