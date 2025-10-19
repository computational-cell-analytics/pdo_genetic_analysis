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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import prepare_dataset
import prepare_model

def visualize_latent_distribution_by_label(encoder, dataloader, device, latent_dim=150, n_batches=15):
    encoder.eval()
    z_all = []
    labels_all = []

    with torch.no_grad():
        for i, (imgs, _, labels, _) in enumerate(dataloader):
            # if i >= n_batches:
            #     break
            imgs = imgs.to(device)
            print(imgs.shape)
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
    plt.savefig("mmd_panc_all_losses_latedist_v5.png")



def tsne_latent_plot(encoder, dataloader, device, n_batches=5):
    encoder.eval()
    z_all = []
    labels_all = []

    with torch.no_grad():
        for i, (imgs, _, labels, _) in enumerate(dataloader):
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
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_test_latent_tsne_v1.png")
    plt.show()


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_binary_score(img_clf, latent_clf, encoder, dataloader, device):
    """
    Returns sigmoid probabilities (confidence scores) for binary classification.
    """
    probs_img = []
    probs_latent = []
    with torch.no_grad():
        for i, (imgs, _, labels, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            z = encoder(imgs)
            logits_img_clf = img_clf(imgs)  # shape: [N]
            logits_latent_clf = latent_clf(z[:, :14])  # shape: [N]

            probs_img_clf = torch.sigmoid(logits_img_clf).squeeze()
            probs_latent_clf = torch.sigmoid(logits_latent_clf).squeeze()
            probs_img.append(probs_img_clf)
            probs_latent.append(probs_latent_clf)
        probs_img = torch.cat(probs_img, dim = 0).cpu()
        probs_latent = torch.cat(probs_latent, dim = 0).cpu()
    return (probs_img, probs_latent)

def plot_binary_classification_scores(img_clf, latent_clf, encoder, dataloader, device):
    """
    Plots binary classification score of latent vectors vs real images.
    """
    scores_real, scores_latent = compute_binary_score(img_clf, latent_clf, encoder, dataloader, device)

    plt.figure(figsize=(6, 6))
    plt.scatter(scores_real, scores_latent, alpha=0.6, color='purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # identity line
    plt.xlabel('Real Image Classification Score (sigmoid)')
    plt.ylabel('Latent Subset Classification Score (sigmoid)')
    plt.title('Latent vs Real Image Classification Confidence (Binary)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_test_correlation_v1.png")
    plt.show()





def compute_latent_class_correlations(encoder, clf_model, dataloader, latent_dim, device):
    import warnings

    encoder.eval()
    clf_model.eval()

    all_latents = []
    all_scores = []
    def safe_pearsonr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            warnings.warn("One of the input arrays is constant; Pearson r is undefined.")
            return 0.0, 1.0  # r = 0, p = 1 (no correlation)
        return pearsonr(x, y)
    with torch.no_grad():
        for imgs, _, labels, _ in dataloader:
            imgs = imgs.to(device)
            z = encoder(imgs)  # shape: [B, latent_dim]
            scores = clf_model(imgs).squeeze()  # shape: [B], assuming output is logits

            # Apply sigmoid if not already applied
            scores = torch.sigmoid(scores)

            all_latents.append(z.cpu())
            all_scores.append(scores.cpu())

    # Stack all data
    Z = torch.cat(all_latents, dim=0).numpy()        # shape: [N, latent_dim]
    S = torch.cat(all_scores, dim=0).numpy()         # shape: [N]

    # Compute Pearson correlation for each latent dim
    correlations = []
    for i in range(latent_dim):
        r, _ = safe_pearsonr(Z[:, i], S)
        correlations.append(r)

    correlations = np.array(correlations)
    correlations[np.isnan(correlations)] = 0
    # Plotting
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(latent_dim), correlations)
    plt.xlabel("Latent Feature Index")
    plt.ylabel("Pearson Correlation with IVF-CLF Score")
    plt.title("Latent Feature vs. IVF-CLF Score Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_test_pearson_v1.png")
    plt.show()

    return correlations


# Retry the plot without using torch (since torch import caused an error)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Simulated image generator
def generate_images(base_image, feature_index, std_range=3, steps=7):
    altered_images = []

    for i in range(steps):
        factor = (i - steps // 2) / (steps // 2) * std_range
        altered = np.clip(base_image + factor * 0.05 * (feature_index + 1), 0, 1)
        altered_images.append(altered)
    return altered_images

# SSIM-based counterfactual difference
def compute_diff_img(img1, img2):
    img1_gray = img1 #cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = img2#cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, diff = ssim(img1_gray, img2_gray, full=True)
    return 1 - diff  # difference map

def create_counter_factual(base_image, features):
    # Create base image
    # base_image = np.ones((64, 64, 3)) * 0.5

    # Define latent features
    features = [0, 10, 11, 12, 1]
    fig, axs = plt.subplots(3, len(features), figsize=(15, 6))

    for i, fidx in enumerate(features):
        img_seq = generate_images(base_image, fidx)
        
        # Top: +direction
        axs[0, i].imshow(img_seq[-1])
        axs[0, i].axis("off")
        axs[0, i].set_title(f'Latent #{fidx}')
        
        # Middle: -direction
        axs[1, i].imshow(img_seq[0])
        axs[1, i].axis("off")
        
        # Bottom: 1 - SSIM (difference)
        diff_img = compute_diff_img(img_seq[0], img_seq[-1])
        axs[2, i].imshow(diff_img, cmap='hot')
        axs[2, i].axis("off")

    plt.suptitle("Counterfactual Visualization (DISCOWER Style)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim




def visualize_counterfactual_changes(img, encoder, decoder, stds, selected_features, std_range=3.0):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim

    def compute_diff_img(img1, img2):
        # Use only one channel since all 3 are identical
        if img1.ndim == 3 and img1.shape[2] == 3:
            img1 = img1[:, :, 0]
            img2 = img2[:, :, 0]
        score, diff = ssim(img1, img2, data_range=1.0, full=True)
        return 1 - diff

    encoder.eval()
    decoder.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = img.to(device)

    with torch.no_grad():
        z_orig = encoder(img.unsqueeze(0)).squeeze(0)

        img_orig = decoder(z_orig.unsqueeze(0)).squeeze().cpu().permute(1, 2, 0).numpy()

        plus_imgs, minus_imgs, plus_diffs, minus_diffs = [], [], [], []

        for feat in selected_features:
            z_plus = z_orig.clone()
            z_minus = z_orig.clone()

            z_plus[feat] += std_range * stds[feat]
            z_minus[feat] -= std_range * stds[feat]

            img_plus = decoder(z_plus.unsqueeze(0)).squeeze().cpu().permute(1, 2, 0).numpy()
            img_minus = decoder(z_minus.unsqueeze(0)).squeeze().cpu().permute(1, 2, 0).numpy()

            diff_plus = compute_diff_img(img_orig, img_plus)
            diff_minus = compute_diff_img(img_orig, img_minus)

            plus_imgs.append(img_plus)
            minus_imgs.append(img_minus)
            plus_diffs.append(diff_plus)
            minus_diffs.append(diff_minus)

        # Plot: original, plus, minus, diff+
        fig, axs = plt.subplots(5, len(selected_features), figsize=(4 * len(selected_features), 9))

        for i in range(len(selected_features)):
            axs[0, i].imshow(img_orig, cmap='gray'); axs[0, i].set_title("Original"); axs[0, i].axis("off")
            axs[1, i].imshow(plus_imgs[i], cmap='gray'); axs[1, i].set_title(f"Z+ in {selected_features[i]}"); axs[1, i].axis("off")
            axs[2, i].imshow(plus_diffs[i], cmap='hot'); axs[3, i].set_title("Diff(+)"); axs[3, i].axis("off")
            axs[3, i].imshow(minus_imgs[i], cmap='gray'); axs[2, i].set_title(f"Z- in {selected_features[i]}"); axs[2, i].axis("off")
            axs[4, i].imshow(minus_diffs[i], cmap='hot'); axs[3, i].set_title("Diff(-)"); axs[3, i].axis("off")

        axs[0, 0].set_ylabel("Original", rotation=0, labelpad=20)
        axs[1, 0].set_ylabel("Z+", rotation=0, labelpad=20)
        axs[2, 0].set_ylabel("Z-", rotation=0, labelpad=20)
        axs[3, 0].set_ylabel("Diff", rotation=0, labelpad=20)

        plt.tight_layout()
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_test_counter_factual_v1.png")
        plt.show()


import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_multiple_latent_traversals(encod, decod, img, z_std, selected_features, steps=7, std_range=3, device='cuda'):
    """
    Visualizes how changing different latent features affects reconstruction.
    Each row corresponds to a feature being traversed.
    Each column shows the reconstruction for a certain std multiple.
    """
    encod.eval()
    decod.eval()

    img = img.to(device)
    z_std = z_std.to(device)

    # Encode image to latent vector
    with torch.no_grad():
        z_orig = encod(img)  # shape: [1, latent_dim]

    multipliers = torch.linspace(-std_range, std_range, steps)

    fig, axs = plt.subplots(len(selected_features), steps, figsize=(steps * 2, len(selected_features) * 2))

    # if len(selected_features) == 1:
    #     axs = axs.unsqueeze(0)  # make it 2D for consistent indexing

    for row_idx, feat_idx in enumerate(selected_features):
        # Create altered versions of z
        z_altered = z_orig.repeat(steps, 1)
        for i, m in enumerate(multipliers):
            z_altered[i, feat_idx] += m * z_std[feat_idx]

        # Decode altered z
        with torch.no_grad():
            recon_imgs = decod(z_altered)  # shape: [steps, C, H, W]

        for col_idx in range(steps):
            img_np = recon_imgs[col_idx].cpu().numpy()#.transpose(1, 2, 0)
            #img_np = np.clip(img_np, 0, 1)

            ax = axs[row_idx][col_idx]
            ax.imshow(img_np[0], cmap = 'gray')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f"{multipliers[col_idx]:.1f}σ", fontsize=8)
        axs[row_idx][0].set_ylabel(f'Feature #{feat_idx}', fontsize=8)

    plt.suptitle("Latent Feature Traversals", fontsize=14)
    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_test_latent_traversal_v1.png")
    plt.show()


# Example usage
# selected_features = [0, 10, 11, 12, 1]
# visualize_counterfactual_changes(image_tensor, encoder_model, decoder_model, selected_features)

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

def compute_latent_feature_variance(encoder, dataloader, device):
    encoder.eval()  # Set encoder to evaluation mode
    all_latents = []

    with torch.no_grad():
        for imgs, _, labels, _ in dataloader:
            imgs = imgs.to(device)
            z = encoder(imgs)  # shape: [batch_size, latent_dim]
            all_latents.append(z.cpu())  # move to CPU to avoid GPU memory issues

    # Concatenate all latent vectors: shape [N_samples, latent_dim]
    all_latents = torch.cat(all_latents, dim=0)

    # Compute variance per feature (dim=0 is across samples)
    feature_variances = torch.std(all_latents, dim=0, unbiased=True)  # shape: [latent_dim]
    feature_means = torch.mean(all_latents, dim=0)

    return feature_variances, feature_means

def plot_latent_histograms_from_loader(data_loader, encoder, device='cuda', num_bins=50, features_per_page=50):
    encoder.eval()
    all_latents = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            z = encoder(x)
            z = z.detach().cpu()

            # Handle if encoder output is a tuple (e.g., z, logits)
            if isinstance(z, (tuple, list)):
                z = z[0]

            all_latents.append(z)

    z_all = torch.cat(all_latents, dim=0).numpy()
    latent_dim = z_all.shape[1]
    x_range = np.linspace(-4, 4, 200)
    normal_pdf = norm.pdf(x_range)

    num_pages = math.ceil(latent_dim / features_per_page)

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min((page + 1) * features_per_page, latent_dim)
        dims_on_page = end_idx - start_idx

        n_cols = 5
        n_rows = math.ceil(dims_on_page / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))

        for i in range(dims_on_page):
            dim_idx = start_idx + i
            row, col = divmod(i, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]

            ax.hist(z_all[:, dim_idx], bins=num_bins, density=True,
                    alpha=0.6, color='skyblue', edgecolor='black')
            ax.plot(x_range, normal_pdf, 'r--')
            ax.set_title(f"Dim {dim_idx}", fontsize=9)
            ax.set_xlim([-4, 4])

        # Hide empty subplots
        for j in range(dims_on_page, n_cols * n_rows):
            row, col = divmod(j, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.suptitle(f'Latent Features {start_idx}–{end_idx - 1}', fontsize=14, y=1.02)
        plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_all_latent_features_{page}.png")


import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def plot_latent_histogram(z, epoch, dims_to_plot=[0, 1, 2], bins=50):
    z = z.detach().cpu().numpy()
    fig, axs = plt.subplots(1, len(dims_to_plot), figsize=(5 * len(dims_to_plot), 4))
    
    x = np.linspace(-4, 4, 1000)
    normal_pdf = stats.norm.pdf(x)

    for i, dim in enumerate(dims_to_plot):
        axs[i].hist(z[:, dim], bins=bins, density=True, alpha=0.6, label="Latent dist")
        axs[i].plot(x, normal_pdf, 'r--', label="N(0,1)")
        axs[i].set_title(f'Latent dim {dim} (epoch {epoch})')
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/disc_gen_loss_latent_hist_epoch_{epoch}.png")
    plt.close()
    
import torchvision.utils as vutils
def sample_and_decode(decoder, latent_dim=150, num_samples=25, device='cuda'):
    """
    Sample from standard normal and decode using the decoder.

    Args:
        decoder (nn.Module): The decoder model (usually part of your AAE or VAE).
        latent_dim (int): Dimensionality of the latent space.
        num_samples (int): Number of samples to generate.
        device (str): 'cuda' or 'cpu'

    Returns:
        images (Tensor): Decoded images as a tensor.
    """
    decoder.eval()  # put decoder in eval mode
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)

        # Decode
        generated = decoder(z)  # shape: (num_samples, C, H, W)

        # Clip or normalize if needed (e.g., tanh output)
        if generated.min() < 0:
            generated = (generated + 1) / 2.0  # from [-1,1] to [0,1]

        # Plot grid
        grid_img = vutils.make_grid(generated.cpu(), nrow=int(num_samples ** 0.5), padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Samples from Latent Space")
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_sampled_realistic_images_v1.png")
        plt.show()

    return generated

def visualize_reconstruction(encod, decod, dataloader, device='cuda', num_images=8):
    """
    Visualize reconstruction ability of an autoencoder model.

    Args:
        model (nn.Module): Trained autoencoder (with encoder and decoder).
        dataloader (DataLoader): DataLoader to fetch input samples.
        device (str): 'cuda' or 'cpu'
        num_images (int): Number of images to visualize (must be <= batch size)
    """
    encod.eval()
    decod.eval()
    with torch.no_grad():
        # Get a batch of data
        for batch in dataloader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device)
            break  # only need the first batch

        # Forward pass through model
        z_img = encod(imgs)
        recon = decod(z_img)

        # If model outputs multiple values (e.g., AAE with discriminator), pick the right one
        # if isinstance(recon, (tuple, list)):
        #     recon = recon[0]

        # Normalize if output is in [-1, 1]
        if recon.min() < 0:
            imgs = (imgs + 1) / 2.0
            recon = (recon + 1) / 2.0

        # Select subset of images
        imgs = imgs[:num_images]
        recon = recon[:num_images]

        # Concatenate original and reconstructed images for visualization
        comparison = torch.cat([imgs, recon], dim=0)  # 2N images
        grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=True)

        # Plot
        plt.figure(figsize=(num_images, 4))
        plt.axis('off')
        plt.title('Original (Top) vs Reconstructed (Bottom)')
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_panc_recontruction_ability_v1.png")
        plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # train_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/NonDemented/*.jpg")
    # train_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/VeryMildDemented/*.jpg")
    # train_paths = np.array(train_no_paths + train_very_paths)
    # test_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/NonDemented/*.jpg")
    # test_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/VeryMildDemented/*.jpg")
    # test_paths = np.array(test_no_paths + test_very_paths)
    # print('number of train paths', len(train_paths))
    # train_labels = np.array(['Non' not in path.split('/')[-2] for path in train_paths]) * 1
    # test_labels = np.array(['Non' not in path.split('/')[-2] for path in test_paths]) * 1

    # train_img_normal_dataset = custom_dataset(train_paths, train_labels, normalise_resize)
    # test_img_normal_dataset = custom_dataset(test_paths, test_labels, normalise_resize)

    batch_size = 64
    # train_loader = DataLoader(train_img_normal_dataset, batch_size = 64, shuffle = True)
    # test_loader = DataLoader(test_img_normal_dataset, batch_size = 64, shuffle = True)


    # all_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/**/NonDemented/*.jpg", recursive = True)
    # all_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/**/VeryMildDemented/*.jpg", recursive = True)
    # all_paths = all_no_paths + all_very_paths
    # all_paths = np.array(all_paths)
    # labels = np.array(['Non' in path.split('/')[-2] for path in all_paths])

    # idx = np.arange(all_paths.shape[0])
    # np.random.shuffle(idx)
    # all_paths = all_paths[idx]
    # labels = labels[idx]
    # batch_size = 64

    # all_img_normal_dataset = custom_dataset(all_paths, labels, normalise_resize)

    # test_dataset = custom_dataset(all_paths, labels, normalise_resize)
    # train_loader = DataLoader(all_img_normal_dataset, batch_size = 64, shuffle = True)
    # test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)
    data_alive_test = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_alive_test.pt", weights_only=False)
    data_dead_test = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_dead_test.pt", weights_only=False)
    test_dataset =  torch.utils.data.ConcatDataset([data_dead_test, data_alive_test])
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)
    encod = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/encod_mmd_version_train_1.pt", map_location = device, weights_only = False).eval()
    decod = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/decod_mmd_version_train_1.pt", map_location = device, weights_only = False).eval()
    # img_clf_model = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", weights_only = False, map_location=device).eval()
    img_clf_model = torch.load("/user/sina.garazhian/u12203/panc_cell/best_clf.pt", weights_only = False, map_location=device).eval()
    
    latent_clf_model = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/subset_mmd_version_train_1.pt", map_location = device, weights_only = False).eval()


    visualize_latent_distribution_by_label(encod, test_loader, device)
    tsne_latent_plot(encod, test_loader, device)
    plot_binary_classification_scores(img_clf_model, latent_clf_model, encod, test_loader, device)
    pearsons_corr = compute_latent_class_correlations(encod, img_clf_model, test_loader, 150, device)
    stds, _ = compute_latent_feature_variance(encod, test_loader, device)
    # print(stds)
    print(pearsons_corr)
    print(pearsons_corr[44])
    print(np.argsort(np.abs(pearsons_corr))[-4:])
    # visualize_counterfactual_changes(test_img_normal_dataset[40][0], encod, decod, stds, np.argsort(np.abs(pearsons_corr))[-4:], std_range=3.0)
    # visualize_multiple_latent_traversals(encod, decod, test_dataset[2][0][np.newaxis, :, :, :], stds, np.argsort(np.abs(pearsons_corr))[-8:], steps=7, std_range=3, device='cuda')
    sample_and_decode(decod, latent_dim=150, num_samples=25, device='cuda')
    visualize_reconstruction(encod, decod, test_loader, device='cuda', num_images=8)
    plot_latent_histograms_from_loader(test_loader , encod, device='cuda', num_bins=50)
    
main()
