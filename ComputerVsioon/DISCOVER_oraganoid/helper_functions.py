from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import torchvision.utils as vutils
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
from dataset import custom_collate_fn
from torchvision.models import vgg19, VGG19_Weights
from matplotlib import pyplot as plt
from utils import VGG
from data_load import custom_dataset, cv_2_transforms, normalise_resize
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
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
from losses import *
import torch.nn.functional as F
import os
import dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
import utils
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

import pandas as pd
import numpy as np
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CellDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def compute_latent_feature_variance(encoder, dataloader, device):
    encoder.eval()  # Set encoder to evaluation mode
    all_latents = []

    with torch.no_grad():
        for imgs, _ , _, _, _ in dataloader:
            imgs = imgs.to(device)
            z, _, _ = encoder(imgs)  # shape: [batch_size, latent_dim]
            all_latents.append(z.cpu())  # move to CPU to avoid GPU memory issues

    # Concatenate all latent vectors: shape [N_samples, latent_dim]
    all_latents = torch.cat(all_latents, dim=0)

    # Compute variance per feature (dim=0 is across samples)
    feature_variances = torch.std(all_latents, dim=0, unbiased=True)  # shape: [latent_dim]
    feature_means = torch.mean(all_latents, dim=0)

    return feature_variances, feature_means


def tsne_latent_plot(encoder, dataloader, device, n_batches=5):
    pdo_metadata = pd.read_csv("/user/sina.garazhian/u12203/panc_cell/pdo_data_drug_info_rectal_d5_s3_chr_sx5_ts_combined_v3_corrected.csv", dtype=object)
    patients = set(pdo_metadata['patient_name'].values)
    patient_dict = {patient: idx for idx, patient in enumerate(patients)} ###get list of all patients
    encoder.eval()
    z_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, _ , _,  labels, _ in dataloader:
            # if i >= n_batches:
            #     break
            imgs = imgs.to(device)
            # print(labels)
            labels = torch.tensor([patient_dict[pat] for pat in labels])
            z, _, _ = encoder(imgs)
            z = z.cpu()
            z_all.append(z[:,:30])
            labels_all.append(labels)

    z_all = torch.cat(z_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0).numpy()
    
    z_2d = TSNE(n_components=2, perplexity=30).fit_transform(z_all.numpy())
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels_all, palette='tab10', s=40, alpha=0.8)
    plt.title('t-SNE Projection of Latent Space by Label')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/kdl_all_losses_test_latent_tsne_v1_resnet_disent_new_data.png")
    plt.show()

def visualize_reconstruction(encod, decod, dataset, device='cuda', num_images=8):
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
        all_imgs = []
        samples_idx = np.random.choice(np.arange(0, len(dataset)), num_images)
        print(samples_idx)
        all_imgs = [dataset[i][1].unsqueeze(0) for i in samples_idx]
        all_imgs = torch.cat(all_imgs, dim = 0)
        print(all_imgs.shape)
        all_imgs = all_imgs.to(device)
        z_imgs, _, _ = encoder(all_imgs)
        imgs_recon = decoder(z_imgs)
        # Concatenate original and reconstructed images for visualization
        comparison = torch.cat([all_imgs, imgs_recon], dim=0)  # 2N images
        grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=True)

        # Plot
        plt.figure(figsize=(num_images, 4))
        plt.axis('off')
        plt.title('Original (Top) vs Reconstructed (Bottom)')
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/kdl_all_losses_test_recontruction_ability_v1_resnet_disent_new_data.png")
        plt.show()

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
        # if generated.min() < 0:
        #     generated = (generated + 1) / 2.0  # from [-1,1] to [0,1]

        # Plot grid
        grid_img = vutils.make_grid(generated.cpu(), nrow=int(num_samples ** 0.5), padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Samples from Latent Space")
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/kdl_all_losses_test_realistic_images_v1_resnet_disent_new_data.png")
        plt.show()


def feature_traversal(encod, decod, feature_idx ,test_img, vars, device='cuda'):
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
    all_zs = []
    with torch.no_grad():
        # Get a batch of data
        # all_imgs = []
        
        img = test_img.to(device)
        
        
        z_imgs, _, _ = encod(img)
        for i in [-3, -2, -1, 0, 1, 2, 3]:
            new_z = z_imgs.squeeze().clone()
            new_z[feature_idx] += i * vars[feature_idx]
            # new_z[feature_idx + 3] += i * vars[feature_idx + 3]
            # new_z[feature_idx + 6] += i * vars[feature_idx + 6]
            # new_z[feature_idx + 9] += i * vars[feature_idx + 9]
            # new_z[:20] += i * vars[:20]
            all_zs.append(new_z.unsqueeze(0))
        
        z_imgs = torch.cat(all_zs, dim = 0)
        # print(z_imgs.shape)
        imgs_recon = decod(z_imgs)
        

        # Plot
        plt.figure(figsize=(10, 10))
        plt.title("Generated Samples from Latent Space")
        plt.axis("off")
        k = 0
        for rec_img in imgs_recon:
            plt.subplot(1,7,k+1)
            plt.imshow(rec_img[0].cpu(), cmap='gray')
            plt.axis("off")
            k += 1
        # grid_img = vutils.make_grid(imgs_recon.cpu(), nrow=int(5 ** 0.5), padding=2, normalize=True)
        
        
        # plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/results/kdl_all_losses_test_latent_traversal_v1_resnet_disent_new_data_feat_{feature_idx}_device.png")
        plt.show()
        plt.close()

def traversing_over_all_latent_features(encod, decod, data_loader,  z_std, device):
    latent_dim = 400
    os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/diff_imgs/*')
    k = 0
    for imgs, _ , _, labels, _ in train_loader:
        labels = torch.tensor([patient_dict[pat] for pat in labels], device=device)
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
        imgs = imgs.to(device)
        # imgs = utils.apply_clahe_batch(imgs).to(device)
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device)
        z_img, mu, logva = encoder(imgs)
        range_alteration = torch.from_numpy(np.random.choice(np.linspace(-2, 2, 70), imgs.shape[0])) ###Alter latent feature in range of std_range 
        range_alteration = range_alteration.to(device)
        z_alter = z_img.clone()
        for i in range(imgs.size(0)): ##change different latent feature per each image
            epsilon = range_alteration[i] * z_std[altered_indices[i]]
            z_alter[i, altered_indices[i]] = z_alter[i, altered_indices[i]] + epsilon ##change different latent feature per each image
        z_combined = torch.cat([z_img, z_alter], dim = 0)
        img_combined_recon = decoder(z_combined)
        img_recon, alt_img_recon = torch.chunk(img_combined_recon, 2, dim=0)
        diff_img = (img_recon - alt_img_recon).abs()
        # diff_img.requires_grad = True
        diff_img.to(device)
        plt.imshow(diff_img[0][0].detach().cpu().numpy() , cmap = 'hot')
        plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/diff_imgs/epoch_batch{k}_feature_{altered_indices[0]}_{list(patient_dict.keys())[0]}_new_data")
        k += 1
import warnings

def safe_pearsonr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            warnings.warn("One of the input arrays is constant; Pearson r is undefined.")
            return 0.0, 1.0  # r = 0, p = 1 (no correlation)
        return pearsonr(x, y)

def spearman_corr_categroy_based(encoder, encod_cellimg, clf_model, dataset, patient_dict, device):
    emp_dic = {}
    encoder.to(device).eval()
    clf_model.to(device).eval()
    encod_cellimg.to(device).eval()
    for idx in range(len(dataset)):
        if dataset[idx][3] not in emp_dic.keys():
            emp_dic[dataset[idx][3]] = []
        else:
            emp_dic[dataset[idx][3]].append(dataset[idx][0].unsqueeze(0))
    
    with torch.no_grad():
        for label in emp_dic.keys():
            print(len(emp_dic[label]))
            print(label)
            imgs = torch.cat(emp_dic[label], dim=0)
            imgs = imgs.to(device)
            z_imgs, _, _ = encoder(imgs)
            logit_value = clf_model(encod_cellimg(imgs))[:, patient_dict[label]].cpu().numpy()
            pear_scores = []
            for feature_idx in range(z_imgs.shape[1]):
                pear_score, _ = safe_pearsonr(logit_value, z_imgs.cpu().numpy()[:, feature_idx])
                pear_scores.append(pear_score)
            pear_scores = np.array(pear_scores)
            pear_scores[np.isnan(pear_scores)] = 0
            plt.figure(figsize=(12, 4))
            plt.bar(np.arange(z_imgs.shape[1]), pear_scores)
            plt.xlabel("Latent Feature Index")
            plt.ylabel("Pearson Correlation with IVF-CLF Score")
            plt.title(f"Latent Feature vs. IVF-CLF Score Correlation in {label}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/kld_test_pearson_v1_{label}_new_data.png")
        
        

pdo_metadata = pd.read_csv("/user/sina.garazhian/u12203/panc_cell/pdo_data_drug_info_rectal_d5_s3_chr_sx5_ts_combined_v3_corrected.csv", dtype=object)
patients = set(pdo_metadata['patient_name'].values)
patient_dict = {patient: idx for idx, patient in enumerate(patients)} ###get list of all patients

device = "cuda" if torch.cuda.is_available() else "cpu"

control_train = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_train_v1.pt", weights_only=False)
control_val = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_val_v1.pt", weights_only=False)
control_test = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_test_v1.pt", weights_only=False)

###build dataloaders
###in train loader, each batch consists of same samples of different patients to have unbiased batch
labels = [label for _,_,_, _, _, label, _ in control_train]  # your dataset labels
sampler = dataset.BalancedBatchSampler(labels, patients_per_batch=8, samples_per_patient=8)
train_loader = DataLoader(control_train, batch_sampler=sampler, collate_fn=custom_collate_fn)
val_loader = DataLoader(control_val, batch_size=64, collate_fn=custom_collate_fn)
test_loader = DataLoader(control_test, batch_size=64, collate_fn=custom_collate_fn)



# file_name = "custom_400_all_res_disent_epoch99_1.4672311737836206_v1.pt"
# file_name = "custom_400_all_res_disent_epoch49_1.9179353392809124_v1.pt"
# file_name = "custom_400_all_res_disent_epoch54_1.8177350542140323_v1_new_data.pt" ##without device loss
file_name = "custom_400_all_res_disent_epoch14_1.9221048950108248_v1_new_data.pt" ##with deive loss

encoder = torch.load(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/encod_{file_name}",
                          map_location='cuda', weights_only= False)
decoder = torch.load(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/decod_{file_name}",
                          map_location='cuda', weights_only= False)

encoder.eval()
decoder.eval()
cellimage_encoder = utils.CellImageEncoder_represent()
encoder_path = "/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/model_epoch32_sep1.8459.pt"
cellimage_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
embeddings_size = 128
n_patients = 10
model_clf = utils.ClassifierHead(in_dim=embeddings_size, n_classes=n_patients)
model_clf = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/clf_model_epoch20.pt", weights_only= False, map_location='cuda')


# tsne_latent_plot(encoder, test_loader, device, n_batches=5)
# visualize_reconstruction(encoder, decoder, control_test, device='cuda', num_images=8)
# sample_and_decode(decoder, latent_dim=400, num_samples=25, device='cuda')
vars, avgs = compute_latent_feature_variance(encoder, test_loader, device)
samples_idx = np.random.choice(np.arange(0, len(control_test)),2)
test_img = control_test[samples_idx[0]][1].unsqueeze(0)
print(control_test[samples_idx[0]][3])
for i in range(31):
    feature_traversal(encoder, decoder,i ,test_img, vars, device='cuda')
# traversing_over_all_latent_features(encoder, decoder, val_loader, vars, 'cuda')
# spearman_corr_categroy_based(encoder, cellimage_encoder, model_clf, control_train, patient_dict, device)
# print(avgs)
