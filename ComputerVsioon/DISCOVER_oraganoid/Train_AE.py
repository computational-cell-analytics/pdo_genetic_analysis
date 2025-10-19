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
import sys
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
from skimage.measure import shannon_entropy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import custom_collate_fn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import CellImageEncoder_represent
import pandas as pd
import numpy as np
import dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CellDataset, CellDataset_longitude
# datase = prepare_dataset.CellDataset()

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    # A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])

def apply_albumentations(batch_images, transform):
    augmented = []
    for img in batch_images:
        img_np = img.cpu().numpy()
        if img_np.ndim == 2:
            img_np = img_np[..., None]  # expand channel if grayscale
        augmented_img = transform(image=img_np)["image"]
        augmented.append(augmented_img)
    return torch.stack(augmented)


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


def validate_model(
    encod, decod, disc, recog, clf_model, subset_model,
    perception_loss_clf_func, clf_loss_func, subset_loss_func, losses,
    valid_loader, latent_dim, std_range, var_weight, gen_weight,
    epoch, device
):
    encod.eval()
    decod.eval()
    disc.eval()
    recog.eval()
    subset_model.eval()
    clf_model.eval()

    val_recon = 0
    val_disc = 0
    val_gen = 0
    val_clf = 0
    val_disent = 0
    val_cov = 0
    val_vae_all = 0
    val_subset = 0

    z_std = compute_latent_feature_variance(encoder, valid_loader, device)

    with torch.no_grad():
        for imgs, _, _, labels, _ in valid_loader:
            imgs = imgs.to(device)
            altered_index = torch.randint(0, latent_dim, (1,)).item()

            z_img = encod(imgs)
            img_recon = decod(z_img)

            range_alteration = (torch.rand(latent_dim) * 2 * std_range) - std_range
            range_alteration = range_alteration.to(device)
            z_alter = z_img.clone()
            z_alter[:, altered_index] = z_alter[:, altered_index] + z_std[altered_index] * range_alteration[altered_index]
            alt_img_recon = decod(z_alter)
            diff_img = (img_recon - alt_img_recon).abs().to(device)

            logits = recog(diff_img)
            altered_labels = torch.full((imgs.size(0),), altered_index, dtype=torch.long, device=device)
            disent_loss = F.cross_entropy(logits, altered_labels)

            recon_loss = perception_loss_clf_func(img_recon, imgs)
            clf_loss = clf_loss_func(img_recon, imgs)
            z_var_loss = losses.latent_z_variance_loss(z_img)
            clf_target = torch.sigmoid(clf_model(imgs))
            subset_loss = subset_loss_func(subset_model(z_img), clf_target)

            total_vae_loss = 5 * recon_loss + 5 * clf_loss + 0.0001 * z_var_loss + subset_loss + disent_loss

            normal_dist = torch.randn((imgs.size(0), z_img.shape[1])).to(device)
            disc_loss = losses.Discriminator_loss(disc(normal_dist), disc(z_img))
            gen_loss = gen_weight * (-torch.mean(torch.log(disc(z_img))))

            val_clf += clf_loss.item()
            val_recon += recon_loss.item()
            val_cov += var_weight * z_var_loss.item()
            val_disent += disent_loss.item()
            val_vae_all += total_vae_loss.item()
            val_subset += subset_loss.item()
            val_disc += disc_loss.item()
            val_gen += gen_loss.item()
        plt.figure()
        f, axarr = plt.subplots(2,5)
        plt.title("Reconstruction Progress report")
        for idx, i in enumerate(img_recon):
            axarr[0,idx].imshow(imgs.cpu().detach().numpy()[idx][0], cmap='gray')
            axarr[1,idx].imshow(i.cpu().detach().numpy()[0], cmap='gray')
            
            plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/{epoch}_new_data.png")
            plt.show()
        
            if idx == 4:
                break
        plt.close()

    return {
        'recon': val_recon / len(valid_loader),
        'disc': val_disc / len(valid_loader),
        'gen': val_gen / len(valid_loader),
        'clf': val_clf / len(valid_loader),
        'disent': val_disent / len(valid_loader),
        'cov': val_cov / len(valid_loader),
        'vae_all': val_vae_all / len(valid_loader),
        'subset': val_subset / len(valid_loader)
    }



###getting pdo metadata
pdo_metadata = pd.read_csv("/user/sina.garazhian/u12203/panc_cell/pdo_data_drug_info_rectal_d5_s3_chr_sx5_ts_combined_v3_corrected.csv", dtype=object)
patients = set(pdo_metadata['patient_name'].values)
patient_dict = {patient: idx for idx, patient in enumerate(patients)} ###get list of all patients

device = 'cuda'
###get cleaned datasets
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



subset_value = int(sys.argv[1])
print(sys.argv[1])
###Create models
latent_size = 350
encoder = utils.CellImageEncoder(latent_size) 
decoder = utils.Decoder(latent_size, 1, 0.0)
cellimage_encoder = CellImageEncoder_represent()
encoder_path = "/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/custom_encoder_new_setup_model_epoch32_sep2.1850_new_data.pt"
cellimage_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
# disc = models.Discriminator(latent_size)
recog = models.Disentangler(1, latent_size)
imgnet_model = vgg19(VGG19_Weights.IMAGENET1K_V1)
subset_model = models.simple_neuron(subset_value, len(patients))
device_model = models.device_network(1, subset_value)
# recog_attend = models.CrossAttentionRecognizer()
# clf_model = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", weights_only = False, map_location=device).eval()

encoder.to(device)
decoder.to(device)
# disc.to(device)
imgnet_model.to(device)
recog.to(device)
subset_model.to(device)
cellimage_encoder.to(device)
device_model.to(device)
# recog_attend.to(device)

lr = 0.0001 
weight_decay = 1e-4
opt_vae = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
# opt_disc = optim.AdamW(disc.parameters(), lr=lr/5, weight_decay = weight_decay)
opt_gen = optim.AdamW(encoder.parameters(), lr=lr/5)
opt_recog = optim.AdamW(recog.parameters(), lr = lr )
opt_subset = optim.AdamW(subset_model.parameters(), lr = lr/5)
opt_device = optim.AdamW(device_model.parameters(), lr=lr/5)
# opt_recog_attend = optim.AdamW(recog_attend.parameters(), lr=1e-4, weight_decay=1e-4)

from torch.optim.lr_scheduler import OneCycleLR

# Assuming you know total number of steps:
# total_steps = num_epochs * len(dataloader)

# === Hyperparameters ===
epochs = 100
batch_size = 64
latent_dim = latent_size
lr_vae = 1e-4
lr_disc = 2e-5
lr_gen = 2e-5
adv_weight = 0.05
var_weight = 0.0001
alpha = 0.5  # for hybrid loss
warmup_epochs = 5
std_range = 3


# scheduler_vae = OneCycleLR(
#     opt_vae,
#     max_lr=1e-4,            # peak learning rate
#     steps_per_epoch=len(train_loader),
#     epochs=epochs,
#     pct_start=0.2,          # % of cycle spent increasing LR
#     anneal_strategy='cos',  # or 'linear'
#     div_factor=25.0,        # initial LR = max_lr / div_factor
#     final_div_factor=1e4    # min LR = initial LR / final_div_factor
# )






recon_losses = []
disc_losses = []
gen_losses = []
clf_losses = []
device_losses = []
disent_losses = []
all_vae_losses = []
mean_losses = []
cov_losses = []
mmd_losses = []
subset_losses = []
recon_losses_test = []
disc_losses_test = []
gen_losses_test = []
clf_losses_test = []
all_vae_losses_test = []
disent_losses_test = []
mean_losses_test = []
cov_losses_test = []
subset_losses_test = []
device_losses_test = []



os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/diff_images/*')

# with open("/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/Training_subset20_logs.txt", 'a') as fi:
#     fi.write("new Training log \n")

with open(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/Training_subset{subset_value}_{latent_size}_logs.txt", 'a') as fi:
    fi.write("new Training log \n")
# os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/*.png')
perception_loss_clf_func = losses.PerceptionLossVGG(device)
pixel_loss = losses.PixelwiseLoss(mode='l1')
target_layers = ['conv1', 'conv_multi_scale.0', 'conv_multi_scale.1', 'conv_multi_scale.2', 'res_block.4', 'fusion']
clf_loss_func = losses.FeatureExtractor_CellImageEncoder(cellimage_encoder, target_layers)
subset_loss_func = nn.CrossEntropyLoss()
focal_loss = losses.FocalLoss()


for epoch in range(epochs):
    k = 0
    encoder.eval()
    z_std, z_mean = compute_latent_feature_variance(encoder, train_loader, device)
    print(f"Epoch {epoch - 1} - latent mean avg: {z_mean.mean():.3f}, std avg: {z_std.mean():.3f}, {(z_std >= 0.1).sum()} latent features have std higher than 0.1")
    # altered_index = torch.randint(0, latent_dim, (1,)).item()
    torch.cuda.empty_cache()

    running_recon = 0
    running_disc = 0
    running_gen = 0
    running_clf = 0
    running_disent = 0
    running_cov = 0
    running_vae_all = 0
    running_mmd = 0
    running_subset = 0 
    running_device = 0
    # z_std = compute_latent_feature_variance(encoder, train_loader, device)
    # current_lr = scheduler_vae.get_last_lr()[0]
    # print(f"Current LR: {current_lr}")
    num_batches = len(train_loader)
    for imgs, _, _, labels, file_names in train_loader:
        labels = torch.tensor([patient_dict[pat] for pat in labels], device=device)
        device_labels = torch.tensor([file_name[-1] == 'S3_chr' for file_name in file_names], device=device) * 1
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
        imgs = imgs.to(device)
        # imgs = utils.apply_clahe_batch(imgs).to(device)
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
        ###Encoder + Decoder
        # disc.eval()
        encoder.train()
        decoder.train()
        recog.train()
        z_img, mu, logva = encoder(imgs)
        # z_top = utils.reparameterize(zs['mu_top'], zs['logvar_top'])
        # z_bottom = utils.reparameterize(zs['mu_bottom'], zs['logvar_bottom'])
        range_alteration = torch.from_numpy(np.random.choice(np.linspace(-2, 2, 70), imgs.shape[0])) ###Alter latent feature in range of std_range 
        range_alteration = range_alteration.to(device)
        z_alter = z_img.clone()
        for i in range(imgs.size(0)): ##change different latent feature per each image
            epsilon = range_alteration[i] * z_std[altered_indices[i]]
            z_alter[i, altered_indices[i]] = z_alter[i, altered_indices[i]] + epsilon ##change different latent feature per each image
        z_combined = torch.cat([z_img, z_alter], dim = 0)
        img_combined_recon = decoder(z_combined)
        img_recon, alt_img_recon = torch.chunk(img_combined_recon, 2, dim=0)
        recon_loss = perception_loss_clf_func(img_recon, imgs)  ###perception loss
        z_var_loss = losses.covariance_loss(z_img) ###Variance loss
        # var_sup   = z_img[:, :20].var(0).mean()
        # var_unsup = z_img[:, 20:].var(0).mean()
        # z_var_loss = (1/var_sup + var_unsup)
        diff_img = (img_recon - alt_img_recon).abs()
        # diff_img.requires_grad = True
        diff_img.to(device)
        if (epoch + 1) % 5 == 0 and k % 100 ==0:
            plt.imshow(diff_img[0][0].detach().cpu().numpy() , cmap = 'hot')
            plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/diff_images/epoch_{epoch}_batch{k}_{labels[0]}_{altered_indices[0]}_new_data")
        k += 1
        # diff_img = (diff_img - diff_img.mean()) / (diff_img.std() + 1e-6)
        logits = recog(diff_img[:, 0:1, :, :])
        disent_loss = F.cross_entropy(logits, altered_indices) ###disentanglemanet loss
        subset_loss = subset_loss_func(subset_model(z_img), labels)
        device_loss = subset_loss_func(device_model(z_img), device_labels)
        # loss, r_loss, total_kl, kld_t, kld_b = utils.loss_fn(img_recon, imgs, zs['mu_top'], zs['logvar_top'], zs['mu_bottom'], zs['logvar_bottom'])
        clf_oriented = torch.tensor([(((clf_loss_func(imgs)[key] - clf_loss_func(img_recon)[key])**2).sum())**0.5 for key in clf_loss_func(imgs).keys()]).sum() ###classification orientation loss
        total_vae_loss =  recon_loss + F.mse_loss(img_recon, imgs) 
        kl_loss = -0.5 * torch.mean(1 + logva - mu.pow(2) - logva.exp())
        loss = 7 * total_vae_loss + 3 * kl_loss +  6 * clf_oriented + 2 * z_var_loss + 2 * disent_loss + 2 * subset_loss + 2 * device_loss
        opt_vae.zero_grad()
        opt_subset.zero_grad()
        opt_recog.zero_grad()
        opt_device.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(encod.parameters(), max_norm=5.0)
        opt_vae.step()
        opt_subset.step()
        opt_recog.step()
        opt_device.step()
        # scheduler_vae.step()
        running_recon += total_vae_loss.item()
        running_cov += z_var_loss.item()
        
        
        running_clf += clf_oriented.item()
        running_cov += z_var_loss.item()
        running_disent += disent_loss.item()
        running_vae_all += total_vae_loss.item()
        running_subset += subset_loss.item()
        running_mmd += kl_loss.item()
        running_vae_all += loss.item()
        running_device += device_loss.item()
        
        ##Classification Oriented training
    # plt.figure()
    # f, axarr = plt.subplots(2,5)
    # plt.title("Reconstruction Progress report")
    # for idx, i in enumerate(img_recon):
    #     axarr[0,idx].imshow(imgs.cpu().detach().numpy()[idx][0], cmap='gray')
    #     axarr[1,idx].imshow(i.cpu().detach().numpy()[0], cmap='gray')

    #     plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/{epoch}.png")
    #     plt.show()
        
    #     if idx == 4:
    #         break
    # plt.close()
        # running_vae_all += ru/4
    ###getting all losses in array
    # if epoch % 10 == 0:
    #     with torch.no_grad():
    #             for x_batch, _ in val_loader:  # or a fixed subset
    #                 z_batch = encod(x_batch.to(device))
    #                 # Just one batch per epoch
    #                 visulize.plot_latent_histogram(z_batch, epoch, dims_to_plot=[0, 1, 2, 3, 4, 5, 15, 20, 30, 40, 50, 60, 100, 140], bins=50)
    if (epoch + 1) % 5 == 0:
        torch.save(encoder, f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/encod_custom_{latent_size}_{subset_value}_all_disent_epoch{epoch}_{recon_losses[-1]}_v1_new_data.pt")
        torch.save(decoder, f'/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/decod_custom_{latent_size}_{subset_value}_all_disent_epoch{epoch}_{recon_losses[-1]}_v1_new_data.pt')
        torch.save(subset_model, f'/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/subset_custom_{latent_size}_{subset_value}_all_disent_epoch{epoch}_{recon_losses[-1]}_v1_new_data.pt')
        # torch.save(subset_model, '/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/subset_spread_v1.pt')
    recon_losses.append(running_recon/num_batches)
    all_vae_losses.append(running_vae_all/num_batches)
    mmd_losses.append(running_mmd/num_batches)
    clf_losses.append(running_clf/num_batches)
    # mean_losses.append(running_mean/64)
    cov_losses.append(running_cov/num_batches)
    disent_losses.append(running_disent/num_batches)
    subset_losses.append(running_subset/num_batches)
    device_losses.append(running_device/num_batches)
    # subset_losses.append(running_subset/64)
    print(f"Epoch {epoch}, recons loss is {recon_losses[-1]} , and ALL loss is {all_vae_losses[-1]}, kld loss is {mmd_losses[-1]}, clf loss is {clf_losses[-1]}, disentanglement loss in {disent_losses[-1]}, cov loss is {cov_losses[-1]}, subset loss is {subset_losses[-1]}, device loss is {device_losses[-1]}")
    print(f"Epoch {epoch} - latent mean avg: {z_mean.mean():.3f}, std avg: {z_std.mean():.3f}, {(z_std >= 0.1).sum()} latent features have std higher than 0.1")
    with open(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/Training_subset{subset_value}_{latent_size}_logs.txt", 'a') as fi:
        fi.write(f"Epoch {epoch}, recons loss is {recon_losses[-1]} , and ALL loss is {all_vae_losses[-1]}, kld loss is {mmd_losses[-1]}, clf loss is {clf_losses[-1]}, disentanglement loss in {disent_losses[-1]}, cov loss is {cov_losses[-1]}, subset loss is {subset_losses[-1]}, device loss is {device_losses[-1]} \n")
    with torch.no_grad():
        running_recon_val = 0
        running_vae_all_val = 0
        for imgs, _, _, labels, _ in val_loader:
            imgs = utils.apply_clahe_batch(imgs).to(device)
            labels = torch.tensor([patient_dict[pat] for pat in labels], device=device)
            encoder.eval()
            decoder.eval()
            z, mu, logva = encoder(imgs)
            img_recon = decoder(z)
            recon_loss = perception_loss_clf_func(img_recon, imgs) 
            
            total_vae_loss =  recon_loss + F.mse_loss(img_recon, imgs)
            kl_loss = -0.5 * torch.mean(1 + logva - mu.pow(2) - logva.exp())
            loss = total_vae_loss + kl_loss
            
            
            

            
            running_recon_val += total_vae_loss.item()
            # running_cov += z_var_loss.item()
            
            running_vae_all_val += loss.item()
    # val_losses = validate_model(
    # encoder, decoder, disc, recog, clf_model, subset_model,
    # perception_loss_clf_func, clf_loss_func, subset_loss_func, losses,
    # val_loader, latent_dim, std_range, var_weight, gen_weight,
    # epoch, device
    #     )
    recon_losses_test.append(running_recon_val)
    # disc_losses_test.append(val_losses['disc'])
    # gen_losses_test.append(val_losses['gen'])
    # clf_losses_test.append(val_losses['clf'])
    # # mean_losses.append(running_mean/64)
    # cov_losses_test.append(val_losses['cov'])
    # disent_losses_test.append(val_losses['disent'])
    all_vae_losses_test.append(running_vae_all_val)
    # subset_losses_test.append(val_losses['subset'])
    # print(f"Validation - Epoch {epoch}, recon loss: {val_losses['recon']}, discriminator loss: {val_losses['disc']}, "
    #   f"generator loss: {val_losses['gen']}, clf loss: {val_losses['clf']}, disent loss: {val_losses['disent']}, "
    #   f"cov loss: {val_losses['cov']}, subset loss: {val_losses['subset']}, total VAE loss: {val_losses['vae_all']}")
    print(f"Epoch {epoch}, recons loss is {recon_losses_test[-1]} , and ALL loss is {all_vae_losses_test[-1]}")
    # running_recon = 0
    # running_disc = 0
    # running_gen = 0
    # running_all = 0
    # for imgs, labels in val_loader:
    #     with torch.no_grad():
    #         disc.eval()
    #         encod.eval()
    #         decod.eval()
    #         imgs = imgs.to(device)
    #         ###Encoder + Decoder
    #         z_img= encod(imgs)
    #         img_recon = decod(z_img)
    #         #recon_loss = pixel_loss(img_recon, imgs)
    #         recon_loss = perception_loss_clf_func(img_recon, imgs)
    #         running_recon += recon_loss.item()
    #         ###Discr
    #         normal_dist = torch.randn( (batch_size, latent_size)).to(device)
    #         disc_loss = -torch.mean((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
    #         running_disc += disc_loss.item()
    #         ##Generatort (Advarserial)
    #         gen_loss = gen_weight * (-torch.mean(torch.log(disc(z_img))))
    #         running_gen += gen_loss.item()
    #         ###Classification loss
    #         clf_loss = clf_loss_func(img_recon, imgs)
    #         running_clf += clf_loss.item()
    #         # running_all += running_recon + running_disc + running_gen + running_clf
            

            
    # recon_losses_test.append(running_recon/64)
    # disc_losses_test.append(running_disc/64)
    # gen_losses_test.append(running_gen/64)
    # clf_losses_test.append(running_clf/64)
    # all_vae_losses_test.append(running_all/64)
    # print(f"Epoch {epoch}, recons loss valid is {recon_losses_test[-1]}, discriminator loss valid is {disc_losses_test[-1]}, generator loss valid is {gen_losses_test[-1]}, and ALL loss valid is {all_losses_test[-1]}")


plt.plot(recon_losses)

# plt.plot(all_losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon'], loc='upper left')
plt.savefig('/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/cunstom_encoder_AE_v1_train_new_data.png')
plt.close()


# plt.plot(recon_losses_test)
# plt.plot(disc_losses_test)
# plt.plot(gen_losses_test)
# # plt.plot(all_vae_losses)
# plt.plot(disent_losses_test)
# plt.plot(cov_losses_test)
# plt.plot(clf_losses_test)
# plt.plot(subset_losses_test)
# # plt.plot(all_losses)
# plt.title('model validation losses')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['recon', 'disc', 'gen', 'disent', 'cov', 'clf', 'subset'], loc='upper left')
# plt.savefig('/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/losses_Sush_6_wdecay_weight_warmup_clf_mean_cov_subset_v5_valid.png')
# plt.close()


# plt.plot(recon_losses_test)
# plt.plot(disc_losses_test)
# plt.plot(gen_losses_test)
# plt.plot(clf_losses_test)
# plt.plot(all_losses_test)
# plt.title('model losses test')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['recon', 'disc', 'gen', 'clf', 'all'], loc='upper left')
# plt.savefig('losses_test_Sush_4_wdecay_weight_warmup_clf.png')

# plt.plot(i[0].item() for i in layers_losses)
# plt.plot(i[1].item() for i in layers_losses)
# plt.plot(i[2].item() for i in layers_losses)
# # plt.plot(all_losses)
# plt.title('layers losses')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['layer1', 'layer2', 'layer3'], loc='upper left')
# plt.savefig('layer_losses_spectral_3.png')
# with open("results_subset.txt",'w') as file:
#     for i in range(len(recon_losses)):
#         file.write(f"recons loss is {recon_losses[i]}, disc loss is {disc_losses[i]}, gen loss is {gen_losses[i]}, disent loss is{disent_losses[i]}, cov loss is{cov_losses[i]}, clf loss is{clf_losses[i]}" + '\n')

# torch.save(encod, "/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/encod_spread_v1.pt")
# torch.save(decod, '/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/decod_spread_v1.pt')
# torch.save(subset_model, '/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/subset_spread_v1.pt')
