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
# import visulize




###getting data
# train_paths = glob("/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/**/*.jpg", recursive = True)
train_no_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/train/NonDemented/*.jpg")
train_very_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/train/VeryMildDemented/*.jpg") + glob("/user/sina.garazhian/u12203/kaggle_alz/train/MildDemented/*.jpg")
train_paths = np.array(train_no_paths + train_very_paths)
test_no_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/test/NonDemented/*.jpg")
test_very_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/test/VeryMildDemented/*.jpg") + glob("/user/sina.garazhian/u12203/kaggle_alz/test/MildDemented/*.jpg")

test_paths = np.array(test_no_paths + test_very_paths)
print('number of train paths', len(train_paths))
train_labels = np.array(['Non' not in path.split('/')[-2] for path in train_paths]) * 1
test_labels = np.array(['Non' not in path.split('/')[-2] for path in test_paths]) * 1


idx = np.arange(train_paths.shape[0])
np.random.shuffle(idx)
train_paths = train_paths[idx]
train_labels = train_labels[idx]
idx = np.arange(test_paths.shape[0])
np.random.shuffle(idx)
test_paths = test_paths[idx]
test_labels = test_labels[idx]

###create dataset torch object


class cv_2_transforms(torch.nn.Module):
    def __init__(self, img_size, margin = 20):
        super(cv_2_transforms, self).__init__()
        self.img_size = img_size
        self.margin = margin
    def forward(self, img):
        #img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[self.margin:img.shape[0]-self.margin , self.margin:img.shape[1]-self.margin]
        img = cv2.resize(img, (self.img_size,self.img_size) , interpolation = cv2.INTER_AREA)
        return img

normalise_resize = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    cv_2_transforms(64),
    #ToTensor(),
    # v2.Normalize(mean=means, std=stds),
                #Grayscale(num_output_channels = 3),
    ToTensor()
])


class custom_dataset(Dataset):
    def __init__(self, img_paths, img_labels, transform = None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
    def __getitem__(self,index):
        img = cv2.imread(self.img_paths[index])
        label = self.img_labels[index]
        if self.transform:
            img = self.transform(img)
        #img = torch.permute(img, (2, 0, 1))
        img = img.repeat(3, 1, 1)
        # img = img/255
        # img = (img - means)/stds
        return img, label
    def __len__(self):
        return(len(self.img_paths))
    
device = "cuda" if torch.cuda.is_available() else "cpu"

###Create dataset and dataloader instances
train_img_normal_dataset = custom_dataset(train_paths, train_labels, normalise_resize)
test_img_normal_dataset = custom_dataset(test_paths, test_labels, normalise_resize)
val_img_normal_dataset, test_img_normal_dataset = torch.utils.data.random_split(test_img_normal_dataset, [0.5, 0.5])
batch_size = 64
train_loader = DataLoader(train_img_normal_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_img_normal_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_img_normal_dataset, batch_size = batch_size, shuffle = True)

###Cretae models
latent_size = 350
latent_dim = 350
encod = models.Encoder(3, 0.03, latent_size)
decod = models.Decoder(latent_size ,1, 0.03)
recog = models.Recognizer(latent_dim)
# recog = models.Disentangler(1, latent_dim)
imgnet_model = vgg19(VGG19_Weights.IMAGENET1K_V1)
subset_model = models.simple_neuron()
clf_model = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg_1.pt", weights_only = False, map_location=device).eval()

encod.to(device)
decod.to(device)
imgnet_model.to(device)
recog.to(device)
subset_model.to(device)

lr = 0.0001 
weight_decay = 1e-4
opt_vae = optim.AdamW(list(encod.parameters()) + list(decod.parameters()), lr=lr)
opt_encoder = optim.AdamW(encod.parameters(), lr)
opt_decoder = optim.AdamW(decod.parameters(), lr)
opt_recog = optim.AdamW(recog.parameters(), lr = lr * 10, weight_decay=0.0001 )
opt_subset = optim.AdamW(subset_model.parameters(), lr = lr/5)


# === Hyperparameters ===
epochs = 100
lr_vae = 1e-4
std_range = 3




recon_losses = []
clf_losses = []
disent_losses = []
all_vae_losses = []
mean_losses = []
cov_losses = []
kld_losses = []
subset_losses = []
recon_losses_val = []
disc_losses_val = []
gen_losses_val = []
clf_losses_val = []
all_vae_losses_val = []
disent_losses_val = []
mean_losses_val = []
cov_losses_val = []
subset_losses_val = []


os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/*.png')
# os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/*.png')
perception_loss_clf_func = losses.PerceptionLossVGG(device)
pixel_loss = losses.PixelwiseLoss(mode='l1')
clf_loss_func = losses.clf_orientedLoss("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", device)
subset_loss_func = nn.BCEWithLogitsLoss()
focal_loss = losses.FocalLoss()
gen_weight = 0.01
for epoch in range(epochs):
    k = 0
    # altered_index = torch.randint(0, latent_dim, (1,)).item()
    torch.cuda.empty_cache()

    running_recon = 0
    running_kld = 0
    running_gen = 0
    running_clf = 0
    running_disent = 0
    running_cov = 0
    running_vae_all = 0
    running_subset = 0
    encod.eval()
    z_std, z_mean = utils.compute_latent_feature_variance(encod, train_loader, device)
    print(f"Epoch {epoch - 1} - latent mean avg: {z_mean.mean():.3f}, std avg: {z_std.mean():.3f}, {(z_std >= 0.1).sum()} latent features have std higher than 0.1")
    for imgs, labels in train_loader:
        
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
        imgs = imgs.to(device)
        ###Encoder + Decoder
        encod.train()
        decod.train()
        recog.train()
        subset_model.train()
        z_img, mu, logvars = encod(imgs)
        img_recon = decod(z_img)

        range_alteration = torch.from_numpy(np.random.choice(np.linspace(-4, 4, 70), imgs.shape[0])) ###Alter latent feature in range of std_range 
        range_alteration = range_alteration.to(device)
        
        kl_loss = -0.5 * torch.mean(1 + logvars - mu.pow(2) - logvars.exp())
        recon_loss = perception_loss_clf_func(img_recon, imgs)
        clf_loss = clf_loss_func(img_recon, imgs)
        #z_var_loss = losses.latent_z_variance_loss(z_img)
        z_var_loss = losses.covariance_loss(z_img)
        clf_target = torch.sigmoid(clf_model(imgs))  
        labels = labels.to(device)
        subset_loss = subset_loss_func(subset_model(z_img), clf_target) 
        z_alter = z_img.clone()
        for i in range(imgs.size(0)): ##change different latent feature per each image
            epsilon = range_alteration[i] * z_std[altered_indices[i]]
            z_alter[i, altered_indices[i]] = z_alter[i, altered_indices[i]] + epsilon ##change different latent feature per each image
        z_combined = torch.cat([z_img, z_alter], dim = 0)
        img_combined_recon = decod(z_combined)
        img_recon, alt_img_recon = torch.chunk(img_combined_recon, 2, dim=0)
        diff_img = (img_recon - alt_img_recon).abs()
        diff_img = 10.0 * diff_img  # scale up
        # plt.imshow(diff_img[0][0].detach().cpu().numpy() , cmap = 'hot')
        # plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/epoch_{epoch}_batch{k}")
        k += 1
        diff_img.to(device)
        logits = recog(diff_img[:, 0:1, :, :])
        disent_loss = F.cross_entropy(logits, altered_indices, label_smoothing= 0.05) ##change different latent feature per each image
        # print(diff_img.mean())
        total_vae_loss = 6 * recon_loss  + 6 * clf_loss + 2 * subset_loss + 15 * kl_loss + 3 * z_var_loss + 3 * disent_loss#+  #+ cf_loss # disent_loss
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        opt_subset.zero_grad()
        opt_recog.zero_grad()
        total_vae_loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        opt_subset.step()
        opt_recog.step()
        decod.eval()
        
        running_clf += clf_loss.item()
        running_recon += recon_loss.item()
        running_cov += z_var_loss.item()
        running_disent += disent_loss.item()
        running_vae_all += total_vae_loss.item()
        running_subset += subset_loss.item()
        running_kld += kl_loss.item()
       
    ###getting all losses in array
    num_batches = len(train_loader)
    recon_losses.append(running_recon/num_batches)
    kld_losses.append(running_kld/num_batches)
    clf_losses.append(running_clf/num_batches)
    cov_losses.append(running_cov/num_batches)
    disent_losses.append(running_disent/num_batches)
    all_vae_losses.append(running_vae_all/num_batches)
    subset_losses.append(running_subset/num_batches)
    print(f"Epoch {epoch}, recons loss is {recon_losses[-1]}, kd loss is {kld_losses[-1]}, clf loss is {clf_losses[-1]}, disentanglement loss in {disent_losses[-1]}, cov loss is {cov_losses[-1]}, subset loss is {subset_losses[-1]}, and ALL loss is {all_vae_losses[-1]}")
    # if epoch >= 9 and (epoch + 1) % 5 == 0:
    #     torch.save(encod, f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER_mri/models/encod_kld_version_train_1_epoch{epoch}.pt")
    #     torch.save(decod, f'/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER_mri/models/decod_kld_version_train_1_epoch{epoch}.pt')
    #     torch.save(recog, f'/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER_mri/models/recog_kld_version_train_1_epoch{epoch}.pt')
    #     torch.save(subset_model, f'/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER_mri/models/subset_kld_version_train_1_epoch{epoch}.pt')
    val_losses = utils.validate_model(
    encod, decod, recog, clf_model, subset_model,
    perception_loss_clf_func, clf_loss_func, subset_loss_func, losses,
    val_loader, latent_dim, std_range,device)
    
    recon_losses_val.append(val_losses['recon'])
    clf_losses_val.append(val_losses['clf'])
    # mean_losses.append(running_mean/64)
    cov_losses_val.append(val_losses['cov'])
    disent_losses_val.append(val_losses['disent'])
    all_vae_losses_val.append(val_losses['vae_all'])
    subset_losses_val.append(val_losses['subset'])
    print(f"Validation - Epoch {epoch}, recon loss: {val_losses['recon']}" +
     f"clf loss: {val_losses['clf']}, disent loss: {val_losses['disent']}, " +
      f"cov loss: {val_losses['cov']}, subset loss: {val_losses['subset']}, total VAE loss: {val_losses['vae_all']}")


plt.plot(recon_losses)
plt.plot(kld_losses)
plt.plot(disent_losses)
plt.plot(cov_losses)
plt.plot(clf_losses)
plt.plot(subset_losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon', 'kld', 'disent', 'cov', 'clf', 'subset'], loc='upper left')
plt.savefig('/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/kld_version_train_1.png')
plt.close()


plt.plot(recon_losses_val)
# plt.plot(all_vae_losses)
plt.plot(disent_losses_val)
plt.plot(cov_losses_val)
plt.plot(clf_losses_val)
plt.plot(subset_losses_val)

plt.title('model validation losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon', 'disent', 'cov', 'clf', 'subset'], loc='upper left')
plt.savefig('/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/kld_version_val_1.png')
plt.close()

