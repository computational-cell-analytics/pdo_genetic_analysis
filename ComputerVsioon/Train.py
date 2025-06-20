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

###getting data
# train_paths = glob("/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/**/*.jpg", recursive = True)
train_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/NonDemented/*.jpg")
train_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/VeryMildDemented/*.jpg")
train_paths = np.array(train_no_paths + train_very_paths)
test_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/NonDemented/*.jpg")
test_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/VeryMildDemented/*.jpg")
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
        img = read_image(self.img_paths[index])
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

batch_size = 64
train_loader = DataLoader(train_img_normal_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_img_normal_dataset, batch_size = 64, shuffle = True)

###Cretae models
latent_size = 350
encod = models.Encoder(3, 0.03)
decod = models.Decoder(350, 0.03)
disc = models.Discriminator(350)
recog = models.Disentangler()
imgnet_model = vgg19(VGG19_Weights.IMAGENET1K_V1)
subset_model = models.simple_neuron()
clf_model = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", weights_only = False, map_location=device).eval()

encod.to(device)
decod.to(device)
disc.to(device)
imgnet_model.to(device)
recog.to(device)
subset_model.to(device)
 
lr = 0.0001 
weight_decay = 1e-4
opt_vae = optim.Adam(list(encod.parameters()) + list(decod.parameters()), lr=lr)
opt_disc = optim.AdamW(disc.parameters(), lr=lr/5, weight_decay = weight_decay)
opt_gen = optim.Adam(encod.parameters(), lr=lr/5)
opt_recog = optim.Adam(recog.parameters(), lr = lr)
opt_subset = optim.Adam(subset_model.parameters(), lr = lr/5)


# === Hyperparameters ===
epochs = 50
batch_size = 64
latent_dim = 350
lr_vae = 1e-4
lr_disc = 2e-5
lr_gen = 2e-5
adv_weight = 0.05
var_weight = 0.0001
alpha = 0.5  # for hybrid loss
warmup_epochs = 5
std_range = 1.5




recon_losses = []
disc_losses = []
gen_losses = []
clf_losses = []
disent_losses = []
all_vae_losses = []
mean_losses = []
cov_losses = []
subset_losses = []
recon_losses_test = []
disc_losses_test = []
gen_losses_test = []
clf_losses_test = []
all_vae_losses_test = []

perception_loss_clf_func = losses.PerceptionLossVGG(device)
pixel_loss = losses.PixelwiseLoss(mode='l1')
clf_loss_func = losses.clf_orientedLoss("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", device)
subset_loss_func = nn.BCEWithLogitsLoss()
gen_weight = 0.01
for epoch in range(epochs):
    # altered_index = torch.randint(0, latent_dim, (1,)).item()
    running_recon = 0
    running_disc = 0
    running_gen = 0
    running_clf = 0
    running_disent = 0
    running_cov = 0
    running_vae_all = 0
    running_subset = 0
    for imgs, labels in train_loader:
        altered_index = torch.randint(0, latent_dim, (1,)).item()
        imgs = imgs.to(device)
        ###Encoder + Decoder
        disc.eval()
        encod.train()
        decod.train()
        z_img = encod(imgs)
        img_recon = decod(z_img)
        z_std = torch.std(z_img, dim = 0).to(device)
        # std_range = std_range.to(device)
        range_alteration = (torch.rand(latent_dim) * 2 * std_range) - std_range
        range_alteration = range_alteration.to(device)
        # z_alter = torch.ones((imgs.shape[0], latent_dim)).to(device) #z_img
        z_alter = z_img.clone()
        z_alter[:, altered_index] = z_alter[:, altered_index] + z_std[altered_index] * range_alteration[altered_index]
        alt_img_recon = decod(z_alter)
        diff_img = img_recon - alt_img_recon
        # diff_img.requires_grad = True
        diff_img.to(device)
        logits = recog(diff_img)
        # altered_label = torch.zeros(latent_dim)
        # altered_label[altered_index] = 1
        # altered_labels = altered_label.repeat(imgs.shape[0], 1)
        altered_labels = torch.full((imgs.size(0),), altered_index, dtype=torch.long, device=device)
        disent_loss = F.cross_entropy(logits, altered_labels)


        altered_labels = altered_labels.to(device)
        disent_loss = F.cross_entropy(logits, altered_labels)
        recon_loss = perception_loss_clf_func(img_recon, imgs)
        clf_loss = clf_loss_func(img_recon, imgs)
        z_var_loss = losses.latent_z_variance_loss(z_img)
        with torch.no_grad():
            clf_target = torch.sigmoid(clf_model(imgs))  
        subset_loss = subset_loss_func(subset_model(z_img), clf_target)
        total_vae_loss = recon_loss  + clf_loss + disent_loss + var_weight * z_var_loss + subset_loss
        opt_vae.zero_grad()
        opt_recog.zero_grad()
        total_vae_loss.backward()
        opt_vae.step()
        opt_recog.step()
        opt_subset.step()
        running_clf += clf_loss.item()
        running_recon += recon_loss.item()
        running_cov += var_weight * z_var_loss.item()
        running_disent += disent_loss.item()
        running_vae_all += total_vae_loss.item()
        running_subset += subset_loss.item()
        ###Discr
        encod.eval()
        decod.eval()
        disc.train()
        z_img= encod(imgs)
        #disc_loss = losses.Discriminator_loss(disc, z_img, 64, 350, device)
        normal_dist = torch.randn( (batch_size, latent_size)).to(device)
        # disc_loss = Discrim_loss((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
        # disc_loss = -torch.mean((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
        disc_loss = losses.Discriminator_loss(disc(normal_dist), disc(z_img))
        if epoch >= 5:
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
        running_disc += disc_loss.item()
        ##Generatort (Advarserial)
        encod.train()
        decod.eval()
        disc.eval()
        z_img_new = encod(imgs)
        gen_loss = gen_weight * (-torch.mean(torch.log(disc(z_img_new))))
        if epoch >= 5:
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
        running_gen += gen_loss.item()
        ##Classification Oriented training
        
        # running_vae_all += ru/4
    ###getting all losses in array
    recon_losses.append(running_recon/64)
    disc_losses.append(running_disc/64)
    gen_losses.append(running_gen/64)
    clf_losses.append(running_clf/64)
    # mean_losses.append(running_mean/64)
    cov_losses.append(running_cov/64)
    disent_losses.append(running_disent/64)
    all_vae_losses.append(running_vae_all/64)
    subset_losses.append(running_subset/64)
    print(f"Epoch {epoch}, recons loss is {recon_losses[-1]}, discriminator loss is {disc_losses[-1]}, generator loss is {gen_losses[-1]}, clf loss is {clf_losses[-1]}, disent loss is {disent_losses[-1]}, cov loss is {cov_losses[-1]}, subset loss is {subset_losses[-1]}, and ALL loss is {all_vae_losses[-1]}")
    running_recon = 0
    running_disc = 0
    running_gen = 0
    running_all = 0
    for imgs, labels in test_loader:
        with torch.no_grad():
            disc.eval()
            encod.eval()
            decod.eval()
            imgs = imgs.to(device)
            ###Encoder + Decoder
            z_img= encod(imgs)
            img_recon = decod(z_img)
            #recon_loss = pixel_loss(img_recon, imgs)
            recon_loss = perception_loss_clf_func(img_recon, imgs)
            running_recon += recon_loss.item()
            ###Discr
            normal_dist = torch.randn( (batch_size, latent_size)).to(device)
            disc_loss = -torch.mean((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
            running_disc += disc_loss.item()
            ##Generatort (Advarserial)
            gen_loss = gen_weight * (-torch.mean(torch.log(disc(z_img))))
            running_gen += gen_loss.item()
            ###Classification loss
            clf_loss = clf_loss_func(img_recon, imgs)
            running_clf += clf_loss.item()
            # running_all += running_recon + running_disc + running_gen + running_clf
    recon_losses_test.append(running_recon/64)
    disc_losses_test.append(running_disc/64)
    gen_losses_test.append(running_gen/64)
    clf_losses_test.append(running_clf/64)
    all_vae_losses_test.append(running_all/64)
    # print(f"Epoch {epoch}, recons loss valid is {recon_losses_test[-1]}, discriminator loss valid is {disc_losses_test[-1]}, generator loss valid is {gen_losses_test[-1]}, and ALL loss valid is {all_losses_test[-1]}")


plt.plot(recon_losses)
plt.plot(disc_losses)
plt.plot(gen_losses)
# plt.plot(all_vae_losses)
plt.plot(disent_losses)
plt.plot(cov_losses)
plt.plot(clf_losses)
plt.plot(subset_losses)
# plt.plot(all_losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon', 'disc', 'gen', 'disent', 'cov', 'clf', 'subset'], loc='upper left')
plt.savefig('losses_Sush_6_wdecay_weight_warmup_clf_mean_cov_subset_v3.png')

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
with open("results_subset.txt",'w') as file:
    for i in range(len(recon_losses)):
        file.write(f"recons loss is {recon_losses[i]}, disc loss is {disc_losses[i]}, gen loss is {gen_losses[i]}, disent loss is{disent_losses[i]}, cov loss is{cov_losses[i]}, clf loss is{clf_losses[i]}" + '\n')

torch.save(encod, "encod_Sush_6_wdecay_weight_warmup_clf_mean_cov_subset_v3.pt")
torch.save(decod, 'decod_Sush_6_wdecay_weight_warmup_clf_mean_cov_subset_v3.pt')
