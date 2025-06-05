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

###getting data
# train_paths = glob("/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/**/*.jpg", recursive = True)
train_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/NonDemented/*.jpg")
train_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/train/VeryMildDemented/*.jpg")
train_paths = train_no_paths + train_very_paths
test_no_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/NonDemented/*.jpg")
test_very_paths = glob("/user/sina.garazhian/u12203/kaggle_alz/test/VeryMildDemented/*.jpg")
test_paths = test_no_paths + test_very_paths
print('number of train paths', len(train_paths))
train_labels = np.array(['Non' not in path.split('/')[-2] for path in train_paths]) * 1
test_labels = np.array(['Non' not in path.split('/')[-2] for path in test_paths]) * 1


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


train_loader = DataLoader(train_img_normal_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_img_normal_dataset, batch_size = 64, shuffle = True)

###Cretae models

encod = models.Encoder(3, 0.03)
decod = models.Decoder(350, 0.03)
disc = models.Discriminator(350)
imgnet_model = vgg19(VGG19_Weights.IMAGENET1K_V1)

encod.to(device)
decod.to(device)
disc.to(device)
imgnet_model.to(device)
 
lr = 0.0005 
opt_vae = optim.Adam(list(encod.parameters()) + list(decod.parameters()), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=0.0001)
opt_gen = optim.Adam(encod.parameters(), lr=0.0001)


epochs = 5





recon_losses = []
disc_losses = []
gen_losses = []
all_losses = []
layers_losses = []
perception_loss_clf_func = losses.PerceptionLossVGG(device)
pixel_loss = losses.PixelwiseLoss(mode='l1')
for epoch in range(epochs):
    running_recon = 0
    running_disc = 0
    running_gen = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        z_img = encod(imgs)
        ###discriminator
        encod.eval()
        decod.eval()
        disc.train()
        disc_loss = losses.Discriminator_loss(disc, z_img, 64, 350, device)
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()
        running_disc += disc_loss.item()
        ##vae
        disc.eval()
        encod.train()
        decod.train()
        z_img= encod(imgs)
        img_recon = decod(z_img)
        #recon_loss = perception_loss_clf_func(imgs, img_recon) #+ losses.adv_loss(disc, z_img_new)
        
        recon_loss = pixel_loss(img_recon, imgs)
        #layer_losses = perception_loss_clf_func(imgs, img_recon)[1]
        #layers_losses.append(layer_losses)
        # recon_loss.requires_grad = True
        opt_vae.zero_grad()
        recon_loss.backward()
        opt_vae.step()
        running_recon += recon_loss.item()
        ##adv loss
        encod.train()
        decod.train()
        disc.eval()
        z_img_new = encod(imgs)
        gen_loss = losses.adv_loss(disc, z_img_new)
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        running_gen += gen_loss.item()





        # img_recon = decod(z_img)
        
        # recon_loss = 5 * perception_loss_clf_func(imgs, img_recon)
        # recon_loss.requires_grad = True
        # opt_vae.zero_grad()
        # recon_loss.backward()
        # opt_vae.step()
        # running_recon += recon_loss.item()
        # ###Discriminator
        # encod.eval()
        # disc_loss = losses.Discriminator_loss(disc, z_img, 64, 350, device)
        # opt_disc.zero_grad()
        # disc_loss.backward()
        # opt_disc.step()
        # running_disc += disc_loss.item()
        # ###Gene (encoder)
        # encod.train()
        # z_img_new = encod(imgs)
        # gen_loss = losses.adv_loss(disc, z_img_new)
        # opt_gen.zero_grad()
        # gen_loss.backward()
        # opt_gen.step()
        # running_gen += gen_loss.item()
        # running_gen += 0
        running_all = running_recon + running_disc + running_gen
    recon_losses.append(running_recon/64)
    disc_losses.append(running_disc/64)
    gen_losses.append(running_gen/64)
    all_losses.append(running_all/64)
    print(f"Epoch {epoch}, recons loss is {recon_losses[-1]}, discriminator loss is {disc_losses[-1]}, generator loss is {gen_losses[-1]}, and ALL loss is {all_losses[-1]}")
    




plt.plot(recon_losses)
plt.plot(disc_losses)
plt.plot(gen_losses)
plt.plot(all_losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon', 'disc', 'gen', 'all'], loc='upper left')
plt.savefig('losses_spectral_3.png')

# plt.plot(i[0].item() for i in layers_losses)
# plt.plot(i[1].item() for i in layers_losses)
# plt.plot(i[2].item() for i in layers_losses)
# # plt.plot(all_losses)
# plt.title('layers losses')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['layer1', 'layer2', 'layer3'], loc='upper left')
# plt.savefig('layer_losses_spectral_3.png')


torch.save(encod, "encod_spectral_pixel.pt")
torch.save(decod, 'decod_spectral_pixel.pt')
