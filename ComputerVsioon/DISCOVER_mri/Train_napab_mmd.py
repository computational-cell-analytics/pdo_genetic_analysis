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
import visulize
import prepare_dataset
import prepare_model

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

    z_std = compute_latent_feature_variance(encod, valid_loader, device)

    with torch.no_grad():
        for imgs, labels, _, _ in valid_loader:
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
       
            plt.savefig(f"/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/{epoch}.png")
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

###getting data
data_alive_train = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_alive_train.pt", weights_only=False)
data_alive_test = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_alive_test.pt", weights_only=False)
data_dead_train = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_dead_train.pt", weights_only=False)
data_dead_test = torch.load("/user/sina.garazhian/u12203/panc_cell/datasets/napab_dead_test.pt", weights_only=False)

print(f'Alive train cells are {len(data_alive_train)}')
print(f'Alive test cells are {len(data_alive_test)}')
print(f'dead train cells are {len(data_dead_train)}')
print(f'dead test cells are {len(data_dead_test)}')


train_dataset = torch.utils.data.ConcatDataset([data_dead_train, data_alive_train])
train_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
test_dataset =  torch.utils.data.ConcatDataset([data_dead_test, data_alive_test])
###create dataset torch object





    
device = "cuda" if torch.cuda.is_available() else "cpu"

###Create dataset and dataloader instances
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

###Cretae models
latent_size = 190
latent_dim = 190
encod = models.Encoder(1, 0.03,latent_size)
decod = models.Decoder(latent_size ,1, 0.03)
disc = models.Discriminator(190)
recog = models.Recognizer(latent_dim)
imgnet_model = vgg19(VGG19_Weights.IMAGENET1K_V1)
subset_model = models.simple_neuron()
recog_attend = models.CrossAttentionRecognizer()
clf_model = torch.load("/user/sina.garazhian/u12203/panc_cell/best_clf.pt", weights_only = False, map_location=device).eval()

encod.to(device)
decod.to(device)
disc.to(device)
imgnet_model.to(device)
recog.to(device)
subset_model.to(device)
recog_attend.to(device)
 
lr = 0.0001 
weight_decay = 1e-4
opt_vae = optim.AdamW(list(encod.parameters()) + list(decod.parameters()), lr=lr)
opt_disc = optim.AdamW(disc.parameters(), lr=lr/5, weight_decay = weight_decay)
opt_gen = optim.AdamW(encod.parameters(), lr=lr/5)
opt_recog = optim.AdamW(recog.parameters(), lr = lr )
opt_subset = optim.AdamW(subset_model.parameters(), lr = lr/5)
opt_recog_attend = optim.AdamW(recog_attend.parameters(), lr=1e-4, weight_decay=1e-4)


# === Hyperparameters ===
epochs = 100
batch_size = 128
lr_vae = 1e-4
lr_disc = 2e-5
lr_gen = 2e-5
adv_weight = 0.05
var_weight = 0.1
alpha = 0.5  # for hybrid loss
warmup_epochs = 5
std_range = 3


def compute_latent_feature_variance(encoder, dataloader, device):
    encoder.eval()  # Set encoder to evaluation mode
    all_latents = []

    with torch.no_grad():
        for imgs, _ , _, _ in dataloader:
            imgs = imgs.to(device)
            z = encoder(imgs)  # shape: [batch_size, latent_dim]
            all_latents.append(z.cpu())  # move to CPU to avoid GPU memory issues

    # Concatenate all latent vectors: shape [N_samples, latent_dim]
    all_latents = torch.cat(all_latents, dim=0)

    # Compute variance per feature (dim=0 is across samples)
    feature_variances = torch.std(all_latents, dim=0, unbiased=True)  # shape: [latent_dim]
    feature_means = torch.mean(all_latents, dim=0)

    return feature_variances, feature_means

recon_losses = []
disc_losses = []
gen_losses = []
clf_losses = []
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


os.system('rm ~/lustere-grete-mine/DISCOWER/diff_imgs/*')
os.system('rm /user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/*.png')
perception_loss_clf_func = losses.PerceptionLossVGG(device)
pixel_loss = losses.PixelwiseLoss(mode='l1')
clf_loss_func = losses.clf_orientedLoss("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", device)
subset_loss_func = nn.BCEWithLogitsLoss()
focal_loss = losses.FocalLoss()
gen_weight = 0.01
for epoch in range(epochs):
    # altered_index = torch.randint(0, latent_dim, (1,)).item()
    torch.cuda.empty_cache()

    running_recon = 0
    running_mmd = 0
    running_gen = 0
    running_clf = 0
    running_disent = 0
    running_cov = 0
    running_vae_all = 0
    running_subset = 0
    encod.eval()
    z_std, z_mean = compute_latent_feature_variance(encod, train_loader, device)
    print(f"Epoch {epoch - 1} - latent mean avg: {z_mean.mean():.3f}, std avg: {z_std.mean():.3f}, {(z_std >= 0.1).sum()} latent features have std higher than 0.1")
    for imgs, labels , _, _ in train_loader:
        
        # altered_index = torch.randint(0, latent_dim, (1,)).item()
        altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
        imgs = imgs.to(device)
        ###Encoder + Decoder
        disc.eval()
        encod.train()
        decod.train()
        recog.train()
        subset_model.train()
        z_img = encod(imgs)
        img_recon = decod(z_img)
        # z_std = torch.std(z_img, dim = 0).to(device)
        # std_range = std_range.to(device)
        range_alteration = (torch.rand(latent_dim) * 2 * std_range) - std_range
        range_alteration = range_alteration.to(device)
        z_alter = z_img.clone()
        #ange_corrected = torch.from_numpy(np.random.choice([-3, 3], 1)).to(device)
        # z_alter[:, altered_index] = z_alter[:, altered_index] + z_std[altered_index] * range_alteration[altered_index] #range_corrected # * range_alteration[altered_index]
        for i in range(imgs.size(0)): ##change different latent feature per each image
            epsilon = torch.empty(1).uniform_(-std_range, std_range).item()
            z_alter[i, altered_indices[i]] += epsilon ##change different latent feature per each image
        alt_img_recon = decod(z_alter)
        diff_img = (img_recon - alt_img_recon).abs()
        # diff_img.requires_grad = True
        diff_img.to(device)
        diff_img = (diff_img - diff_img.mean()) / (diff_img.std() + 1e-6)
        # print(diff_img[:, 0:1, :, :].shape)
        # logits = recog_attend(diff_img)
        logits = recog(diff_img[:, 0:1, :, :])
       
        # print("Logits stats:", logits.shape, logits.min().item(), logits.max().item())
        # altered_label = torch.zeros(latent_dim)
        # altered_label[altered_index] = 1
        # # altered_labels = altered_label.repeat(imgs.shape[0], 1)
        # altered_labels = torch.full((imgs.size(0),), altered_index, dtype=torch.long, device=device)
        # disent_loss = F.cross_entropy(logits, altered_labels)
        disent_loss = F.cross_entropy(logits, altered_indices, label_smoothing=0.1) ##change different latent feature per each image
        disent_loss = focal_loss(logits, altered_indices)
        # with torch.no_grad():
        #     preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        #     print(f"Predicted indices: {preds[:5]}")
        #     print(f"Ground truth: {altered_indices[:5]}")
        # plt.imshow(diff_img[0][0].detach().cpu().squeeze(), cmap='hot')
        # plt.title(f"Actual change is {altered_indices[0]}, Predicted is {torch.argmax(logits, dim = 1)[0]}")
        # plt.savefig(f"/mnt/lustre-grete/usr/u12203/DISCOWER/diff_imgs/{epoch}_{disent_loss.item()}.png")
        # plt.show()
        # plt.close()
        # selected_indices = torch.randperm(latent_dim)[:7].tolist()
        # cf_loss = losses.counterfactual_consistency_loss(encod, decod, clf_model, imgs, 
        #     z_std, selected_indices, epsilon=1.5, device='cuda')
        #altered_labels = altered_labels.to(device)
        # disent_loss = F.cross_entropy(logits, altered_labels, label_smoothing=0.05)
        recon_loss = perception_loss_clf_func(img_recon, imgs)
        clf_loss = clf_loss_func(img_recon, imgs)
        #z_var_loss = losses.latent_z_variance_loss(z_img)
        z_var_loss = losses.covariance_loss(z_img)
        with torch.no_grad():
            clf_target = torch.sigmoid(clf_model(imgs))  
        subset_loss = subset_loss_func(subset_model(z_img), clf_target) #+ 0.0001 * z_var_loss
        # opt_recog_attend.zero_grad()
        
        # disent_loss.backward(retain_graph=True)
        
        # opt_recog.step()
        # z_prior = torch.randn_like(z_img)
        mmd_loss = losses.mmd_loss_adaptive(z_img)
        total_vae_loss = 5 * recon_loss  + 5 * clf_loss + subset_loss + 5000 * mmd_loss + 0.01 * z_var_loss + disent_loss#+  #+ cf_loss # disent_loss
        opt_recog.zero_grad()
        opt_vae.zero_grad()
        opt_subset.zero_grad()
        total_vae_loss.backward()
        total_norm = 0.0
        for p in recog.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        # total_norm = total_norm ** 0.5
        # print(f"Gradient Norm: {total_norm:.6f}")

        opt_vae.step()
        opt_subset.step()
        opt_recog.step()
        running_clf += clf_loss.item()
        running_recon += recon_loss.item()
        running_cov += z_var_loss.item()
        running_disent += disent_loss.item()
        running_vae_all += total_vae_loss.item()
        running_subset += subset_loss.item()
        running_mmd += mmd_loss.item()
        # with torch.no_grad():
        #     for x_batch, _ in val_loader:  # or a fixed subset
        #         z_batch = encod(x_batch.to(device))
        #           # Just one batch per epoch
        #         visulize.plot_latent_histogram(z_batch, epoch, dims_to_plot=[0, 1, 2, 3, 4, 5, 15, 20, 30, 40, 50, 60, 100, 150], bins=50)
                

        #         break        

        ###Discr
        # encod.eval()
        # decod.eval()
        # disc.train()
        # z_img= encod(imgs)
        # #disc_loss = losses.Discriminator_loss(disc, z_img, 64, 350, device)
        # normal_dist = torch.randn( (batch_size, latent_size)).to(device)
        # # disc_loss = Discrim_loss((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
        # # disc_loss = -torch.mean((torch.log(disc(normal_dist) + 1e-8) + torch.log(1 - disc(z_img) + 1e-8)))
        # disc_loss = losses.Discriminator_loss(disc(normal_dist), disc(z_img))
        # if epoch >= 5:
        #     opt_disc.zero_grad()
        #     disc_loss.backward()
        #     opt_disc.step()
        # running_disc += disc_loss.item()
        # ##Generatort (Advarserial)
        # encod.train()
        # decod.eval()
        # disc.eval()
        # z_img_new = encod(imgs)
        # gen_loss = gen_weight * (-torch.mean(torch.log(disc(z_img_new))))
        # if epoch >= 5:
        #     opt_gen.zero_grad()
        #     gen_loss.backward()
        #     opt_gen.step()
        # running_gen += gen_loss.item()
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
    recon_losses.append(running_recon/64)
    mmd_losses.append(running_mmd/64)
    gen_losses.append(running_gen/64)
    clf_losses.append(running_clf/64)
    # mean_losses.append(running_mean/64)
    cov_losses.append(running_cov/64)
    disent_losses.append(running_disent/64)
    all_vae_losses.append(running_vae_all/64)
    subset_losses.append(running_subset/64)
    print(f"Epoch {epoch}, recons loss is {recon_losses[-1]}, mmd loss is {mmd_losses[-1]}, clf loss is {clf_losses[-1]}, disentanglement loss in {disent_losses[-1]}, cov loss is {cov_losses[-1]}, subset loss is {subset_losses[-1]}, and ALL loss is {all_vae_losses[-1]}")
    # val_losses = validate_model(
    # encod, decod, disc, recog, clf_model, subset_model,
    # perception_loss_clf_func, clf_loss_func, subset_loss_func, losses,
    # val_loader, latent_dim, std_range, var_weight, gen_weight,
    # epoch, device
    #     )
    # recon_losses_test.append(val_losses['recon'])
    # disc_losses_test.append(val_losses['disc'])
    # gen_losses_test.append(val_losses['gen'])
    # clf_losses_test.append(val_losses['clf'])
    # # mean_losses.append(running_mean/64)
    # cov_losses_test.append(val_losses['cov'])
    # disent_losses_test.append(val_losses['disent'])
    # all_vae_losses_test.append(val_losses['vae_all'])
    # subset_losses_test.append(val_losses['subset'])
    # print(f"Validation - Epoch {epoch}, recon loss: {val_losses['recon']}, discriminator loss: {val_losses['disc']}, "
    #   f"generator loss: {val_losses['gen']}, clf loss: {val_losses['clf']}, disent loss: {val_losses['disent']}, "
    #   f"cov loss: {val_losses['cov']}, subset loss: {val_losses['subset']}, total VAE loss: {val_losses['vae_all']}")

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
plt.plot(mmd_losses)
# plt.plot(gen_losses)
# plt.plot(all_vae_losses)
plt.plot(disent_losses)
plt.plot(cov_losses)
plt.plot(clf_losses)
plt.plot(subset_losses)
# plt.plot(all_losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['recon', 'mmd', 'disent', 'cov', 'clf', 'subset'], loc='upper left')
plt.savefig('/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/models/mmd_version_train_1.png')
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

torch.save(encod, "/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/encod_mmd_version_train_1.pt")
torch.save(decod, '/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/decod_mmd_version_train_1.pt')
torch.save(subset_model, '/user/sina.garazhian/u12203/lustere-grete-mine/DISCOWER/napab_models/subset_mmd_version_train_1.pt')
