###Define Resnetblock downsample
import torch
from torch import nn
from torchvision.transforms.v2 import GaussianNoise
import torch.nn.functional as F
import numpy as np 
import losses
std = 0.03


class resblock_down_sample(nn.Module):
    def __init__(self, in_feat, num_feat, std):
        super(resblock_down_sample, self).__init__()
        self.in_feat = in_feat
        self.num_feat = num_feat
        self.std = std
        self.down_sample = nn.Sequential(
        nn.Conv2d(self.in_feat, self.num_feat, 3, 2, 1)
        ,GaussianNoise(self.std)
        ,nn.BatchNorm2d(self.num_feat)
        ,nn.ReLU()
        ,nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 'same')
        ,GaussianNoise(self.std)
        ,nn.BatchNorm2d(self.num_feat)
        ,nn.ReLU())
        self.res = nn.Conv2d(self.in_feat, self.num_feat, 1, 2)
    def forward(self, x):
        inp = x
        x = self.down_sample(x)
        d_res = self.res(inp)
        x = x + d_res
        return x


class resblock_up_sample(nn.Module):
    def __init__(self, in_feat, num_feat, std = 0):
        super(resblock_up_sample, self).__init__()
        self.in_feat = in_feat
        self.num_feat = num_feat
        self.std = std
        self.up_block = nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(self.in_feat, self.num_feat, 3, 1, 'same'),
        nn.BatchNorm2d(self.num_feat),
        GaussianNoise(self.std),
        nn.ReLU(),
        nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 'same'),
        nn.BatchNorm2d(self.num_feat),
        GaussianNoise(self.std),
        nn.ReLU())
        self.res = nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(self.in_feat, self.num_feat, 3, 1, 'same'))
        
    def forward(self, x):
        inp = x
        x = self.up_block(x)
        d_res = self.res(inp)
        x = x + d_res
        return x


###build encoder
class Encoder(nn.Module):
    def __init__(self, inp_ch, std, latent_size):
        super(Encoder, self).__init__()
        self.std = std
        self.latent_size = latent_size
        self.input = nn.Conv2d(inp_ch, 64, 3, 1, 'same')
        self.block1 = resblock_down_sample(64, 512, std)
        self.block2 = resblock_down_sample(512, 1024, std)
        self.block3 = resblock_down_sample(1024, 1024, std)
        self.block4 = resblock_down_sample(1024, 1024, std)
        self.denses = nn.Sequential(nn.Flatten(), GaussianNoise(self.std), nn.Linear(16384, self.latent_size), nn.Dropout(0.1), nn.SiLU(),
                                     nn.BatchNorm1d(self.latent_size))
        self.avg_layer = nn.Linear(16384, latent_size)
        self.std_layer = nn.Linear(16384, latent_size)
    def reparametrize_trick(self, mu, logvars):
        rand_dist = torch.randn_like(logvars)
        z_latent = (torch.exp(0.5 * logvars) * rand_dist) + mu
        return z_latent
    
    def forward(self, x):
        # print('input', x.shape)
        x = self.input(x)
        # print('after in[', x.shape)
        # x = nn.ReLU()(x) ##commented
        x = self.block1(x)
        # print('b1', x.shape)
        x = self.block2(x)
        # print('b2',x.shape)
        x = self.block3(x)
        # print('b3',x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = torch.flatten(x, 1, -1)
        # print(x.shape)
        mu = self.avg_layer(x)
        logvars = self.std_layer(x)
        x = self.reparametrize_trick(mu, logvars)
        # print(x.shape)
        # print('b4',x.shape)
        # x = self.denses(x)
        return x, mu, logvars


class Decoder(nn.Module):
    def __init__(self, latent_shape, output_channel, std):
        super(Decoder, self).__init__()
        self.std = std
        self.latent = latent_shape
        self.input = nn.Linear(self.latent, 16384)
        self.xattn0 = CrossAttn2d(1024, latent_shape)
        self.block1 = resblock_up_sample(1024, 1024, std)
        self.block2 = resblock_up_sample(1024, 1024, std)
        self.block3 = resblock_up_sample(1024, 512, std)
        self.block4 = resblock_up_sample(512, 512, std)
        self.to_img = nn.Sequential(nn.Conv2d(512, output_channel, 3, 1, 'same'), nn.Sigmoid())
        
    def forward(self, x):
        z = x
        x =  self.input(x)
        
        # print(x.shape)
        x = GaussianNoise(self.std)(x)
        x = torch.relu(x)
        # x = nn.Flatten()(x)
        x = x.view(-1, 1024, 4, 4)
        # x = self.xattn0(x, z)
        # print(x.shape)
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = self.to_img(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, inp_size):
        super(Discriminator, self).__init__()
        self.inp_size = inp_size
        self.denses = nn.Sequential(nn.utils.spectral_norm(nn.Linear(inp_size, 1024)),
            #nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 2048)),
            #nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            # nn.utils.spectral_norm(nn.Linear(2048, 2048)),
            # nn.BatchNorm1d(2048),
            # nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2048, 2048)),
            nn.Dropout(0.35), ##just added
            #nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2048, 1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.Dropout(0.5),
            nn.utils.spectral_norm(nn.Linear(512, 1)),
            nn.Sigmoid())
    def forward(self, x):
        x = self.denses(x)
        return x


class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        self.model = model
        self.clf = nn.ModuleList([nn.ReLU(), nn.Linear(512, 128),
                                  nn.ReLU(), nn.Linear(128, 32),
                                  nn.ReLU(), nn.Linear(32, 1)])
        #self.sig = 
        self.clf1 = nn.ModuleList([nn.ReLU(), nn.Linear(16, 1)])

    def forward(self, x):
        x = self.model(x)
        x = self.clf1[0](x)
        x = self.clf1[1](x)
        # for i in self.clf:
        #     x = i(x)
        # x = torch.nn.Sigmoid(x)
        return x


class Disentangler(nn.Module):
    def __init__(self, in_channels=1, num_classes=350):
        super().__init__()
        std = 0
        self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                                              nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.res1 = resblock_down_sample(64, 128, std)
        self.res2 = resblock_down_sample(128, 256, std)
        self.res3 = resblock_down_sample(256, 512, std)
        self.res4 = resblock_down_sample(512, 1024, std)
        self.mlp = nn.Sequential(nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x = self.initial(x)  # (64,64,64)
        # x = F.Relu(x) ###Commented
        x = self.res1(x)             # (512,32,32)
        x = self.res2(x)             # (512,16,16)
        x = self.res3(x)             # (1024,8,8)
        x = self.res4(x)             # (1024,4,4)
        x = torch._adaptive_avg_pool2d(x, 1)
        # print(x.shape)
        # x = self.swish(x)
        x = self.mlp(x)               # (350,)
        return x 

class Recognizer(nn.Module):
    def __init__(self, latent_dim, in_channels=1):
        super(Recognizer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),           # -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),          # -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),         # -> 8x8
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 512)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.output_logits = nn.Linear(512, latent_dim)  # Softmax will be applied in loss
        # self.value_head = nn.Linear(512, 1)              # Optional regression output

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        logits = self.output_logits(x)
        # value = self.value_head(x)
        return logits





class simple_neuron(nn.Module):
    def __init__(self, sub_features = 14):
        super().__init__()
        self.sub_features = sub_features
        self.simple_clf = nn.Linear(self.sub_features, 1)

    def forward(self, x):
        x = self.simple_clf(x[:, :self.sub_features])  # (64,64,64)
        return x



class CrossAttentionRecognizer(nn.Module):
    def __init__(self, latent_dim=350, img_channels=3, feature_dim=256):
        super().__init__()
        # CNN encoder for 64x64 images
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # [B, 128, 8, 8]
            nn.ReLU(),
            nn.Flatten(),                                                     # [B, 8192]
        )
        self.feature_proj = nn.Linear(128 * 8 * 8, feature_dim)               # [B, D]

        # Learnable query per latent dim (cross-attn)
        self.latent_queries = nn.Parameter(torch.randn(latent_dim, feature_dim))  # [latent_dim, D]

        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Linear(feature_dim, 1)  # Classify each query's attended result
        

    def forward(self, diff_img):
        B = diff_img.size(0)
        x = self.encoder(diff_img)                    # [B, 8192]
        x = self.feature_proj(x).unsqueeze(1)         # [B, 1, D]
        queries = self.latent_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, latent_dim, D]
        attended, _ = self.cross_attn(queries, x, x)                # [B, latent_dim, D]
        logits = self.classifier(attended).squeeze(-1)              # [B, latent_dim]
        return logits

import torch, torch.nn as nn, torch.nn.functional as F

class CrossAttn2d(nn.Module):
    def __init__(self, in_channels, z_dim, heads=4, dim_head=64, tokens=32, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Conv2d(in_channels, heads*dim_head, 1)     # queries from feature map
        self.to_kv_tokens = nn.Linear(z_dim, tokens*2*dim_head)   # keys/values from z → tokens
        self.proj = nn.Conv2d(heads*dim_head, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        # learnable gate (start at 0 → safe/stable)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):
        B, C, H, W = x.shape
        q = self.to_q(x)                                # [B, h*d, H, W]
        q = q.view(B, self.heads, self.dim_head, H*W)   # [B,h,d,HW]
        q = q.permute(0,1,3,2)                          # [B,h,HW,d]

        kv = self.to_kv_tokens(z)                       # [B, tokens*2*d]
        kv = kv.view(B, 2, -1, self.dim_head)           # [B, 2, T, d]
        k, v = kv[:,0], kv[:,1]                         # [B, T, d], [B, T, d]
        # expand heads
        k = k.unsqueeze(1).expand(B, self.heads, -1, -1)   # [B,h,T,d]
        v = v.unsqueeze(1).expand(B, self.heads, -1, -1)   # [B,h,T,d]

        attn = torch.matmul(q, k.transpose(-1,-2)) * self.scale   # [B,h,HW,T]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                     # [B,h,HW,d]
        out = out.permute(0,1,3,2).contiguous()         # [B,h,d,HW]
        out = out.view(B, self.heads*self.dim_head, H, W)
        out = self.proj(out)
        return x + self.gamma * out

def validate_model(
    encod, decod, recog, clf_model, subset_model,
    perception_loss_clf_func, clf_loss_func, subset_loss_func, losses,
    valid_loader, latent_dim, std_range,
    device
):
    encod.eval()
    decod.eval()
    recog.eval()
    subset_model.eval()
    clf_model.eval()

    val_recon = 0
    val_clf = 0
    val_disent = 0
    val_cov = 0
    val_vae_all = 0
    val_subset = 0

    z_std, _ = compute_latent_feature_variance(encod, valid_loader, device)

    with torch.no_grad():
        num_batches = len(valid_loader)
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            altered_indices = torch.randint(0, latent_dim, (imgs.size(0),), device=device) ##change different latent feature per each image
            z_img, mu, logvars = encod(imgs)
            range_alteration = torch.from_numpy(np.random.choice(np.linspace(-4, 4, 70), imgs.shape[0])) ###Alter latent feature in range of std_range 
            range_alteration = range_alteration.to(device)
            kl_loss = -0.5 * torch.mean(1 + logvars - mu.pow(2) - logvars.exp())
            z_alter = z_img.clone()
            for i in range(imgs.size(0)): ##change different latent feature per each image
                epsilon = range_alteration[i] * z_std[altered_indices[i]]
                z_alter[i, altered_indices[i]] = z_alter[i, altered_indices[i]] + epsilon ##change different latent feature per each image
            z_combined = torch.cat([z_img, z_alter], dim = 0)
            img_combined_recon = decod(z_combined)
            img_recon, alt_img_recon = torch.chunk(img_combined_recon, 2, dim=0)
            recon_loss = perception_loss_clf_func(img_recon, imgs)
            clf_loss = clf_loss_func(img_recon, imgs)
            z_var_loss = losses.covariance_loss(z_img)
            clf_target = torch.sigmoid(clf_model(imgs))  
            labels = labels.to(device)
            subset_loss = subset_loss_func(subset_model(z_img), clf_target)
            diff_img = (img_recon - alt_img_recon).abs()
            diff_img = 10.0 * diff_img  # scale up
            diff_img.to(device)
            logits = recog(diff_img[:, 0:1, :, :])
            disent_loss = F.cross_entropy(logits, altered_indices, label_smoothing= 0.05) ##change different latent feature per each image
            # print(diff_img.mean())
            total_vae_loss = 6 * recon_loss  + 6 * clf_loss + 2 * subset_loss + 15 * kl_loss + 3 * z_var_loss + 2 * disent_loss#+  #+ cf_loss # disent_loss

            val_clf += clf_loss.item()
            val_recon += recon_loss.item()
            val_cov += z_var_loss.item()
            val_disent += disent_loss.item()
            val_vae_all += total_vae_loss.item()
            val_subset += subset_loss.item()

    return {
        'recon': val_recon / num_batches,
        'clf': val_clf / num_batches,
        'disent': val_disent / num_batches,
        'cov': val_cov / num_batches,
        'vae_all': val_vae_all / num_batches,
        'subset': val_subset / num_batches
    }

def compute_latent_feature_variance(encoder, dataloader, device):
    encoder.eval()  # Set encoder to evaluation mode
    all_latents = []

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            z, _, _ = encoder(imgs)  # shape: [batch_size, latent_dim]
            all_latents.append(z.cpu())  # move to CPU to avoid GPU memory issues

    # Concatenate all latent vectors: shape [N_samples, latent_dim]
    all_latents = torch.cat(all_latents, dim=0)

    # Compute variance per feature (dim=0 is across samples)
    feature_variances = torch.std(all_latents, dim=0, unbiased=True)  # shape: [latent_dim]
    feature_means = torch.mean(all_latents, dim=0)

    return feature_variances, feature_means
