###Define Resnetblock downsample
import torch
from torch import nn
from torchvision.transforms.v2 import GaussianNoise
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) using N(0, 1).

    Args:
        mu (Tensor): Mean of the latent Gaussian [B, latent_dim]
        logvar (Tensor): Log-variance of the latent Gaussian [B, latent_dim]

    Returns:
        z (Tensor): Sampled latent vector [B, latent_dim]
    """
    std = torch.exp(0.5 * logvar)  # standard deviation
    eps = torch.randn_like(std)    # random normal noise
    return mu + eps * std

import torch.nn.functional as F
import losses
def loss_fn(recon, x, mu_top, logvar_top, mu_bottom, logvar_bottom):
    """
    recon: reconstructed image (B, 1, H, W)
    x: original input image (B, 1, H, W)
    mu_top, logvar_top: top-level latent distribution
    mu_bottom, logvar_bottom: bottom-level latent distribution
    recon_type: 'mse' or 'bce'
    beta: scaling factor for KL divergence
    """
    # --- Reconstruction loss ---
    perception_loss_clf_func = losses.PerceptionLossVGG('cuda')
    recon_loss = perception_loss_clf_func(recon, x) 
    total_recon_loss =  recon_loss + F.mse_loss(recon, x)
    # --- KL Divergence for each latent level ---
    kl_top = -0.5 * torch.mean(1 + logvar_top - mu_top.pow(2) - logvar_top.exp())
    kl_bottom = -0.5 * torch.mean(1 + logvar_bottom - mu_bottom.pow(2) - logvar_bottom.exp())

    total_kl = kl_top + kl_bottom

    # --- Total loss ---
    loss = total_recon_loss + total_kl

    return loss, recon_loss, total_kl, kl_top, kl_bottom


        

class CellImageDecoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CellImageDecoder, self).__init__()

        # Suggestion: 128 → 128 → 64 → 32
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 8 * 8),
            nn.ReLU()
        )

        # New upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 → 16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 → 32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 → 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Final conv → 1 channel output
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # [B, 32, 64, 64] → [B, 1, 64, 64]
            nn.Sigmoid()  # for grayscale pixel values in [0, 1]
        )

    def forward(self, z):
        x = self.projector(z)  # [B, 64*8*8]
        x = x.view(-1, 128, 8, 8)  # reshape to [B, 64, 8, 8]
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.output_layer(x)
        return x



class CellImageEncoder_represent(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CellImageEncoder_represent, self).__init__()

        # --- Initial Low-level Texture + Edge Detectors ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 64, 64] -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- Multi-Scale (Inception-like) Block ---
        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=1, padding=0),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=5, padding=2),   # [B, 32, 64, 64]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),   # fuse channels → [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- Residual Block with Attention ---
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, 64, 1, 1]
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Global Pooling and Projection to Embedding ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)   # → [B, 64, 1, 1]
        self.projector = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)  # low-level
        multi_scale_feats = [F.relu(conv(x)) for conv in self.conv_multi_scale]
        x = torch.cat(multi_scale_feats, dim=1)
        x = self.fusion(x)

        # Residual attention block
        residual = self.res_block(x)
        attn = self.attention(x)
        x = x + residual * attn

        # Global pooling and embedding
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 64]
        x = self.projector(x)  # [B, embedding_dim]
        x = F.normalize(x, dim=1)  # normalize for contrastive learning
        return x



class Discriminator(nn.Module):
    def __init__(self, inp_size):
        super(Discriminator, self).__init__()
        self.inp_size = inp_size
        self.denses = nn.Sequential(nn.Linear(inp_size, 256),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5), ##just added
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            
            nn.utils.spectral_norm(nn.Linear(256, 1))
            )
    def forward(self, x):
        x = self.denses(x) 
        return torch.clamp(x, min=-4, max=4)



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
    def __init__(self, in_channels=1, num_classes=400):
        super().__init__()
        std = 0.1
        self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = resblock_down_sample(64, 128, std)
        self.res2 = resblock_down_sample(128, 256, std)
        self.res3 = resblock_down_sample(256, 512, std)
        self.res4 = resblock_down_sample(512, 1024, std)
        self.flatten = nn.Flatten()
        self.swish = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),
            nn.SiLU(),
            nn.Linear(1024, num_classes)
            )

    def forward(self, x):
        x = self.initial(x)  # (64,64,64)
        x = self.res1(x)             # (512,32,32)
        x = self.res2(x)             # (512,16,16)
        x = self.res3(x)             # (1024,8,8)
        x = self.res4(x)             # (1024,4,4)
        x = self.flatten(x)          # (16,384)
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
    def __init__(self, sub_features = 14, patient_numbers = 9):
        super().__init__()
        self.sub_features = sub_features
        self.simple_clf = nn.Linear(self.sub_features, patient_numbers)

    def forward(self, x):
        x = self.simple_clf(x[:, :self.sub_features])  # (64,64,64)
        return x

class device_network(nn.Module):
    def __init__(self, input_idx = 30 ,neurons = 20, num_inputs = 1):
        super().__init__()
        self.input_feat = input_idx
        self.sub_features = neurons
        self.num_inputs = num_inputs
        self.simple_clf = nn.Sequential(nn.Linear(num_inputs, self.sub_features),
                                        nn.ReLU(),
                                        nn.Linear(self.sub_features, 2))

    def forward(self, x):
        x = self.simple_clf(x[:, self.input_feat:self.input_feat + self.num_inputs])  # (64,64,64)
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

class CellImageEncoder(nn.Module):
    def __init__(self, embedding_dim=300):
        super(CellImageEncoder, self).__init__()

        # --- Initial Low-level Texture + Edge Detectors ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 64, 64] -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- Multi-Scale (Inception-like) Block ---
        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, kernel_size=1, padding=0),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=3, padding=1),   # [B, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 32, 7, padding=3)   # [B, 32, 64, 64]
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # fuse channels → [B, 64, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # --- Residual Block with Attention ---
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B, 64, 1, 1]
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # → [B, 256, 32, 32]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # → [B, 512, 16, 16]
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # → [B, 512, 8, 8]
            nn.ReLU(),
        )

        # --- Global Pooling and Projection to Embedding ---
        # self.latent = nn.Sequential(nn.Flatten(),
        #                             nn.Linear(128*64*64, 32768), nn.ReLU(), nn.BatchNorm1d(),
        #                             nn.Linear(32768, 2048), nn.BatchNorm1d(), nn.ReLU(),
        #                             nn.Linear(2048, 300), nn.BatchNorm1d(), nn.ReLU())# → [B, 64, 1, 1]
        # # self.projector = nn.Linear(300, embedding_dim)
        # --- Mean and logvar heads ---
        self.fc_mu = nn.Linear(32768, embedding_dim)
        self.fc_logvar = nn.Linear(32768, embedding_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.conv1(x)  # low-level
        multi_scale_feats = [F.relu(conv(x)) for conv in self.conv_multi_scale]
        x = torch.cat(multi_scale_feats, dim=1)
        x = self.fusion(x)

        # Residual attention block
        residual = self.res_block(x)
        attn = self.attention(x)
        x = x + residual * attn
        x = self.bottleneck(x)
        # Global pooling and embedding
        # x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 64]
        x = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(x)         # [B, embedding_dim]
        logvar = self.fc_logvar(x) # [B, embedding_dim]
        z = self.reparameterize(mu, logvar)  # [B, embedding_dim]
        # x = F.normalize(x, dim=1)  # normalize for contrastive learning
        return z, mu, logvar

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

class Patient_classifier(nn.Module):
    def __init__(self, encoder_path, classfier_path, device):
        super(Patient_classifier, self ).__init__()
        self.encoder = CellImageEncoder_represent()
        self.encoder.load_state_dict(torch.load(encoder_path, map_location = torch.device(device)))
        self.clf = torch.load(classfier_path, weights_only = False, map_location = torch.device(device))
    def forward (self, x):
        x  = self.encoder(x)
        logit = self.clf(x)
        return logit

class CellImageEncoderWithSkips(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, 1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.Conv2d(32, 32, 7, padding=3)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Linear(128, embedding_dim)

    def forward(self, x):
        skip0 = self.conv1(x)
        multi = [F.relu(conv(skip0)) for conv in self.conv_multi_scale]
        x = torch.cat(multi, dim=1)
        skip1 = self.fusion(x) #128*64*64

        residual = self.res_block(skip1) #128*64*64
        x = skip1 + residual * self.attention(residual)

        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        z = self.projector(pooled)
        z = F.normalize(z, dim=1)
        return z, [skip0, skip1]

class CellImageDecoderWithSkips(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 8 * 8),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 → 16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 → 32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 → 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, skips):
        x = self.projector(z).view(-1, 128, 8, 8)
        x = self.up1(x)
        x = x + F.interpolate(skips[1], size=x.shape[-2:], mode='nearest')  # skip1: [B, 64, 64, 64]

        x = self.up2(x)
        

        x = self.up3(x)
        x = x +  F.interpolate(skips[0], size=x.shape[-2:], mode='nearest')  # skip0: [B, 32, 64, 64]
        return self.output_layer(x)

####vq_vae_2 model

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = flat_x.view(-1, self.embedding_dim)

        # Compute distances and find nearest embeddings
        distances = (flat_x ** 2).sum(dim=1, keepdim=True) - 2 * flat_x @ self.embeddings.weight.t() + (self.embeddings.weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices.view(x.shape[0], x.shape[2], x.shape[3])

class HierarchicalVQVAEEncoder(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_multi_scale = nn.ModuleList([
            nn.Conv2d(32, 32, 1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 5, padding=2),
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Downsample: 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.quantizer_top = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.quantizer_bottom = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.bottom_conv = nn.Conv2d(128, embedding_dim, 1)
        self.top_conv = nn.Conv2d(128, embedding_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        multi = [F.relu(conv(x)) for conv in self.conv_multi_scale]
        x = torch.cat(multi, dim=1)
        x = self.fusion(x)
        x = self.res_block(x)  # [B, 128, 16, 16]

        z_top = self.top_conv(x)  # [B, embed_dim, 16, 16]
        z_top_q, loss_top, idx_top = self.quantizer_top(z_top)

        z_bottom = self.bottom_conv(x)  # [B, embed_dim, 16, 16]
        z_bottom_q, loss_bottom, idx_bottom = self.quantizer_bottom(z_bottom)

        return {
            "z_top_q": z_top_q,
            "z_bottom_q": z_bottom_q,
            "loss": loss_top + loss_bottom,
            "indices": {"top": idx_top, "bottom": idx_bottom}
        }



####the main discower paper encoder architucture
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
        self.block1 = resblock_down_sample(64, 512, std) #1024, 32, 32
        self.block2 = resblock_down_sample(512, 1024, std) #1024, 16, 16
        self.block3 = resblock_down_sample(1024, 1024, std) #1024, 8, 8
        self.block4 = resblock_down_sample(1024, 1024, std) #1024, 4, 4
        self.denses = nn.Sequential(nn.Flatten(), GaussianNoise(self.std), nn.Linear(16384, self.latent_size), nn.SiLU(),
                                      nn.Dropout(0.1), GaussianNoise(self.std), nn.BatchNorm1d(self.latent_size))
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
        # print('b4',x.shape)
        x = self.denses(x)
        return(x)

#self.to_img = nn.Sequential(nn.Conv2d(512, 3, 3, 1, 'same'), nn.BatchNorm2d(3), nn.Sigmoid())

class Decoder(nn.Module):
    def __init__(self, latent_shape, output_channel, std):
        super(Decoder, self).__init__()
        self.std = std
        self.latent = latent_shape
        self.input = nn.Sequential(nn.Linear(self.latent, 16384)
        ,GaussianNoise(self.std), nn.ReLU())
        self.block1 = resblock_up_sample(1024, 1024, std)
        self.block2 = resblock_up_sample(1024, 1024, std * 0.8)
        self.block3 = resblock_up_sample(1024, 512, std * 0.6)
        self.block4 = resblock_up_sample(512, 512, std * 0.4)
        self.to_img = nn.Sequential(nn.Conv2d(512, output_channel, 3, 1, 'same'), nn.Sigmoid())
    def forward(self, x):
        # x = F.layer_norm(x, x.shape[1:])
        x =  self.input(x)
        # print(x.shape)
        x = x.view(-1, 1024, 4, 4)
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


# Hierarchical VAE Encoder with probabilistic latent variables
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalVAEEncoder(nn.Module):
    def __init__(self, inp_ch=1, std=0.1, latent_dim_top=128, latent_dim_bottom=128):
        super().__init__()
        self.std = std

        self.input = nn.Conv2d(inp_ch, 64, 3, padding=1)
        self.block1 = resblock_down_sample(64, 512, std)   # → 32x32
        self.block2 = resblock_down_sample(512, 1024, std) # → 16x16
        self.block3 = resblock_down_sample(1024, 1024, std) # → 8x8
        self.block4 = resblock_down_sample(1024, 1024, std) # → 4x4

        # Bottom latent (from 8x8 features)
        self.bottom_mu = nn.Conv2d(1024, latent_dim_bottom, 1)
        self.bottom_logvar = nn.Conv2d(1024, latent_dim_bottom, 1)

        # Top latent (from 4x4 features)
        self.top_mu = nn.Conv2d(1024, latent_dim_top, 1)
        self.top_logvar = nn.Conv2d(1024, latent_dim_top, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.block1(x)
        x = self.block2(x)

        feat_8x8 = self.block3(x)
        feat_4x4 = self.block4(feat_8x8)

        # Bottom latent
        mu_bottom = F.adaptive_avg_pool2d(self.bottom_mu(feat_8x8), 1).squeeze(-1).squeeze(-1)
        logvar_bottom = F.adaptive_avg_pool2d(self.bottom_logvar(feat_8x8), 1).squeeze(-1).squeeze(-1)

        # Top latent
        mu_top = F.adaptive_avg_pool2d(self.top_mu(feat_4x4), 1).squeeze(-1).squeeze(-1)
        logvar_top = F.adaptive_avg_pool2d(self.top_logvar(feat_4x4), 1).squeeze(-1).squeeze(-1)

        return {
            'mu_top': mu_top,
            'logvar_top': logvar_top,
            'mu_bottom': mu_bottom,
            'logvar_bottom': logvar_bottom
        }


class HierarchicalVAEDecoder(nn.Module):
    def __init__(self, z_top_dim=128, z_bottom_dim=128, output_ch=1):
        super().__init__()
        self.z_top_dim = z_top_dim
        self.z_bottom_dim = z_bottom_dim

        # Project top latent to coarse 4x4 feature map
        self.top_fc = nn.Sequential(
            nn.Linear(z_top_dim, 1024 * 4 * 4),
            nn.ReLU()
        )

        # Project bottom latent to finer 8x8 map, conditioned later
        self.bottom_fc = nn.Sequential(
            nn.Linear(z_bottom_dim, 1024 * 8 * 8),
            nn.ReLU()
        )

        # Upsample top
        self.block_top1 = resblock_up_sample(1024, 1024)
        self.block_top2 = resblock_up_sample(1024, 1024)  # Reaches 16x16

        # Combine top and bottom
        self.combine = nn.Conv2d(1024 + 1024, 1024, 1)  # fuse top + bottom

        # Final upsampling
        self.block3 = resblock_up_sample(1024, 512)  # → 32x32
        self.block4 = resblock_up_sample(512, 256)   # → 64x64

        self.out_conv = nn.Sequential(
            nn.Conv2d(256, output_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_top, z_bottom):
        # Decode top latent to 4x4
        x_top = self.top_fc(z_top).view(-1, 1024, 4, 4)
        x_top = self.block_top1(x_top)  # 8x8
        x_top = self.block_top2(x_top)  # 16x16

        # Decode bottom latent to 8x8 and upsample to 16x16
        x_bottom = self.bottom_fc(z_bottom).view(-1, 1024, 8, 8)
        x_bottom = F.interpolate(x_bottom, scale_factor=2)  # → 16x16

        # Concatenate and fuse
        x = torch.cat([x_top, x_bottom], dim=1)
        x = self.combine(x)  # → [B, 1024, 16, 16]

        x = self.block3(x)  # → 32x32
        x = self.block4(x)  # → 64x64
        x = self.out_conv(x)
        return x


import cv2
import numpy as np
from skimage.measure import shannon_entropy


# Main function: apply CLAHE to a batch of grayscale images if needed
def apply_clahe_batch(batch_images):
    """
    Args:
        batch_images: list of grayscale images in [0, 1] float32 format (H x W)
        threshold: fuzzy threshold for deciding CLAHE usefulness

    Returns:
        List of processed images (same shape, same [0,1] float range)
    """
    processed = []
    # Create CLAHE transformer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    output_images = []
    batch_images = batch_images.detach().cpu().numpy()
    for img in batch_images:
        img = img.squeeze(0)
        # Convert [0, 1] float32 → [0, 255] uint8
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        if shannon_entropy(img_uint8) <= 5.90:
        # Apply CLAHE
            clahe_img = clahe.apply(img_uint8)
            processed.append(torch.tensor(clahe_img.astype(np.float32) / 255.0).unsqueeze(0))
        else:
            processed.append(torch.tensor(img.astype(np.float32)).unsqueeze(0))
        # processed.append(torch.tensor(output_images).unsqueeze(0))  # add channel dim

    return torch.stack(processed)

