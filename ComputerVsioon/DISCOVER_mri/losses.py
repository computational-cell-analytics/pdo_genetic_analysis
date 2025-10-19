import torchvision
from torchvision.models import vgg19, VGG19_Weights, vgg16
import math
import numpy as np
import torch
from torch import nn
import utils
import torch.nn.functional as F
class perception_loss_clf(nn.Module):
    def __init__(self, model, device):
        super(perception_loss_clf, self).__init__()
        self.clf_model = model
        self.means = torch.tensor([0.485, 0.456, 0.406], device = device).view(1, 3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225], device= device).view(1, 3, 1, 1)
        self.activations = {}
        
        for idx in [0, 5, 10]:
            self.clf_model.features[idx].register_forward_hook(self._get_activation(str(idx)))
        

    def _get_activation(self, name):
            def hook(model, input, output):
                self.activations[name] = output
            return hook
    
    def forward(self, output, target):
        k_layers = 0
        self.activation = {}
        losses = []
        loss = 0
        # output = (output - self.means)/self.stds
        # target = (target - self.means)/self.stds
        mae = torch.nn.L1Loss()
        # def get_activation(name):
        #     def hook(model, input, output):
        #         activation[name] = output#.detach()
        #     return hook
        # for layer_idx in [0, 5, 10]:
        #     if self.clf_model.features[layer_idx].__class__.__name__ == 'Conv2d':
        #         k_layers += 1
        #         h = self.clf_model.features[layer_idx].register_forward_hook(get_activation(f'features_{layer_idx}'))
        #         pred_output = self.clf_model(output)
        #         pred_output = activation[f'features_{layer_idx}']
        #         target_output = self.clf_model(target)
        #         target_output = activation[f'features_{layer_idx}']
        #         #loss += torch.abs((target_output - pred_output)).sum()/(pred_output.shape[-1] * pred_output.shape[-2]* pred_output.shape[-3])
        #         loss += mae(target_output, pred_output)
        #         #losses.append(mae(target_output, pred_output).item())
        for name in self.activations:
            loss += mae(self.activations[name], ...)
        #         h.remove()
        return loss/k_layers

class PerceptionLossVGG(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Layer indices for conv1_1, conv2_1, conv3_1
        self.selected_layers = {
            '0': 1.0,  # conv1_1
            '5': 0.8,  # conv2_1
            '10': 0.2  # conv3_1
        }

        self.means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mse = torch.nn.L1Loss()

    def forward(self, output, target):
        #output = (output + 1)/2
        # target = target.repeat(1, 3, 1, 1)
        # print(target.shape)
        output = output.repeat(1, 3, 1, 1)
        output = (output - self.means) / self.stds
        target = (target - self.means) / self.stds

        loss = 0.0
        x = output
        y = target
        k = 0
        for name, module in self.vgg._modules.items():
            x = module(x)
            y = module(y)
            if name in self.selected_layers:
                loss += self.selected_layers[name] * self.mse(x, y)
                k += 1
        return loss/k



###classification oriented loss
class clf_orientedLoss(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        # vgg = vgg16(weights='IMAGENET1K_V1')
        # vgg.classifier[6] = nn.Linear(4096, 16, bias = True)
        # self.trained_clf = utils.clf_VGG(vgg)
        self.trained_clf = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", weights_only = False, map_location=device).model.eval()
        # self.clf
        for param in self.trained_clf.parameters():
            param.requires_grad = False

        # Layer indices for conv1_1, conv2_1, conv3_1
        self.selected_layers = {
            'features': {
                '10': 1.0,  # conv3_1
                '12': 1,  # conv3_2
                '14': 1,  # conv3_3
                '17': 1.0,  # conv4_1
                '19': 0.8,  # conv4_2
                '21': 0.2,  # conv4_3
                '24': 1.0,  # conv5_1
                '26': 0.8,  # conv5_2
                '28': 0.2  # conv5_3
            },
            'classifier': {
                '3': 1, # Dense(.., 4096)
                '6': 1 #Dense(4096,16)
            }
        }

        self.means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mse = torch.nn.L1Loss()

    def forward(self, output, target):
        #output = (output + 1)/2
        output = (output - self.means) / self.stds
        target = (target - self.means) / self.stds

        loss = 0.0
        x = output
        y = target
        k = 0
        total_modules = self.trained_clf.features._modules | self.trained_clf.classifier._modules
        for name, module in self.trained_clf.features._modules.items():
            x = module(x)
            y = module(y)
            if name in self.selected_layers['features'] or self.selected_layers['classifier']:
                loss += self.mse(x, y)
                k += 1
        return loss/k



# ========== Latent Regularization Losses ==========
def latent_z_mean_loss(z):
    return torch.mean(z)

def latent_z_variance_loss(z):
    # Ensure mean-centered latent vectors
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Covariance matrix: shape [latent_dim, latent_dim]
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)

    # Penalize off-diagonal elements
    off_diag = cov - torch.diag(torch.diag(cov))
    loss = (off_diag ** 2).sum()


    # z = torch.permute(z, (1, 0))
    # cov_z = torch.cov(z)
    # I = torch.eye(z.size(0), device=z.device)     # Identity matrix
    # loss = torch.norm(cov_z - I, p='fro') ** 2      # Frobenius norm
    return loss



def Discriminator_loss(real, fake):
    #normal_dist = np.random.normal(0, 1, (batch_size, latent_size))
    real_labels = torch.empty_like(real).uniform_(0.8, 1.0)
    fake_labels = torch.empty_like(fake).uniform_(0.0, 0.2)

    real_loss = F.binary_cross_entropy(real, real_labels)
    fake_loss = F.binary_cross_entropy(fake, fake_labels)

    disc_loss = real_loss + fake_loss

    return disc_loss

def adv_loss(model_disc, z_img):
    loss = torch.mean(torch.log(model_disc(z_img) + 1e-8))
    return -loss

class PixelwiseLoss(nn.Module):
    def __init__(self, mode='l2'):
        super().__init__()
        if mode == 'l1':
            self.loss_fn = nn.L1Loss()
        elif mode == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported mode. Choose 'l1' or 'l2'.")

    def forward(self, output, target):
        return self.loss_fn(output, target)

def counterfactual_consistency_loss(z_img, decoder, classifier, imgs, z_std, selected_indices, epsilon=1.5, device='cuda'):
    decoder.eval()
    classifier.eval()

    # with torch.no_grad():
    #     imgs = imgs.to(device)
    #     z = encoder(imgs)

    cf_loss = 0.0
    for idx in selected_indices:
        std_val = z_std[idx].item()

        # +ε perturbation
        z_pos = z_img.clone()
        z_pos[:, idx] += epsilon * std_val
        x_pos = decoder(z_pos)
        s_pos = torch.sigmoid(classifier(x_pos)).view(-1)

        # -ε perturbation
        z_neg = z_img.clone()
        z_neg[:, idx] -= epsilon * std_val
        x_neg = decoder(z_neg)
        s_neg = torch.sigmoid(classifier(x_neg)).view(-1)

        # ReLU(s_neg - s_pos): we expect s_pos > s_neg for monotonicity
        cf_loss += F.relu(s_neg - s_pos).mean()

    return cf_loss / len(selected_indices)

def covariance_loss(z):
    """
    z: [batch_size, latent_dim] - latent representations
    """
    batch_size, latent_dim = z.size()

    # Center the latent codes
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Compute covariance matrix: [latent_dim, latent_dim]
    cov = (z_centered.T @ z_centered) / (batch_size - 1)

    # Diagonal and off-diagonal elements
    diag = torch.diag(cov)
    off_diag = cov - torch.diag_embed(diag)

    # Loss = 0.5 * (||diag - 1||^2 + ||off_diag||^2)
    loss = 0.5 * (torch.mean((diag - 1) ** 2) + torch.mean(off_diag ** 2))

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return self.alpha * loss.mean()

loss_fn = FocalLoss()


import torch
import torch.nn.functional as F

def compute_kernel(x, y):
    """
    Compute the RBF kernel between two sets of samples.
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    # Gaussian (RBF) kernel
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / dim
    return torch.exp(-kernel_input)  # shape: [x_size, y_size]

def mmd_loss_function(z, z_prior):
    """
    MMD loss between encoded latent vectors `z` and prior samples `z_prior`.
    """
    z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
    xx = compute_kernel(z, z)         # shape: [B, B]
    yy = compute_kernel(z_prior, z_prior)
    xy = compute_kernel(z, z_prior)

    return xx.mean() + yy.mean() - 2 * xy.mean()

import torch
import torch.nn.functional as F


def compute_pairwise_distances(x, y):
    """
    Compute squared Euclidean distance between all pairs (x_i, y_j).
    x: [B, D]
    y: [B, D]
    returns: [B, B]
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.T)
    return dist


def adaptive_rbf_kernel(x, y, bandwidth_scale=1.0, eps=1e-8):
    """
    RBF kernel with adaptive bandwidth based on the median heuristic.
    """
    pairwise_dists = compute_pairwise_distances(x, y)
    
    # Estimate bandwidth using median heuristic
    if x is y:
        bandwidth = torch.median(pairwise_dists[pairwise_dists != 0])
    else:
        all_dists = torch.cat([compute_pairwise_distances(x, x).flatten(),
                               compute_pairwise_distances(y, y).flatten(),
                               pairwise_dists.flatten()], dim=0)
        bandwidth = torch.median(all_dists[all_dists != 0])
    
    bandwidth = bandwidth_scale * bandwidth.detach()
    kernel = torch.exp(-pairwise_dists / (2 * bandwidth + eps))
    return kernel


def mmd_loss_adaptive(z, prior=None, bandwidth_scale=1.0):
    """
    Computes MMD between encoded z and standard normal prior using adaptive kernel.
    z: [B, D] encoded latent
    prior: [B, D] optional (default: samples from N(0, 1))
    """
    z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
    B, D = z.shape
    if prior is None:
        prior = torch.randn_like(z)

    k_zz = adaptive_rbf_kernel(z, z, bandwidth_scale)
    k_pp = adaptive_rbf_kernel(prior, prior, bandwidth_scale)
    k_zp = adaptive_rbf_kernel(z, prior, bandwidth_scale)

    # MMD = E[k_zz] + E[k_pp] - 2E[k_zp]
    mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
    return mmd
