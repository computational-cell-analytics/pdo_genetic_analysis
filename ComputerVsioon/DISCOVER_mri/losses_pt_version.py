import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

# ========== Perceptual Loss using VGG19 ==========
class PerceptionLossVGG(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:11].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.selected_layers = {
            '0': 1.0,   # conv1_1
            '5': 0.8,   # conv2_1
            '10': 0.5   # conv3_1
        }

        self.means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output = (output - self.means) / self.stds
        target = (target - self.means) / self.stds
        loss = 0.0
        x = output
        y = target
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            y = layer(y)
            if name in self.selected_layers:
                loss += self.selected_layers[name] * self.mse(x, y)
        return loss


# ========== Pixel-wise Loss ==========
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


# ========== Latent Regularization Losses ==========
def latent_z_mean_loss(z):
    return torch.mean(z)

def latent_z_variance_loss(z):
    return torch.mean((z - torch.mean(z))**2)


# ========== Adversarial Losses ==========
def discriminator_loss(disc, z_fake, z_real):
    pred_fake = disc(z_fake)
    pred_real = disc(z_real)
    return -torch.mean(torch.log(pred_real + 1e-8) + torch.log(1 - pred_fake + 1e-8))

def generator_loss(disc, z):
    pred = disc(z)
    return -torch.mean(torch.log(pred + 1e-8))


# ========== Classification Loss ==========
def classifier_loss(logits, labels):
    return F.cross_entropy(logits, labels)


# ========== Hybrid Loss ==========
def hybrid_reconstruction_loss(percep_fn, pixel_fn, recon, target, alpha=0.5):
    return alpha * percep_fn(recon, target) + (1 - alpha) * pixel_fn(recon, target)
