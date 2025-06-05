import torchvision
from torchvision.models import vgg19, VGG19_Weights
import math
import numpy as np
import torch
from torch import nn

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
        output = (output - self.means)/self.stds
        target = (target - self.means)/self.stds
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
        output = (output + 1)/2
        output = (output - self.means) / self.stds
        target = (target - self.means) / self.stds

        loss = 0.0
        x = output
        y = target
        for name, module in self.vgg._modules.items():
            x = module(x)
            y = module(y)
            if name in self.selected_layers:
                loss += self.selected_layers[name] * self.mse(x, y)
        return loss



# def perception_loss_clf(input, target, clf_model):
#     activation = {}
#     model = clf_model
#     loss = 0
#     mae = torch.nn.L1Loss()
#     def get_activation(name):
#         def hook(model, input, output):
#             activation[name] = output.detach()
#         return hook
#     for layer_idx in range(3, 30):
#         if clf_model.features[layer_idx].__class__.__name__ == 'Conv2d':
#             h = clf_model.features[layer_idx].register_forward_hook(get_activation(f'features_{layer_idx}'))
#             pred_output = model(input)
#             pred_output = activation[f'features_{layer_idx}']
#             target_output = model(target)
#             target_output = activation[f'features_{layer_idx}']
#             #loss += torch.abs((target_output - pred_output)).sum()/(pred_output.shape[-1] * pred_output.shape[-2]* pred_output.shape[-3])
#             loss += mae(target_output, pred_output)
#             h.remove()
#     return loss

# class Discriminator_loss(nn.Module):
#     def __init__(self, model_disc, )

def Discriminator_loss(model_disc, z_img, batch_size, latent_size, device):
    #normal_dist = np.random.normal(0, 1, (batch_size, latent_size))
    normal_dist = torch.randn( (batch_size, latent_size))
    normal_dist = normal_dist.to(device)
    loss = torch.mean((torch.log(model_disc(normal_dist) + 1e-8) + torch.log(1 - model_disc(z_img) + 1e-8)))
    return -loss

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
