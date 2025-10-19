from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torch import nn
from torchvision.models import vgg16, VGG16_Weights, efficientnet_b1, EfficientNet_B1_Weights
import torch


#resnet = resnet18(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet = resnet50()
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = resnet.fc.in_features
out_ftrs = resnet.fc.out_features
resnet.fc = nn.Linear(num_ftrs, 512)
# for name, layer in resnet.named_children():
#     if name != 'fc':
#         for param in layer.parameters():
#             param.requires_grad = False
#     for param in layer.parameters():
#         if param.requires_grad:
#             print(param)
#             print(name)
            
class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        self.model = model
        self.clf = nn.ModuleList([nn.ReLU(), nn.Linear(512, 128), nn.Dropout(0.4),
                                  nn.ReLU(), nn.Linear(128, 32),nn.Dropout(0.3),
                                  nn.ReLU(), nn.Linear(32, 1)])
        #self.sig = 

    def forward(self, x):
        x = self.model(x)
        for i in self.clf:
            x = i(x)
        # x = torch.nn.Sigmoid(x)
        return x
        # return x

model = ResNet(resnet)



# vgg = vgg16(VGG16_Weights.IMAGENET1K_V1)
vgg = vgg16()
vgg.classifier[6] = nn.Linear(4096, 16, bias = True)
vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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
model_vgg19 = VGG(vgg)

model_eff_b = efficientnet_b1(weights= EfficientNet_B1_Weights.IMAGENET1K_V2)
model_eff_b.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
model_eff_b.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
