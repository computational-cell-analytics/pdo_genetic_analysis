###Define Resnetblock downsample
import torch
from torch import nn
from torchvision.transforms.v2 import GaussianNoise
import torch.nn.functional as F
std = 0.03
# class GaussianNoise(nn.Module):
#     """Gaussian noise regularizer.

#     Args:
#         sigma (float, optional): relative standard deviation used to generate the
#             noise. Relative means that it will be multiplied by the magnitude of
#             the value your are adding the noise to. This means that sigma can be
#             the same regardless of the scale of the vector.
#         is_relative_detach (bool, optional): whether to detach the variable before
#             computing the scale of the noise. If `False` then the scale of the noise
#             won't be seen as a constant but something to optimize: this will bias the
#             network to generate vectors with smaller values.
#     """

#     def __init__(self, sigma=0.1, is_relative_detach=True):
#         super().__init__()
#         self.sigma = sigma
#         self.is_relative_detach = is_relative_detach
#         self.noise = torch.tensor(0)
# #.to(device)
#     def forward(self, x):
#         if self.training and self.sigma != 0:
#             scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
#             sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
#             x = x + sampled_noise
#         return x


class resblock_down_sample(nn.Module):
    def __init__(self, in_feat, num_feat, std):
        super(resblock_down_sample, self).__init__()
        self.in_feat = in_feat
        self.num_feat = num_feat
        # self.num_feat2 = num_feat2
        # self.num_feat2 = num_feat2
        self.std = std
        self.down_sample = nn.Sequential(nn.BatchNorm2d(self.in_feat)
        ,nn.ReLU()
        ,nn.Conv2d(self.in_feat, self.num_feat, 3, 2, 1)
        # print(x.shape)
        ,GaussianNoise(self.std)
        ,nn.BatchNorm2d(self.num_feat)
        ,nn.ReLU()
        ,nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 'same')
        ,GaussianNoise(self.std))
        self.res = nn.Conv2d(self.in_feat, self.num_feat, 1, 2)
    def forward(self, x):
        inp = x
        x = self.down_sample(x)
        # x = nn.BatchNorm2d(self.in_feat)(x)
        # # print(x.shape)
        # x = nn.ReLU()(x)
        # x = nn.Conv2d(self.in_feat, self.num_feat, 3, 2, 1)(x)
        # # print(x.shape)
        # x = GaussianNoise(self.std)(x)
        # x = nn.BatchNorm2d(self.num_feat)(x)
        # x = nn.ReLU()(x)
        # x = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 'same')(x)
        # x = GaussianNoise(self.std)(x)
        d_res = self.res(inp)
        # print(d_res.shape)
        # print(x.shape)
        x = x + d_res
        return x
        # x = nn.BatchNorm1d(self.num_feat2)(x)
        # x = nn.ReLU()(x)
        # x = nn.Conv2D(self.num_feat2, self.num_feat3, 3, 2, 'same')(x)

class resblock_up_sample(nn.Module):
    def __init__(self, in_feat, num_feat, std):
        super(resblock_up_sample, self).__init__()
        self.in_feat = in_feat
        self.num_feat = num_feat
        self.std = std
        self.up_block = nn.Sequential(nn.InstanceNorm2d(self.in_feat),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(self.in_feat, self.num_feat, 3, 1, 'same'),
        GaussianNoise(self.std),
        nn.InstanceNorm2d(self.num_feat),
        nn.ReLU(),
        nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 'same'),
        GaussianNoise(self.std))
        self.res = nn.Sequential(nn.Upsample(scale_factor=2),
        nn.Conv2d(self.in_feat, self.num_feat, 3, 1, 'same'))
        # self.num_feat2 = num_feat2
        # self.num_feat3 = num_feat3
        
    def forward(self, x):
        inp = x
        x = self.up_block(x)
        d_res = self.res(inp)
        x = x + d_res
        return x


###build encoder
class Encoder(nn.Module):
    def __init__(self, inp_ch, std):
        super(Encoder, self).__init__()
        self.std = std
        self.input = nn.Conv2d(inp_ch, 64, 3, 1, 'same')
        self.block1 = resblock_down_sample(64, 512, std)
        self.block2 = resblock_down_sample(512, 1024, std)
        self.block3 = resblock_down_sample(1024, 1024, std)
        self.block4 = resblock_down_sample(1024, 1024, std)
        self.denses = nn.Sequential(nn.Flatten(), GaussianNoise(self.std), nn.SiLU(),
                                     nn.Linear(16384, 350), GaussianNoise(self.std), nn.BatchNorm1d(350))
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
        # x = nn.Flatten()(x)
        # x = GaussianNoise(self.std)(x)
        # x = nn.SiLU()(x)
        # print('silu')
        # print(x.shape)
        # x = nn.Linear(16384, 350)(x)
        # x = GaussianNoise(self.std)(x)
        # x = nn.BatchNorm1d(350)(x)
        # #intermediate = z[:, :14]
        # #clf_prob = nn.Linear(14, 1)(intermediate)
        # #clf_prob = nn.Sigmoid()(clf_prob)
        return(x)

#self.to_img = nn.Sequential(nn.Conv2d(512, 3, 3, 1, 'same'), nn.BatchNorm2d(3), nn.Sigmoid())

class Decoder(nn.Module):
    def __init__(self, latent_shape, std):
        super(Decoder, self).__init__()
        self.std = std
        self.latent = latent_shape
        self.input = nn.Linear(self.latent, 16384)
        self.block1 = resblock_up_sample(1024, 1024, std)
        self.block2 = resblock_up_sample(1024, 1024, std)
        self.block3 = resblock_up_sample(1024, 512, std)
        self.block4 = resblock_up_sample(512, 512, std)
        self.to_img = nn.Sequential(nn.Conv2d(512, 3, 3, 1, 'same'), nn.Sigmoid())
    def forward(self, x):
        x =  self.input(x)
        # print(x.shape)
        x = GaussianNoise(self.std)(x)
        # x = nn.Flatten()(x)
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
        # x = nn.Conv2d(512, 3, 3, 1, 'same')(x)
        # print('conv', x.shape)
        # x = nn.BatchNorm2d(1)(x)
        # print(x.shape)
        # x = nn.ReLU()(x)
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
        # x = nn.Linear(self.inp_size, 2048)(x) ##layer1
        # x = nn.BatchNorm1d(2048)(x)
        # x = nn.LeakyReLU(0.2)(x)
        # x = nn.Linear(2048, 2048)(x) ##layer2
        # x = nn.BatchNorm1d(2048)(x)
        # x = nn.LeakyReLU(0.2)(x)
        # x = nn.Linear(2048, 2048)(x) ##layer3
        # x = nn.BatchNorm1d(2048)(x)
        # x = nn.LeakyReLU(0.2)(x)
        # x = nn.Linear(2048, 2048)(x) ##layer4
        # x = nn.BatchNorm1d(2048)(x)
        # x = nn.LeakyReLU(0.2)(x)
        # x = nn.Linear(2048, 2048)(x) ##layer5
        # x = nn.LeakyReLU(0.2)(x)
        # x = nn.Linear(2048, 2048)(x) ##layer6
        # x = nn.Dropout(0.5)(x)
        # x = nn.Linear(2048, 1)(x)
        # x = nn.Sigmoid()(x)
        return x

class Recognizer(nn.Module):
    def __init__(self, ):
        super().__init__()


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
    def __init__(self, in_channels=3, num_classes=350):
        super().__init__()
        self.initial = nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        self.res1 = resblock_down_sample(64, 512, std)
        self.res2 = resblock_down_sample(512, 1024, std)
        self.res3 = resblock_down_sample(1024, 1024, std)
        self.res4 = resblock_down_sample(1024, 1024, std)
        self.flatten = nn.Flatten()
        self.swish = nn.SiLU()
        self.fc = nn.Linear(1024 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.initial(x)  # (64,64,64)
        # x = F.Relu(x) ###Commented
        x = self.res1(x)             # (512,32,32)
        x = self.res2(x)             # (512,16,16)
        x = self.res3(x)             # (1024,8,8)
        x = self.res4(x)             # (1024,4,4)
        x = self.flatten(x)          # (16,384)
        x = self.swish(x)
        x = self.fc(x)               # (350,)
        return x #F.softmax(x, dim=1)

class simple_neuron(nn.Module):
    def __init__(self, sub_features = 14):
        super().__init__()
        self.sub_features = sub_features
        self.simple_clf = nn.Linear(self.sub_features, 1)

    def forward(self, x):
        x = self.simple_clf(x[:, :self.sub_features])  # (64,64,64)
        return x
