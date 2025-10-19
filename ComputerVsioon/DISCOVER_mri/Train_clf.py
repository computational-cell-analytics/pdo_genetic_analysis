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
# import visulize


###getting data
# train_paths = glob("/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/**/*.jpg", recursive = True)
train_no_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/train/NonDemented/*.jpg")
train_very_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/train/VeryMildDemented/*.jpg") + glob("/user/sina.garazhian/u12203/kaggle_alz/train/MildDemented/*.jpg")
train_paths = np.array(train_no_paths + train_very_paths)
test_no_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/test/NonDemented/*.jpg")
test_very_paths = glob("/user/sina.garazhian/u12203/lustere-grete-mine/kaggle_alz/test/VeryMildDemented/*.jpg") + glob("/user/sina.garazhian/u12203/kaggle_alz/test/MildDemented/*.jpg")
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
val_img_normal_dataset, test_img_normal_dataset = torch.utils.data.random_split(test_img_normal_dataset, [0.5, 0.5])
batch_size = 128
train_loader = DataLoader(train_img_normal_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_img_normal_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_img_normal_dataset, batch_size = batch_size, shuffle = True)

###Cretae models
clf_model = torch.load("/user/sina.garazhian/u12203/DISCOWER/best_vgg.pt", weights_only = False, map_location=device)

##training
from copy import deepcopy
epochs = 200

#best_model = deepcopy(model)
def run_training(model, epochs, optimizer, criterion, train_loader, val_loader, device):
    best_acc = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    patience = 0
    for epoch in range(epochs):
        model.train()
        diff = 0
        acc = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(imgs.shape)
            out = model(imgs)
            # print(out.shape)
            out = torch.squeeze(model(imgs))
            loss = criterion(out, labels.float())
            diff += loss.item()
            acc += ((out >= 0) == labels.float()).sum().item()
            total += out.size(0)
    
            loss.backward()
            optimizer.step()
        train_loss += [diff/total]
        train_acc += [acc/total]
        model.eval()
        diff = 0
        acc = 0
        total = 0
        for imgs, labels in val_loader:
            with torch.no_grad():
                imgs, labels = imgs.to(device), labels.to(device)
                #optimizer.zero_grad()
        
                out = torch.squeeze(model(imgs))
                loss = criterion(out, labels.float())
                diff += loss.item()
                
                acc += ((out >= 0) == labels.float()).sum().item()
                total += out.size(0)
                del imgs, labels
                torch.cuda.empty_cache()
        val_loss += [diff/total]
        val_acc += [acc/total]
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            best_model = deepcopy(model)
        if val_acc[-1] <= best_acc:
            patience += 1
            # scheduler.step()
        if epoch >= 10 and patience >= 4:
            break
    
        
    
        print("Epoch {} train loss {} acc {} val loss {} acc {}".format(epoch, train_loss[-1], train_acc[-1],
                                                                       val_loss[-1], val_acc[-1]))
    return(train_loss, train_acc, val_loss, val_acc, best_model )

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(clf_model.parameters(), lr=0.00001)
train_losss, train_accs, val_losss, val_accs, best_model = run_training(clf_model, epochs, optimizer, criterion, train_loader, val_loader, device)

##testing the model
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

all_labels = []
all_probs = []

torch.save(best_model, "/user/sina.garazhian/u12203/DISCOWER/best_vgg_1.pt")
with torch.no_grad():
    CM = 0
    best_model.eval()
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = torch.squeeze(best_model(imgs))
        preds_bool = preds >= 0
        labels = labels.float()
        CM += confusion_matrix(labels.cpu(), preds_bool.cpu(),labels=[0,1])
        probs = torch.sigmoid(preds).squeeze()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("/user/sina.garazhian/u12203/DISCOWER/best_added_clf.png")
tn=CM[0][0]
tp=CM[1][1]
fp=CM[0][1]
fn=CM[1][0]
acc=np.sum(np.diag(CM)/np.sum(CM))
sensitivity=tp/(tp+fn)
precision=tp/(tp+fp)
        
print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
print()
print('Confusion Matirx : ')
print(CM)
print('- Sensitivity : ',(tp/(tp+fn))*100)
print('- Specificity : ',(tn/(tn+fp))*100)
print('- Precision: ',(tp/(tp+fp))*100)
print('- NPV: ',(tn/(tn+fn))*100)
print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
