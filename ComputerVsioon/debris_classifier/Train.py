import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from cellpose import models
import torch.nn as nn
import torch.nn.functional as F
from dataset import normal_ds
from dataset import CellImageEncoder, ClassifierHead, enc_cls

class CellPoseClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(CellPoseClassifier, self).__init__()
        self.encoder = encoder
        self.freeze_encoder_layers()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def freeze_encoder_layers(self):
        # Freeze encoder blocks (not neck)
        for name, param in self.encoder.named_parameters():
            if 'blocks' in name or 'patch_embed' in name:
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)  # Output from encoder.neck: [B, 256, 8, 8]
        pooled = self.global_pool(features)  # [B, 256, 1, 1]
        out = self.classifier(pooled)
        return out



def train_classifier_model(
    model, train_dataset, val_dataset=None,
    epochs=20, batch_size=32, lr=1e-4,
    weight_decay=1e-5, device='cuda'
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) if val_dataset else None

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_score = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs.repeat(1, 3, 1, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f} | Train Acc = {100.*correct/total:.2f}%")

        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs.repeat(1, 3, 1, 1))
                    _, preds = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += preds.eq(labels).sum().item()
            print(f"Validation Accuracy: {100. * val_correct / val_total:.2f}%")
        if val_correct > best_val_score:
            best_model = model
            best_val_score = val_correct
    return best_model

# train_dataset = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/debris_classifier/hdbscan_40_1.pt", weights_only=False)
train_dataset = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/debris_classifier/hdbscan_40_2_no_aug.pt", weights_only=False)

model = models.CellposeModel(gpu=True)
model = model.net
cellpose_encoder = model.encoder
model_cellpose_clf = CellPoseClassifier(encoder=cellpose_encoder, num_classes=2)
model_cellpose_clf = model_cellpose_clf.to(torch.float32)
original_pos_embed = model_cellpose_clf.encoder.pos_embed
# Resize positional embeddings to [1, 8, 8, 1024]
pos_embed_resized = torch.nn.functional.interpolate(
    original_pos_embed.permute(0, 3, 1, 2),  # [1, C, H, W]
    size=(8, 8),
    mode='bilinear',
    align_corners=False
).permute(0, 2, 3, 1)  # back to [1, H, W, C]

# Replace in model
model_cellpose_clf.encoder.pos_embed = torch.nn.Parameter(pos_embed_resized)


strong_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.7, alpha=250, sigma=250 * 0.05),
        A.GaussNoise(p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.OneOf([
            A.MedianBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
        ], p=0.5),
    ])

enc = CellImageEncoder()
clf_he = ClassifierHead(128, 2)
custom_model = enc_cls(enc, clf_he)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

model_trained = train_classifier_model(
    model_cellpose_clf, train_dataset, val_dataset=val_dataset,
    epochs=100, batch_size=32, lr=1e-4,
    weight_decay=1e-5, device='cuda'
)
torch.save(model_trained, "/user/sina.garazhian/u12203/lustere-grete-mine/debris_classifier/cellpose_classifier_40_2_no_aug_custom_enc.pt")
