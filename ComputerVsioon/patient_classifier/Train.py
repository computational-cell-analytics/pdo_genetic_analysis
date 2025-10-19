import torch.optim as optim
from sklearn.metrics import accuracy_score
import utils
import torch
import dataset
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    # A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])



def apply_albumentations(batch_images, transform):
    augmented = []
    for img in batch_images:
        img_np = img.cpu().numpy()
        if img_np.ndim == 2:
            img_np = img_np[..., None]  # expand channel if grayscale
        augmented_img = transform(image=img_np)["image"]
        augmented.append(augmented_img)
    return torch.stack(augmented)


pdo_metadata = pd.read_csv("/user/sina.garazhian/u12203/panc_cell/pdo_data_drug_info_rectal_d5_s3_chr_sx5_ts_combined_v3_corrected.csv", dtype=object)
patients = set(pdo_metadata['patient_name'].values)
patient_dict = {patient: idx for idx, patient in enumerate(patients)}

def compute_prototypes(embeddings, labels) -> Dict[int, torch.Tensor]:
    prototypes = {}
    for pid in torch.unique(labels):
        mask = labels == pid
        if mask.sum() > 0:
            proto = embeddings[mask].mean(dim=0)
            prototypes[pid.item()] = proto
    return prototypes


def train_model(model, train_loader, val_loader, device, epochs=20):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_losses, train_preds, train_targets = [], [], []
        for imgs, _, _, labels in train_loader:
            labels = torch.Tensor([float(patient_dict[patient]) for patient in labels])
            # print(labels)
            imgs, labels = apply_albumentations(torch.permute(imgs.repeat(1, 3, 1, 1), (0, 2, 3, 1)), albumentations_transform).to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        best_score = 1000
        all_embeddings, all_labels = [], []
        val_losses, val_preds, val_targets = [], [], []
        with torch.no_grad():
            for imgs, _, _, labels in val_loader:
                labels = torch.Tensor([float(patient_dict[patient]) for patient in labels])
                imgs, labels = imgs.repeat(1, 3, 1, 1).to(device), labels.long().to(device)
                embeds = model.model(imgs)
                outputs = model(imgs)
                all_embeddings.append(embeds)
                all_labels.append(labels)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                val_targets.extend(labels.cpu().tolist())
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            sep_stats = utils.compute_class_separability(all_embeddings, all_labels)
            current_sep_score = sep_stats["separability_score"]
            if sum(val_losses)/len(val_losses) < best_score:
                best_score = sum(val_losses)/len(val_losses)
                best_model = model
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}: Train Loss={sum(train_losses)/len(train_losses):.4f} Acc={train_acc:.4f}, Val Loss={sum(val_losses)/len(val_losses):.4f} Acc={val_acc:.4f}, class_sep={current_sep_score}")
    return best_model


device = 'cuda'
model = utils.model
model.to(device)
control_train = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_cleaned_train.pt", weights_only=False)
control_val = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_cleaned_val.pt", weights_only=False)
control_test = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/Control_40_cleaned_test.pt", weights_only=False)


labels = [label for _,_,_, label in control_train]  # your dataset labels
sampler = dataset.BalancedBatchSampler(labels, patients_per_batch=8, samples_per_patient=8)
train_loader = DataLoader(control_train, batch_sampler=sampler)
val_loader = DataLoader(control_val, batch_size=64)
test_loader = DataLoader(control_test, batch_size=64)



model_clf = train_model(model, train_loader, val_loader, device, epochs=20)
torch.save(model_clf, "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/resnet50_best_represent.pt")

def evaluate_classifier(model_clf, test_loader, class_names=None, device='cuda', plot_cm=True):
    
    model_clf.to(device)
    
    model_clf.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, _, _, labels in test_loader:
            images, labels = images.repeat(1, 3, 1, 1).to(device), torch.tensor([patient_dict[pat] for pat in labels], device=device)
            logits = model_clf(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")

    # F1 scores
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    if plot_cm:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/confusion_mat_epoch10.png")
        # plt.show()

    return {
        "accuracy": acc,
        "f1_macro": f1_score(all_labels, all_preds, average='macro'),
        "f1_micro": f1_score(all_labels, all_preds, average='micro'),
        "f1_per_class": f1_score(all_labels, all_preds, average=None),
        "confusion_matrix": cm
    }
    
class_names = np.unique([patient_dict[pat] for pat in labels])
metrics = evaluate_classifier(model_clf, test_loader, class_names=class_names)
