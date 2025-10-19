import torch.optim as optim
from sklearn.metrics import accuracy_score
import utils
import torch
import dataset
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import helper_function
import pandas as pd
import numpy as np
from dataset import custom_collate_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

device = 'cuda'
control_train = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_train_v1.pt", weights_only=False)
control_val = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_val_v1.pt", weights_only=False)
control_test = torch.load( "/user/sina.garazhian/u12203/lustere-grete-mine/patient_classifier/cleaned_Control_40_metadata_test_v1.pt", weights_only=False)


labels = [label for _,_,_, _, _, label, _ in control_train]  # your dataset labels
sampler = dataset.BalancedBatchSampler(labels, patients_per_batch=8, samples_per_patient=8)
train_loader = DataLoader(control_train, batch_sampler=sampler, collate_fn=custom_collate_fn)
val_loader = DataLoader(control_val, batch_size=64, collate_fn=custom_collate_fn)
test_loader = DataLoader(control_test, batch_size=64, collate_fn=custom_collate_fn)








# Re-execute after kernel reset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from typing import Dict, List


# Contrastive loss using patient prototypes
def prototype_contrastive_loss(embeddings, labels, prototypes, margin=1):
    loss = 0.0
    for i in range(embeddings.size(0)):
        emb_i = embeddings[i]
        label_i = labels[i].item()
        for pid, proto in prototypes.items():
            dist = torch.norm(emb_i - proto)
            if pid == label_i:
                loss += 0.5 * dist.pow(2)
            else:
                loss += 0.5 * torch.clamp(margin - dist, min=0).pow(2)
    return loss / embeddings.size(0)

def compute_prototypes(embeddings, labels) -> Dict[int, torch.Tensor]:
    prototypes = {}
    for pid in torch.unique(labels):
        mask = labels == pid
        if mask.sum() > 0:
            proto = embeddings[mask].mean(dim=0)
            prototypes[pid.item()] = proto
    return prototypes

# Full training script
def train_model(model, train_loader, device='cuda', epochs=50, patients_per_batch=8, samples_per_patient=8, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_sep_score = 0.0
    patience_counter = 0
    sep_checkpoints = []  # list of saved model filenames
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_embeddings, all_labels = [], []

        # Step 1: cache all batches with augmentation
        augmented_batches = []
        for images, _, _, labels, _ in train_loader:
            # Apply albumentations ONCE
            images_aug = apply_albumentations(
                torch.permute(images.repeat(1, 3, 1, 1), (0, 2, 3, 1)),
                albumentations_transform
            ).to(device)
            
            images_aug = images_aug[:, 0, :, :].unsqueeze(1)  # if you want grayscale again
            # print(images_aug.shape)
            labels_tensor = torch.tensor([patient_dict[pat] for pat in labels], device=device)

            augmented_batches.append((images_aug, labels_tensor))

            # Save for prototype
            with torch.no_grad():
                emb = model(images_aug)
                all_embeddings.append(emb)
                all_labels.append(labels_tensor)

        # Step 2: compute prototypes from cached embeddings
        prototypes = compute_prototypes(torch.cat(all_embeddings), torch.cat(all_labels))

        # Step 3: training step — reuse same augmented batches
        for images_aug, labels_tensor in augmented_batches:
            optimizer.zero_grad()
            embeddings = model(images_aug)
            loss = prototype_contrastive_loss(embeddings, labels_tensor, prototypes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(augmented_batches):.4f}")
        # Step 3: validation loss every 2 epochs
        if epoch > 0: #(epoch + 1) % 2 == 0:
            all_embeddings, all_labels = [], []
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _, _, labels, _ in val_loader:
                    images = images.to(device)
                    labels = torch.tensor([patient_dict[pat] for pat in labels], device=device)
                    embeddings = model(images)
                    loss = prototype_contrastive_loss(embeddings, labels, prototypes)
                    val_loss += loss.item()
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
                all_embeddings = torch.cat(all_embeddings, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                sep_stats = utils.compute_class_separability(all_embeddings, all_labels)
            avg_val_loss = val_loss / len(val_loader)
            current_sep_score = sep_stats["separability_score"]
            print(f"          ↪ Validation Loss: {avg_val_loss:.4f} | Separability Score: {current_sep_score:.4f} | "
                f"Intra: {sep_stats['intra_class_dist']:.4f} | "
                f"Inter: {sep_stats['inter_class_dist']:.4f}")

            if current_sep_score >= best_sep_score:
                # Model is stable or improved — keep as best so far
                best_model = model
                best_sep_score = current_sep_score
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⚠️ Separability dropped for {patience_counter} epoch(s).")

            if patience_counter >= 2:
                # Save current model (not best) after plateau for 2 epochs
                model_path = f"/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/resnet18_encoder_new_setup_model_epoch{epoch+1}_sep{current_sep_score:.4f}_new_data.pt"
                torch.save(best_model.state_dict(), model_path)
                sep_checkpoints.append(model_path)
                print(f"✅ Saved model with separability {current_sep_score:.4f} at epoch {epoch+1}")
                features, labels = helper_function.extract_flat_features(best_model)
                print("getting featurs done")
                helper_function.tsne_and_plot(features, labels, f"{epoch + 1}_88")
                # Accept current as new baseline
                # best_sep_score = current_sep_score
                patience_counter = 0

    return best_model


###doing representation learning
# model = utils.CellImageEncoder()
encoder = utils.resnet_encoder_18()
# model = utils.resnet_enocer()
# model_trained = train_model(model, train_loader, device='cuda', epochs=50)
# torch.save(model_trained, "/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/represent_model_epoch50.pt")



# Training loop
def train_classifier(model_clf, encoder, train_loader, val_loader, device='cuda', epochs=15, lr=1e-3):
    encoder.to(device)
    encoder.eval()
    model_clf.to(device)
    optimizer = torch.optim.Adam(model_clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_score = 1000
    for epoch in range(epochs):
        model_clf.train()
        train_correct, train_total, train_loss = 0, 0, 0
        for images, _, _, labels, _ in train_loader:
            images, labels = images.to(device), torch.tensor([patient_dict[pat] for pat in labels], device=device)
            optimizer.zero_grad()
            embeds = encoder(images)
            embeds.to(device)
            logits = model_clf(embeds)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model_clf.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, _, _, labels, _ in val_loader:
                images, labels = images.to(device), torch.tensor([patient_dict[pat] for pat in labels], device=device)
                embeds = encoder(images)
                embeds.to(device)
                logits = model_clf(embeds)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
            if val_loss <= best_score:
                best_score = val_loss
                best_model = model_clf
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.3f} | Train Acc={train_correct/train_total:.3f} | Val Acc={val_correct/val_total:.3f} | Val Loss={val_loss:.3f}")
    return best_model

###doing classification task
embeddings_size = 128
n_patients = 10
model_clf = utils.ClassifierHead(in_dim=embeddings_size, n_classes=n_patients)
# encoder = utils.CellImageEncoder()
encoder.load_state_dict(torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/checkpoints_88/resnet18_encoder_new_setup_model_epoch42_sep2.1274_new_data.pt"))

model_clf = train_classifier(model_clf, encoder, train_loader, val_loader)
# torch.save(model_clf, "/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/clf_model_epoch20_new_data_res18.pt")
# model_clf = torch.load("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/clf_model_epoch20_new_data_res18.pt", weights_only=False)

def evaluate_classifier(encoder, model_clf, test_loader, class_names=None, device='cuda', plot_cm=True):
    encoder.to(device)
    model_clf.to(device)
    encoder.eval()
    model_clf.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, _, _, labels, _ in test_loader:
            images, labels = images.to(device), torch.tensor([patient_dict[pat] for pat in labels], device=device)
            embeds = encoder(images)
            logits = model_clf(embeds)
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
        plt.savefig("/user/sina.garazhian/u12203/lustere-grete-mine/representaion_learning/confusion_mat_epoch88_20_20_new_data_res18.png")
        # plt.show()

    return {
        "accuracy": acc,
        "f1_macro": f1_score(all_labels, all_preds, average='macro'),
        "f1_micro": f1_score(all_labels, all_preds, average='micro'),
        "f1_per_class": f1_score(all_labels, all_preds, average=None),
        "confusion_matrix": cm
    }
class_names = np.unique([patient_dict[pat] for pat in labels])
metrics = evaluate_classifier(encoder, model_clf, test_loader, class_names=class_names)
