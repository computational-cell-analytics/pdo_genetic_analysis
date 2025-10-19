import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EncoderUNet, DecoderUNet, Discriminator1D, LatentClassifierHead
from losses import *

# === Hyperparameters ===
batch_size = 64
latent_dim = 350
epochs = 50
lr_vae = 1e-4
lr_disc = 2e-5
lr_gen = 2e-5
adv_weight = 0.05
zreg_weight = 0.1
alpha = 0.5  # for hybrid loss
warmup_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Models ===
encoder = EncoderUNet(in_channels=3, latent_dim=latent_dim).to(device)
decoder = DecoderUNet(latent_dim=latent_dim).to(device)
discriminator = Discriminator1D(latent_dim=latent_dim).to(device)
classifier = LatentClassifierHead(latent_dim=latent_dim).to(device)  # optional

# === Losses ===
percep_loss_fn = PerceptionLossVGG(device)
pixel_loss_fn = PixelwiseLoss(mode='l2')

# === Optimizers ===
opt_vae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr_vae)
opt_disc = optim.AdamW(discriminator.parameters(), lr=lr_disc, weight_decay=1e-4)
opt_gen = optim.Adam(encoder.parameters(), lr=lr_gen)
opt_clf = optim.Adam(classifier.parameters(), lr=lr_vae)  # optional

# === Placeholder for dataset ===
# train_loader = DataLoader(MyCustomDataset(...), batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    running_recon = 0
    running_disc = 0
    running_gen = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # === VAE: Encoder + Decoder ===
        discriminator.eval()
        encoder.train()
        decoder.train()

        z = encoder(imgs)
        recon = decoder(z)

        rec_loss = hybrid_reconstruction_loss(percep_loss_fn, pixel_loss_fn, recon, imgs, alpha)
        z_mean_loss = latent_z_mean_loss(z)
        z_var_loss = latent_z_variance_loss(z)
        total_vae_loss = rec_loss + zreg_weight * (z_mean_loss + z_var_loss)

        opt_vae.zero_grad()
        total_vae_loss.backward()
        opt_vae.step()
        running_recon += total_vae_loss.item()

        # === Discriminator Phase ===
        if epoch >= warmup_epochs:
            encoder.eval()
            discriminator.train()

            z_fake = encoder(imgs).detach()
            z_real = torch.randn_like(z_fake).to(device)

            d_loss = discriminator_loss(discriminator, z_fake, z_real)
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()
            running_disc += d_loss.item()

        # === Generator (Adversarial) Phase ===
        if epoch >= warmup_epochs:
            encoder.train()
            discriminator.eval()

            z_adv = encoder(imgs)
            g_loss = adv_weight * generator_loss(discriminator, z_adv)
            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()
            running_gen += g_loss.item()

        # === Optional Classifier ===
        # clf_logits = classifier(z.detach())
        # clf_loss = classifier_loss(clf_logits, labels)
        # opt_clf.zero_grad()
        # clf_loss.backward()
        # opt_clf.step()

    print(f"Epoch {epoch}: VAE loss={running_recon:.3f}, D loss={running_disc:.3f}, G loss={running_gen:.3f}")
