import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from src.data import get_mnist_dataloader
from src.models import StandardGAN, ImprovedGAN, z_dim, image_size
from src.visualize import save_generated_grid, plot_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
learning_rate = 0.0002
num_epochs = 70
real_data_mix_ratio = 0.2
noise_scale = 0.2
mix_probability = 0.9

os.makedirs('outputs', exist_ok=True)
os.makedirs('docs/figures', exist_ok=True)

dataloader = get_mnist_dataloader()

standard_gan = StandardGAN(device)
improved_gan = ImprovedGAN(device)

criterion = nn.BCELoss()
optimizer_G_std = optim.Adam(standard_gan.generator.parameters(), lr=learning_rate)
optimizer_D_std = optim.Adam(standard_gan.discriminator.parameters(), lr=learning_rate)
optimizer_G_imp = optim.Adam(improved_gan.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_imp = optim.Adam(improved_gan.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

std_g_losses, std_d_losses, std_wd, std_generated = [], [], [], []
imp_g_losses, imp_d_losses, imp_wd, imp_generated = [], [], [], []

real_images_buffer = []

for epoch in range(num_epochs):
    epoch_d_loss_std, epoch_g_loss_std = 0.0, 0.0
    epoch_d_loss_imp, epoch_g_loss_imp = 0.0, 0.0
    real_samples_std, fake_samples_std = [], []
    real_samples_imp, fake_samples_imp = [], []
    bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, (real_images, _) in enumerate(bar):
        bc = real_images.size(0)
        real_images = real_images.view(bc, -1).to(device)
        real_labels = torch.ones(bc, 1).to(device)
        fake_labels = torch.zeros(bc, 1).to(device)

        optimizer_D_std.zero_grad()
        output_real = standard_gan.discriminator(real_images)
        loss_real = criterion(output_real, real_labels)
        z = torch.randn(bc, z_dim).to(device)
        fake_images = standard_gan.generator(z)
        output_fake = standard_gan.discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_D_std.step()

        optimizer_G_std.zero_grad()
        output_fake_for_g = standard_gan.discriminator(fake_images)
        g_loss = criterion(output_fake_for_g, real_labels)
        g_loss.backward()
        optimizer_G_std.step()

        epoch_d_loss_std += d_loss.item()
        epoch_g_loss_std += g_loss.item()

        if i % 20 == 0:
            real_samples_std.extend(real_images.cpu().numpy()[:10])
            fake_samples_std.extend(fake_images.detach().cpu().numpy()[:10])

        if len(real_images_buffer) < 1000:
            real_images_buffer.extend(real_images.cpu().numpy()[:max(1, bc//4)])
        else:
            num_replace = min(bc//4, len(real_images_buffer)//10)
            start_idx = np.random.randint(0, len(real_images_buffer) - num_replace + 1)
            for j in range(num_replace):
                real_images_buffer[start_idx + j] = real_images[j].cpu().numpy()

        optimizer_D_imp.zero_grad()
        output_real_imp = improved_gan.discriminator(real_images)
        loss_real_imp = criterion(output_real_imp, real_labels)

        z_imp = torch.randn(bc, z_dim).to(device)
        use_mixing = np.random.random() < mix_probability and len(real_images_buffer) >= bc
        if use_mixing:
            buffer_indices = np.random.choice(len(real_images_buffer), bc, replace=True)
            sampled_real = torch.tensor([real_images_buffer[idx] for idx in buffer_indices]).float().to(device)
            noisy_real = sampled_real + torch.randn_like(sampled_real) * noise_scale
            noisy_real = torch.clamp(noisy_real, -1, 1)
            mixed_input = torch.cat([z_imp, noisy_real * real_data_mix_ratio], dim=1)
        else:
            mixed_input = torch.cat([z_imp, torch.randn(bc, image_size).to(device) * 0.1], dim=1)

        fake_images_imp = improved_gan.generator(mixed_input)
        output_fake_imp = improved_gan.discriminator(fake_images_imp.detach())
        loss_fake_imp = criterion(output_fake_imp, fake_labels)
        d_loss_imp = loss_real_imp + loss_fake_imp
        d_loss_imp.backward()
        optimizer_D_imp.step()

        optimizer_G_imp.zero_grad()
        output_fake_for_g_imp = improved_gan.discriminator(fake_images_imp)
        g_loss_imp = criterion(output_fake_for_g_imp, real_labels)
        g_loss_imp.backward()
        optimizer_G_imp.step()

        epoch_d_loss_imp += d_loss_imp.item()
        epoch_g_loss_imp += g_loss_imp.item()

        if i % 15 == 0:
            real_samples_imp.extend(real_images.cpu().numpy()[:8])
            fake_samples_imp.extend(fake_images_imp.detach().cpu().numpy()[:8])

        bar.set_postfix({'D_std': d_loss.item(), 'G_std': g_loss.item(), 'D_imp': d_loss_imp.item(), 'G_imp': g_loss_imp.item()})

    std_g_losses.append(epoch_g_loss_std/len(dataloader))
    std_d_losses.append(epoch_d_loss_std/len(dataloader))
    imp_g_losses.append(epoch_g_loss_imp/len(dataloader))
    imp_d_losses.append(epoch_d_loss_imp/len(dataloader))

    if real_samples_std and fake_samples_std:
        real_flat = np.array(real_samples_std).flatten()
        fake_flat = np.array(fake_samples_std).flatten()
        wd = wasserstein_distance(real_flat, fake_flat)
        std_wd.append(wd)

    if real_samples_imp and fake_samples_imp:
        real_flat = np.array(real_samples_imp).flatten()
        fake_flat = np.array(fake_samples_imp).flatten()
        wd_imp = wasserstein_distance(real_flat, fake_flat)
        imp_wd.append(wd_imp)

    with torch.no_grad():
        z_eval = torch.randn(16, z_dim).to(device)
        fake_eval_std = standard_gan.generator(z_eval).view(-1, 28, 28).cpu().numpy()
        fake_eval_imp = None
        if len(real_images_buffer) >= 16:
            idxs = np.random.choice(len(real_images_buffer), 16, replace=True)
            sampled_real = torch.tensor([real_images_buffer[idx] for idx in idxs]).float().to(device)
            noisy_real = sampled_real + torch.randn_like(sampled_real) * noise_scale
            noisy_real = torch.clamp(noisy_real, -1, 1)
            mixed_eval = torch.cat([z_eval, noisy_real * real_data_mix_ratio], dim=1)
            fake_eval_imp = improved_gan.generator(mixed_eval).view(-1, 28, 28).cpu().numpy()
        else:
            mixed_eval = torch.cat([z_eval, torch.randn(16, image_size).to(device) * 0.1], dim=1)
            fake_eval_imp = improved_gan.generator(mixed_eval).view(-1, 28, 28).cpu().numpy()

        save_generated_grid(fake_eval_std, f'plots_std_epoch_{epoch+1}.png')
        save_generated_grid(fake_eval_imp, f'plots_imp_epoch_{epoch+1}.png')

    if len(std_wd) > 0:
        pass

plot_losses(std_g_losses, std_d_losses, 'outputs/standard_losses.png')
plot_losses(imp_g_losses, imp_d_losses, 'outputs/improved_losses.png')
if len(std_wd) > 0 and len(imp_wd) > 0:
    final_std = std_wd[-1]
    final_imp = imp_wd[-1]
torch.save(standard_gan.generator.state_dict(), 'outputs/standard_generator.pth')
torch.save(improved_gan.generator.state_dict(), 'outputs/improved_generator.pth')

