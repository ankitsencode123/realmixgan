import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_palette("husl")

def save_generated_grid(images, path, nrow=4):
    images = images.copy()
    images = (images + 1) / 2
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_losses(g_losses, d_losses, path):
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=200)
    plt.close()

