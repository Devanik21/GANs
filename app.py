import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import gc
import os

# Set page config with colorful theme
st.set_page_config(
    page_title="🎨 AI Art Generator - GAN Training Studio", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎨"
)

# Custom CSS for better styling using Streamlit's color system
st.markdown("""
<style>
    .big-title {
        font-size: 3rem !important;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Device configuration for memory efficiency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== SIMPLE GAN MODELS =====================
class SimpleGenerator(nn.Module):
    """Lightweight generator network"""
    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        
        # Calculate initial size for reshape
        init_size = img_size // 4
        self.init_size = init_size
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 128 * init_size ** 2),
            nn.LeakyReLU(0.2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class SimpleDiscriminator(nn.Module):
    """Lightweight discriminator network"""
    def __init__(self, img_channels=3, img_size=64):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2):
            return [
                nn.Conv2d(in_filters, out_filters, 3, stride, 1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, 2),
            *discriminator_block(16, 32, 2),
            *discriminator_block(32, 64, 2),
            *discriminator_block(64, 128, 2),
        )
        
        # Calculate the size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# ===================== DCGAN MODELS =====================
# ===================== DCGAN MODELS (DYNAMIC & IMPROVED) =====================
# ===================== DCGAN MODELS (DYNAMIC & IMPROVED) =====================
# ===================== DCGAN MODELS (DYNAMIC & IMPROVED) =====================
class Generator(nn.Module):
    """Powerful and FLEXIBLE DCGAN-style generator"""
    def __init__(self, latent_dim=100, img_channels=3, img_size=64, feature_maps=64):
        super().__init__()
        
        # Calculate the number of upsampling blocks needed to go from 4x4 to img_size
        num_blocks = int(np.log2(img_size) - 2)
        if num_blocks < 1:
            raise ValueError(f"Image size {img_size} is too small for this DCGAN architecture. Minimum is 8.")

        layers = []
        
        # Initial layer: from latent_dim to a 4x4 feature map
        out_features = feature_maps * (2 ** (num_blocks - 1))
        layers.extend([
            nn.ConvTranspose2d(latent_dim, out_features, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True)
        ])
        
        # Upsampling blocks
        for i in range(num_blocks - 1):
            in_features = out_features
            out_features = in_features // 2
            layers.extend([
                nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(True)
            ])
            
        # Final layer: to img_size and target channels
        layers.extend([
            nn.ConvTranspose2d(out_features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        # Reshape z from (batch, latent_dim) to (batch, latent_dim, 1, 1)
        z = z.view(z.shape[0], -1, 1, 1)
        return self.main(z)

class Discriminator(nn.Module):
    """Powerful and FLEXIBLE DCGAN-style discriminator (Critic)"""
    def __init__(self, img_channels=3, img_size=64, feature_maps=64):
        super().__init__()
        
        # Calculate the number of downsampling blocks
        num_blocks = int(np.log2(img_size) - 2)
        if num_blocks < 1:
            raise ValueError(f"Image size {img_size} is too small for this DCGAN architecture. Minimum is 8.")

        layers = []
        
        # Initial layer
        layers.extend([
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        in_features = feature_maps
        # Downsampling blocks
        for i in range(num_blocks - 1):
            out_features = in_features * 2
            layers.extend([
                nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_features = out_features
            
        # Final layer: downsample to a 1x1 feature map (a single score)
        layers.extend([
            nn.Conv2d(in_features, 1, 4, 1, 0, bias=False)
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, img):
        return self.main(img)

class Discriminator(nn.Module):
    """Powerful and FLEXIBLE DCGAN-style discriminator (Critic)"""
    def __init__(self, img_channels=3, img_size=64, feature_maps=64):
        super().__init__()
        
        # Calculate the number of downsampling blocks
        num_blocks = int(np.log2(img_size) - 2)
        if num_blocks < 1:
            raise ValueError(f"Image size {img_size} is too small for this DCGAN architecture. Minimum is 8.")

        layers = []
        
        # Initial layer
        layers.extend([
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        in_features = feature_maps
        # Downsampling blocks
        for i in range(num_blocks - 1):
            out_features = in_features * 2
            layers.extend([
                nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_features = out_features
            
        # Final layer: downsample to a 1x1 feature map (a single score)
        layers.extend([
            nn.Conv2d(in_features, 1, 4, 1, 0, bias=False)
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, img):
        return self.main(img)

class Discriminator(nn.Module):
    """Powerful and FLEXIBLE DCGAN-style discriminator (Critic)"""
    def __init__(self, img_channels=3, img_size=64, feature_maps=64):
        super().__init__()
        
        # Calculate the number of downsampling blocks
        num_blocks = int(np.log2(img_size) - 2)
        if num_blocks < 1:
            raise ValueError(f"Image size {img_size} is too small for this DCGAN architecture. Minimum is 8.")

        layers = []
        
        # Initial layer
        layers.extend([
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        in_features = feature_maps
        # Downsampling blocks
        for i in range(num_blocks - 1):
            out_features = in_features * 2
            layers.extend([
                nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_features = out_features
            
        # Final layer: downsample to a 1x1 feature map (a single score)
        layers.extend([
            nn.Conv2d(in_features, 1, 4, 1, 0, bias=False)
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, img):
        return self.main(img)

class Discriminator(nn.Module):
    """Powerful DCGAN-style discriminator (Critic)"""
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input size: (img_channels) x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            # Final output is a single raw score, not a probability
        )

    def forward(self, img):
        return self.main(img)

class ImageDataset(Dataset):
    """Custom dataset for uploaded images"""
    def __init__(self, images, transform=None, img_size=64):
        self.images = images
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        return image

def preprocess_images(uploaded_files, img_size=64):
    """Preprocess uploaded images"""
    images = []
    valid_files = 0
    
    progress_bar = st.progress(0, text="📄 Processing images...")
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        try:
            # Update progress with colorful text
            progress_bar.progress((idx + 1) / len(uploaded_files), 
                                text=f"🎨 Processing image {idx + 1}/{len(uploaded_files)}")
            status_text.text(f"📸 Current: {file.name}")
            
            image = Image.open(file)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic validation - check if image is too small
            if image.size[0] < 16 or image.size[1] < 16:
                st.warning(f"⚠️ Skipping {file.name}: image too small (minimum 16x16)")
                continue
                
            images.append(image)
            valid_files += 1
            
        except Exception as e:
            st.error(f"❌ Could not process file {file.name}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return images, valid_files

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_simple_gan(images, epochs=50, batch_size=4, lr=0.0002, latent_dim=100, img_size=64):
    """Train the Simple GAN with basic loss"""
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(images, transform=transform, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if len(dataloader) == 0:
        st.error(f"🚫 Cannot start training. The number of valid images ({len(dataset)}) is less than the batch size ({batch_size}). Please upload more images or reduce the batch size.")
        return None, None, [], []

    # Initialize networks
    generator = SimpleGenerator(latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = SimpleDiscriminator(img_size=img_size).to(device)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training progress containers with colorful headers
    st.markdown("### 🎯 Training Progress Dashboard")
    progress_bar = st.progress(0, text="🚀 Starting training...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        d_loss_metric = st.metric("🔴 Discriminator Loss", "0.0000")
    with col2:
        g_loss_metric = st.metric("🔵 Generator Loss", "0.0000") 
    with col3:
        epoch_metric = st.metric("⏳ Current Epoch", "0")
    
    loss_chart = st.empty()
    sample_container = st.empty()
    
    # Training history
    d_losses = []
    g_losses = []
    
    # Training loop
    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        batches = 0
        
        for i, real_images in enumerate(dataloader):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            batches += 1
            
            # Memory cleanup
            del real_images, fake_images, z, real_output, fake_output
            if i % 5 == 0:  # Cleanup every 5 batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        if batches == 0:
            st.warning(f"⚠️ Epoch {epoch+1}/{epochs} - No batches processed. Is number of images < batch size?")
            continue
        
        # Record losses
        avg_d_loss = epoch_d_loss / batches
        avg_g_loss = epoch_g_loss / batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Update progress and metrics with emojis
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress, text=f"🎨 Training in progress... Epoch {epoch+1}/{epochs}")
        
        d_loss_metric.metric("🔴 Discriminator Loss", f"{avg_d_loss:.4f}")
        g_loss_metric.metric("🔵 Generator Loss", f"{avg_g_loss:.4f}")
        epoch_metric.metric("⏳ Current Epoch", f"{epoch+1}/{epochs}")
        
        # Update loss chart every 5 epochs with colorful styling
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(d_losses, label='🔴 Discriminator Loss', color='#ff6b6b', linewidth=2.5, alpha=0.8)
            ax.plot(g_losses, label='🔵 Generator Loss', color='#4ecdc4', linewidth=2.5, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax.set_title('📈 Training Loss Curves', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            loss_chart.pyplot(fig)
            plt.close(fig)
        
        # Generate sample images every 10 epochs with better styling
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                sample_z = torch.randn(4, latent_dim, device=device)
                sample_images = generator(sample_z)
                sample_images = (sample_images + 1) / 2  # Denormalize
                
                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                fig.suptitle(f'🎨 Generated Images - Epoch {epoch+1}', fontsize=16, fontweight='bold', y=1.05)
                
                for j in range(4):
                    img = sample_images[j].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[j].imshow(img)
                    axes[j].axis('off')
                    axes[j].set_title(f'✨ Sample {j+1}', fontsize=12, pad=10)
                    # Add colorful border
                    for spine in axes[j].spines.values():
                        spine.set_edgecolor(['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'][j])
                        spine.set_linewidth(3)
                
                plt.tight_layout()
                sample_container.pyplot(fig)
                plt.close(fig)
                
                del sample_z, sample_images
    
    return generator, discriminator, d_losses, g_losses

def train_dcgan_wgan(images, epochs=50, batch_size=4, lr=0.0002, latent_dim=100, img_size=64):
    """Train the WGAN-GP with memory-efficient approach and data augmentation"""


    if img_size != 64:
    st.warning(f"⚠️ DCGAN-WGAN model is optimized for 64x64 images. Forcing resolution to 64px.")
    img_size = 64
    # 1. HEAVY DATA AUGMENTATION
    # This is key for training on a small number of images
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(images, transform=transform, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if len(dataloader) == 0:
        st.error(f"🚫 Cannot start training. The number of valid images ({len(dataset)}) is less than the batch size ({batch_size}). Please upload more images or reduce the batch size.")
        return None, None, [], []

    # 2. UPDATED MODEL INITIALIZATION (PASSING IMG_SIZE)
    # This now uses the flexible models which adapt to the image size
    generator = Generator(latent_dim=latent_dim, img_channels=3, img_size=img_size, feature_maps=64).to(device)
     # Correct line
    discriminator = Discriminator(img_channels=3, feature_maps=64).to(device)
    
    # WGAN-GP parameters
    lambda_gp = 10
    n_critic = 5 # Train discriminator more often than generator
    
    # Use Adam optimizers with parameters recommended for WGAN
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    st.markdown("### 🎯 Training Progress Dashboard")
    progress_bar = st.progress(0, text="🚀 Starting training...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        d_loss_metric = st.metric("🔴 Critic Loss", "0.00")
    with col2:
        g_loss_metric = st.metric("🔵 Generator Loss", "0.00") 
    with col3:
        epoch_metric = st.metric("⏳ Current Epoch", "0")
    
    loss_chart = st.empty()
    sample_container = st.empty()
    
    d_losses = []
    g_losses = []
    
    # 3. NEW WGAN-GP TRAINING LOOP
    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        batches = 0
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # ---------------------
            #  Train Discriminator (Critic)
            # ---------------------
            optimizer_D.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = generator(z)
            
            # Real and fake images scores
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images.detach())
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
            
            # Critic loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            
            # Train the generator only every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate a batch of images
                # Reuse the z from the critic step for efficiency
                gen_imgs = generator(z)
                
                # Generator loss
                g_loss = -torch.mean(discriminator(gen_imgs))
                
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()

            batches += 1
            
            # Memory cleanup
            del real_images, fake_images, z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if batches == 0:
            st.warning(f"⚠️ Epoch {epoch+1}/{epochs} - No batches processed.")
            continue
        
        avg_d_loss = epoch_d_loss / batches
        avg_g_loss = epoch_g_loss / (batches / n_critic) if batches > 0 else 0
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress, text=f"🎨 Training in progress... Epoch {epoch+1}/{epochs}")
        
        d_loss_metric.metric("🔴 Critic Loss", f"{avg_d_loss:.2f}")
        g_loss_metric.metric("🔵 Generator Loss", f"{avg_g_loss:.2f}")
        epoch_metric.metric("⏳ Current Epoch", f"{epoch+1}/{epochs}")
        
        # Update loss chart every 5 epochs with colorful styling
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(d_losses, label='🔴 Critic Loss', color='#ff6b6b', linewidth=2.5, alpha=0.8)
            ax.plot(g_losses, label='🔵 Generator Loss', color='#4ecdc4', linewidth=2.5, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax.set_title('📈 Training Loss Curves', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            loss_chart.pyplot(fig)
            plt.close(fig)

        # Generate sample images every 10 epochs with better styling
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                sample_z = torch.randn(4, latent_dim, device=device)
                sample_images = generator(sample_z)
                sample_images = (sample_images + 1) / 2
                
                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                fig.suptitle(f'🎨 Generated Images - Epoch {epoch+1}', fontsize=16, fontweight='bold', y=1.05)
                
                for j in range(4):
                    img = sample_images[j].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[j].imshow(img)
                    axes[j].axis('off')
                    axes[j].set_title(f'✨ Sample {j+1}', fontsize=12, pad=10)
                    for spine in axes[j].spines.values():
                        spine.set_edgecolor(['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'][j])
                        spine.set_linewidth(3)
                
                plt.tight_layout()
                sample_container.pyplot(fig)
                plt.close(fig)
                
                del sample_z, sample_images
                
    return generator, discriminator, d_losses, g_losses

def generate_images(generator, num_images=8, latent_dim=100, is_simple=False):
    """Generate new images using trained generator"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        if not is_simple:
            # For DCGAN, reshape z for ConvTranspose2d input
            z = z.view(z.shape[0], -1, 1, 1)
        generated_images = generator(z)
        generated_images = (generated_images + 1) / 2  # Denormalize
        
        # Create grid with better styling
        rows = 2
        cols = (num_images + 1) // rows
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
        
        for i in range(num_images):
            img = generated_images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'✨ Generated #{i+1}', fontsize=11, fontweight='bold', pad=8)
            # Add colorful border
            for spine in axes[i].spines.values():
                spine.set_edgecolor(colors[i % len(colors)])
                spine.set_linewidth(3)
        
        # Turn off unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle('🎨 AI Generated Masterpieces', fontsize=18, fontweight='bold', y=0.98)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        return fig

# Streamlit UI
def main():
    # Colorful animated title
    st.markdown('<h1 class="big-title">🎨 AI Art Generator Studio</h1>', unsafe_allow_html=True)
    
    # Colorful subtitle with gradient
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3 style="color: #666; font-weight: 300;">
            ✨ Train powerful GANs on your images • 🚀 Multiple model architectures • 🎯 Professional results
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with colorful styling
    with st.sidebar:
        st.markdown("### 🎛️ **Training Control Panel**")
        
        # Add some colorful info boxes
        st.info("💡 **Pro Tip**: DCGAN-WGAN works better with fewer images!")
        
        # MODEL SELECTION DROPDOWN
        st.markdown("#### 🤖 **Model Architecture**")
        model_type = st.selectbox(
            "🧠 Choose GAN Model",
            ["Simple GAN", "DCGAN-WGAN"],
            index=0,
            help="Simple GAN: Fast, good for beginners | DCGAN-WGAN: Advanced, better quality"
        )
        
        # Show model info
        if model_type == "Simple GAN":
            st.success("⚡ **Fast Training** • Good for beginners")
        else:
            st.success("🎨 **High Quality** • Advanced architecture")
        
        st.markdown("#### 🖼️ **Image Settings**")
        img_size = st.selectbox("📏 Image Resolution", [32, 64, 128], index=1, 
                               help="Higher = better quality, more memory")
        
        st.markdown("#### ⚡ **Training Parameters**")
        epochs = st.slider("🔥 Training Epochs", 10, 2000, 500, 
                          help="More epochs = better results, longer training")
        batch_size = st.slider("📦 Batch Size", 1, 8, 4,
                              help="Lower if running out of memory")
        lr = st.number_input("🎯 Learning Rate", 0.0001, 0.01, 0.0002, format="%.4f",
                           help="Controls how fast the AI learns")
        latent_dim = st.slider("🌌 Latent Dimension", 50, 200, 100,
                              help="Complexity of generated features")
        
        # Memory warning with emoji
        if img_size > 64 or batch_size > 4:
            st.warning("⚠️ **Memory Alert**: High settings may exceed 4GB limit!")
        else:
            st.success("✅ **Memory**: Settings look good!")
        
        # Device info with colors
        device_emoji = "🚀" if device.type == "cuda" else "💻"
        st.info(f"{device_emoji} **Device**: {device.type.upper()}")
    
    # Main interface with tabs
    tab1, tab2, tab3 = st.tabs(["📤 **Upload & Train**", "🎨 **Generate Art**", "📊 **Model Info**"])
    
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📤 **Upload Your Training Images**")
            
            # Styled file uploader
            uploaded_files = st.file_uploader(
                "🖼️ Choose your images (1-100)", 
                type=['png', 'jpg', 'jpeg'], 
                accept_multiple_files=True,
                help="Upload 1-100 images to train your AI artist! 🎨"
            )
            
            if uploaded_files:
                num_files = len(uploaded_files)
                if num_files > 100:
                    st.error("🚫 **Too many files!** Please upload maximum 100 images")
                    return
                
                # Success message with colors
                st.success(f"🎉 **Success!** {num_files} images ready for training")
                
                # Progress bar for file count
                progress_percentage = min(num_files / 20, 1.0)  # Ideal around 20 images
                st.progress(progress_percentage, f"📈 Dataset completeness: {num_files}/20+ images")
                
                # Show sample images with better styling
                if st.checkbox("👀 **Preview uploaded images**", help="See what you're training on"):
                    st.markdown("#### 🖼️ **Image Preview**")
                    cols = st.columns(min(4, num_files))
                    for i, file in enumerate(uploaded_files[:4]):
                        try:
                            img = Image.open(file)
                            with cols[i]:
                                st.image(img, caption=f"✨ Image {i+1}", use_container_width=True)
                        except Exception as e:
                            cols[i].error(f"❌ Error: {file.name}")
        
        with col2:
            st.markdown("### 🚀 **Start Training**")
            
            if not uploaded_files:
                st.info("👆 **Upload images first** to start training your AI!")
            else:
                # Memory estimation with colors
                estimated_memory = (len(uploaded_files) * img_size * img_size * 3 * 4) / (1024**3)
                
                col_mem1, col_mem2 = st.columns(2)
                with col_mem1:
                    st.metric("💾 **Est. Memory**", f"{estimated_memory:.1f} GB")
                with col_mem2:
                    memory_status = "🟢 Good" if estimated_memory < 2 else "🟡 High" if estimated_memory < 3 else "🔴 Too High"
                    st.metric("📊 **Memory Status**", memory_status)
                
                if estimated_memory > 3:
                    st.warning("⚠️ **High memory usage!** Consider reducing image size or batch size.")
                
                # Display selected model info
                st.info(f"🤖 **Selected Model**: {model_type}")
                
                # Big colorful training button
                if st.button("🚀 **START TRAINING**", type="primary", use_container_width=True):
                    if len(uploaded_files) < 1:
                        st.error("🚫 Please upload at least 1 image")
                        return
                    
                    with st.spinner("🔥 Preprocessing your masterpieces..."):
                        images, valid_files = preprocess_images(uploaded_files, img_size)
                    
                    if valid_files == 0:
                        st.error("❌ No valid images found")
                        return
                    
                    # Celebration animation
                    st.success(f"✅ Processing {valid_files} beautiful images")
                    
                    # Train the GAN based on selected model
                    try:
                        if model_type == "Simple GAN":
                            generator, discriminator, d_losses, g_losses = train_simple_gan(
                                images, epochs, batch_size, lr, latent_dim, img_size
                            )
                        else:  # DCGAN-WGAN
                            generator, discriminator, d_losses, g_losses = train_dcgan_wgan(
                                images, epochs, batch_size, lr, latent_dim, img_size
                            )
                        
                        if generator is not None:
                            # Another celebration
                            st.success("🎉 **Training Complete!** Your AI artist is ready!")
                            
                            # Store models in session state
                            st.session_state['generator'] = generator
                            st.session_state['discriminator'] = discriminator
                            st.session_state['trained'] = True
                            st.session_state['latent_dim'] = latent_dim
                            st.session_state['model_type'] = model_type
                            
                            # Show training summary
                            col_summary1, col_summary2 = st.columns(2)
                            with col_summary1:
                                loss_label = "🎯 **Final D-Loss**" if model_type == "Simple GAN" else "🎯 **Final C-Loss**"
                                st.metric(loss_label, f"{d_losses[-1]:.4f}")
                            with col_summary2:
                                st.metric("🎨 **Final G-Loss**", f"{g_losses[-1]:.4f}")
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            st.error("🔥 **Out of memory!** Try reducing batch size or image size.")
                        else:
                            st.error(f"💥 Training error: {e}")
                    except Exception as e:
                        st.error(f"⚡ Unexpected error: {e}")
    
    with tab2:
        if st.session_state.get('trained', False):
            st.markdown("### 🎨 **Generate Amazing Art**")
            
            col_gen1, col_gen2 = st.columns([1, 2], gap="large")
            
            with col_gen1:
                st.markdown("#### 🎛️ **Generation Settings**")
                num_generate = st.slider("🖼️ **Number of images**", 1, 16, 8,
                                        help="How many masterpieces to create")
                
                st.markdown("#### 🎲 **Random Seed**")
                use_seed = st.checkbox("🔒 **Use fixed seed**", help="For reproducible results")
                if use_seed:
                    seed_value = st.number_input("🌱 **Seed value**", 0, 9999, 42)
                    torch.manual_seed(seed_value)
                
                # Display current model info
                current_model = st.session_state.get('model_type', 'Unknown')
                st.info(f"🤖 **Current Model**: {current_model}")
                
                # Big generation button
                if st.button("🎨 **GENERATE ART**", type="primary", use_container_width=True):
                    with st.spinner("🎭 Your AI is painting masterpieces..."):
                        try:
                            is_simple = st.session_state.get('model_type', 'Simple GAN') == 'Simple GAN'
                            fig = generate_images(
                                st.session_state['generator'], 
                                num_generate, 
                                st.session_state['latent_dim'],
                                is_simple=is_simple
                            )
                            with col_gen2:
                                st.pyplot(fig)
                                st.success("🎉 **Fresh art created!** Save any you like!")
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"🎨 Generation error: {e}")
            
            with col_gen2:
                if 'generator' not in st.session_state:
                    st.info("🎭 **Generated images will appear here** after clicking the generate button!")
        else:
            st.markdown("### 🎨 **Art Generation Studio**")
            st.info("👈 **Train your AI first** in the Upload & Train tab to unlock the art generation magic!")
            
            # Show some example placeholder
            st.markdown("#### 🌟 **What you'll be able to create:**")
            st.markdown("""
            - 🖼️ **Unique AI-generated images** based on your training data
            - 🎨 **Infinite variations** with different random seeds  
            - 📊 **Batch generation** of multiple images at once
            - 🎭 **Style consistency** learned from your uploaded images
            - 🤖 **Choice of model architectures** for different quality levels
            """)
    
    with tab3:
        st.markdown("### 📊 **Model Architecture & Info**")
        
        if st.session_state.get('trained', False):
            current_model = st.session_state.get('model_type', 'Unknown')
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("#### 🧠 **Generator Network**")
                st.success("✅ **Status**: Trained and Ready")
                st.info(f"🤖 **Model Type**: {current_model}")
                st.info(f"🌌 **Latent Dimensions**: {st.session_state.get('latent_dim', 100)}")
                
                if current_model == "Simple GAN":
                    st.info("🏗️ **Architecture**: Lightweight CNN with Upsampling")
                else:
                    st.info("🏗️ **Architecture**: DCGAN with ConvTranspose2d")
                
            with col_info2:
                st.markdown("#### 🕵️ **Discriminator Network**") 
                st.success("✅ **Status**: Trained and Ready")
                st.info("🔍 **Purpose**: Real vs Fake Image Classification")
                
                if current_model == "Simple GAN":
                    st.info("🏗️ **Architecture**: Convolutional Classifier")
                else:
                    st.info("🏗️ **Architecture**: WGAN-GP Critic")
                    st.info("⚖️ **Loss**: Wasserstein + Gradient Penalty")
                
        else:
            st.info("📊 **Model information will appear here after training**")
            
        st.markdown("#### 🔧 **Technical Specifications**")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            st.metric("🖥️ **Framework**", "PyTorch")
        with tech_col2:  
            st.metric("💾 **Memory Optimized**", "4GB")
        with tech_col3:
            current_arch = st.session_state.get('model_type', 'Not Selected')
            st.metric("⚡ **Architecture**", current_arch)
        
        # Model comparison table
        st.markdown("#### 📈 **Model Comparison**")
        comparison_data = {
            "Feature": ["Training Speed", "Image Quality", "Memory Usage", "Stability", "Data Augmentation"],
            "Simple GAN": ["⚡ Fast", "🟡 Good", "💚 Low", "🟡 Moderate", "❌ None"],
            "DCGAN-WGAN": ["🐌 Slower", "💚 Excellent", "🟡 Higher", "💚 Very Stable", "✅ Heavy"]
        }
        st.table(comparison_data)
    
    # Bottom section with cleanup
    st.markdown("---")
    col_clean1, col_clean2, col_clean3 = st.columns([2, 1, 1])
    
    with col_clean2:
        if st.button("🧹 **Clear Memory**", help="Clear GPU/CPU memory and reset app state", use_container_width=True):
            # Clear specific keys
            for key in ['generator', 'discriminator', 'trained', 'latent_dim', 'model_type']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            st.success("🧹 Memory cleared and state reset!")
    
    with col_clean3:
        if st.button("ℹ️ **About**", use_container_width=True):
            st.info("""
            🎨 **AI Art Generator Studio**
            
            A comprehensive GAN training application featuring:
            • 💾 4GB memory optimization
            • 🚀 Multiple model architectures
            • 🎯 High-quality results
            • 🖼️ Custom image datasets
            • 📊 Advanced training techniques
            
            **Models Available:**
            • Simple GAN: Fast training, good for beginners
            • DCGAN-WGAN: Advanced architecture with better quality
            """)

if __name__ == "__main__":
    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False
    
    main()
