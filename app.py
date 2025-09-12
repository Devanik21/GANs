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

# Set page config
st.set_page_config(page_title="Simple GAN Training", layout="wide")

# Device configuration for memory efficiency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        try:
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
            status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {file.name}")
            
            image = Image.open(file)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic validation - check if image is too small
            if image.size[0] < 16 or image.size[1] < 16:
                st.warning(f"Skipping {file.name}: image too small (minimum 16x16)")
                continue
                
            images.append(image)
            valid_files += 1
            
        except Exception as e:
            st.warning(f"Could not process file {file.name}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return images, valid_files

def train_gan(images, epochs=50, batch_size=4, lr=0.0002, latent_dim=100, img_size=64):
    """Train the GAN with memory-efficient approach"""
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(images, transform=transform, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize networks
    generator = SimpleGenerator(latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = SimpleDiscriminator(img_size=img_size).to(device)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Training progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
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
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        # Record losses
        avg_d_loss = epoch_d_loss / batches
        avg_g_loss = epoch_g_loss / batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Update loss chart every 5 epochs
        if (epoch + 1) % 5 == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
            ax.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            loss_chart.pyplot(fig)
            plt.close(fig)
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_z = torch.randn(4, latent_dim, device=device)
                sample_images = generator(sample_z)
                sample_images = (sample_images + 1) / 2  # Denormalize
                
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                for j in range(4):
                    img = sample_images[j].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    axes[j].imshow(img)
                    axes[j].axis('off')
                    axes[j].set_title(f'Generated {j+1}')
                
                plt.suptitle(f'Generated Images - Epoch {epoch+1}')
                plt.tight_layout()
                sample_container.pyplot(fig)
                plt.close(fig)
                
                del sample_z, sample_images
    
    return generator, discriminator, d_losses, g_losses

def generate_images(generator, num_images=8, latent_dim=100):
    """Generate new images using trained generator"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        generated_images = generator(z)
        generated_images = (generated_images + 1) / 2  # Denormalize
        
        # Create grid
        rows = 2
        cols = num_images // rows
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(num_images):
            img = generated_images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Generated {i+1}')
        
        plt.suptitle('Generated Images')
        plt.tight_layout()
        return fig

# Streamlit UI
def main():
    st.title("🎨 Simple GAN Training App")
    st.markdown("Train a Generative Adversarial Network on your images (optimized for 4GB memory)")
    
    # Sidebar for parameters
    st.sidebar.header("Training Parameters")
    img_size = st.sidebar.selectbox("Image Size", [32, 64, 128], index=1)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch Size", 1, 8, 4)
    lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.0002, format="%.4f")
    latent_dim = st.sidebar.slider("Latent Dimension", 50, 200, 100)
    
    # Memory warning
    if img_size > 64 or batch_size > 4:
        st.sidebar.warning("⚠️ High settings may exceed 4GB memory limit")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Training Images")
        uploaded_files = st.file_uploader(
            "Choose images (1-100)", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True,
            help="Upload 1-100 images for training"
        )
        
        if uploaded_files:
            num_files = len(uploaded_files)
            if num_files > 100:
                st.error("Please upload maximum 100 images")
                return
            
            st.success(f"✅ {num_files} images uploaded")
            
            # Show sample images
            if st.checkbox("Preview uploaded images"):
                cols = st.columns(min(4, num_files))
                for i, file in enumerate(uploaded_files[:4]):
                    try:
                        img = Image.open(file)
                        cols[i].image(img, caption=f"Image {i+1}", use_column_width=True)
                    except Exception as e:
                        cols[i].error(f"Error loading {file.name}")
    
    with col2:
        st.header("🚀 Training")
        
        if uploaded_files and st.button("Start Training", type="primary"):
            if len(uploaded_files) < 1:
                st.error("Please upload at least 1 image")
                return
            
            with st.spinner("Preprocessing images..."):
                images, valid_files = preprocess_images(uploaded_files, img_size)
            
            if valid_files == 0:
                st.error("No valid images found")
                return
            
            st.success(f"Processing {valid_files} valid images")
            
            # Memory estimate
            estimated_memory = (valid_files * img_size * img_size * 3 * 4) / (1024**3)  # GB
            st.info(f"Estimated memory usage: {estimated_memory:.2f} GB")
            
            if estimated_memory > 3:
                st.warning("High memory usage detected. Consider reducing image size or batch size.")
            
            # Train the GAN
            try:
                st.header("📊 Training Progress")
                generator, discriminator, d_losses, g_losses = train_gan(
                    images, epochs, batch_size, lr, latent_dim, img_size
                )
                
                st.success("🎉 Training completed!")
                
                # Store models in session state
                st.session_state['generator'] = generator
                st.session_state['discriminator'] = discriminator
                st.session_state['trained'] = True
                st.session_state['latent_dim'] = latent_dim
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    st.error("❌ Out of memory! Try reducing batch size or image size.")
                else:
                    st.error(f"Training error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    
    # Generation section
    if st.session_state.get('trained', False):
        st.header("🎨 Generate New Images")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_generate = st.slider("Number of images to generate", 1, 16, 8)
            
            if st.button("Generate Images", type="secondary"):
                with st.spinner("Generating images..."):
                    try:
                        fig = generate_images(
                            st.session_state['generator'], 
                            num_generate, 
                            st.session_state['latent_dim']
                        )
                        with col2:
                            st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Generation error: {e}")
    
    # Clear memory button
    if st.button("🧹 Clear Memory", help="Clear GPU/CPU memory"):
        if 'generator' in st.session_state:
            del st.session_state['generator']
        if 'discriminator' in st.session_state:
            del st.session_state['discriminator']
        st.session_state['trained'] = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        st.success("Memory cleared!")

if __name__ == "__main__":
    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False
    
    main()
