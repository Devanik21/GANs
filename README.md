# GAN Studio: A Deep Learning Framework for Generative Adversarial Networks

![GAN Studio Preview](Aesthetic.png)

Welcome to **GAN Studio**, a sophisticated, high-performance framework designed for the exploration, training, and deployment of Generative Adversarial Networks (GANs). This repository provides a robust environment for synthesizing high-fidelity imagery using state-of-the-art architectures, specifically tailored for stability and efficiency in constrained hardware environments.

## Overview

GAN Studio bridges the gap between theoretical deep learning and practical implementation. It features a Streamlit-based interface that allows for real-time monitoring of the adversarial game, hyperparameter tuning, and latent space exploration. The framework supports both traditional Multi-Layer Perceptron (MLP) based GANs and Deep Convolutional GANs (DCGAN) enhanced with Wasserstein Loss and Gradient Penalty (WGAN-GP).

---

## Technical Architecture & Deep Dive

### 1. The Adversarial Framework
At its core, the system implements the fundamental GAN objective, where a Generator ($G$) and a Discriminator ($D$) engage in a zero-sum minimax game. The Generator attempts to map a latent noise vector $z \sim p_z(z)$ to the data distribution $p_{data}$, while the Discriminator attempts to distinguish between real and synthetic samples.

#### The Minimax Objective:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

### 2. Wasserstein GAN with Gradient Penalty (WGAN-GP)
To mitigate common failure modes such as mode collapse and vanishing gradients, this repository defaults to the **Wasserstein GAN** formulation. Unlike traditional GANs that minimize the Jensen-Shannon divergence, WGAN minimizes the **Earth Mover's (EM) distance** or Wasserstein-1 distance:

$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x-y\|]$$

Through the Kantorovich-Rubinstein duality, the objective for the Critic ($D$) becomes:
$$\max_{D \in \|D\|_L \leq 1} \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{x \sim P_g}[D(x)]$$

#### Gradient Penalty (GP):
To enforce the 1-Lipschitz continuity constraint on the Critic without the drawbacks of weight clipping, we implement a Gradient Penalty term:
$$L_{GP} = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$
where $\hat{x}$ is sampled uniformly along straight lines between pairs of points from the real and generated distributions.

### 3. Network Architectures
*   **Simple GAN (MLP):** A baseline architecture utilizing fully connected layers with LeakyReLU activations and Batch Normalization. Ideal for lower-dimensional manifolds and rapid prototyping.
*   **DCGAN (Deep Convolutional):** Employs transposed convolutions for upsampling in the Generator and strided convolutions in the Discriminator. This architecture leverages spatial hierarchies in image data for superior synthesis quality.

---

## Hardware Optimization & Persistence

GAN Studio is meticulously optimized for consumer-grade GPUs, specifically targeting environments with **4GB of VRAM**. This is achieved through:
-   Efficient memory buffer management during the backpropagation of the Gradient Penalty.
-   Adaptive batch sizing relative to image resolution (32px, 64px, 128px).

### Data Management
The framework utilizes **TinyDB** for a lightweight, document-oriented persistence layer. This ensures that:
-   Hyperparameter configurations are versioned.
-   Training progress and loss metrics are logged for post-hoc analysis.
-   Model checkpoints are indexed for seamless resumption of training.

---

## Installation & Setup

Ensure you have a Python 3.8+ environment. It is highly recommended to use a virtual environment or Conda.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/gan-studio.git
   cd gan-studio
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Studio:**
   ```bash
   streamlit run app.py
   ```

---

## Usage Guide

1.  **Configuration:** Select your desired architecture (Simple or DCGAN) and resolution via the sidebar.
2.  **Hyperparameter Tuning:** Adjust the learning rate ($\alpha$), coefficients for the Adam optimizer ($\beta_1, \beta_2$), and the Gradient Penalty coefficient ($\lambda$).
3.  **Training:** Initiate the training loop and observe the Critic/Generator loss curves in real-time.
4.  **Inference:** Once trained, use the "Inference" mode to sample from the latent space and generate novel imagery.

---

## Mathematical Notation Reference
-   $G(z)$: Generator function mapping noise $z$ to data space.
-   $D(x)$: Critic/Discriminator score for input $x$.
-   $\mathbb{E}$: Expected value.
-   $\nabla$: Gradient operator.
-   $\|\cdot\|_2$: $L_2$ norm.

Thank you for exploring GAN Studio. We hope this tool empowers your research and creative endeavors in the field of generative modeling. For any technical inquiries or contributions, please feel free to open an issue or pull request.
