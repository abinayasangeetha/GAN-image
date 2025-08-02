# GAN Image Generator - MNIST

---

## Creating Realistic Images with GANs: A Deep Learning Challenge

This project implements a **Generative Adversarial Network (GAN)** using **PyTorch** to generate realistic-looking handwritten digits similar to those in the **MNIST dataset**.

A GAN consists of two neural networks — the **Generator** and the **Discriminator** — that compete in a zero-sum game. The Generator tries to create convincing fake images, while the Discriminator attempts to distinguish fake images from real ones. Over time, the Generator learns to produce highly realistic images from random noise.

---

##  Project Structure

```
GAN-MNIST/
│
├── gan_mnist.py                 # Full training script for the GAN
├── generated_images/           # Folder to store generated images every 10 epochs
├── data/                       # Auto-downloaded MNIST dataset
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

##  Dataset

We use the **MNIST dataset**, a collection of 70,000 grayscale handwritten digit images (0–9), each of size 28x28. The dataset is loaded using `torchvision.datasets.MNIST` with normalization to [-1, 1] to stabilize training.

---

## Model Architecture

### Generator

- **Input**: Random noise vector `z` of size 100
- **Layers**: Fully connected layers with `LeakyReLU` and `BatchNorm1d`
- **Output**: 28x28 image using `Tanh` activation

### Discriminator

- **Input**: Flattened 28x28 image (784 values)
- **Layers**: Fully connected layers with `LeakyReLU`
- **Output**: Single sigmoid value indicating real or fake

---

## Training Configuration

- **Epochs**: 50
- **Batch size**: 64
- **Optimizer**: Adam (`lr=0.0002`, `betas=(0.5, 0.999)`)
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Image Saving**: Generates and saves 64 samples every 10 epochs

---

## How to Run

1. Ensure PyTorch and required libraries are installed (see below).
2. Run the training script:

```bash
python gan_mnist.py
```

3. Generated images will be saved to the `generated_images/` folder every 10 epochs.

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision matplotlib
```

---

##  Results

- GAN progressively improves image quality with each epoch.
- Early results are noisy, but later epochs produce sharp digit-like images.
- Image samples are saved during training and can be found in the `generated_images/` directory.

Example output:

```
generated_images/
├── epoch_1.png
├── epoch_10.png
├── epoch_20.png
├── ...
└── epoch_50.png
```

---
## Output Images:
#### epoch_0
<img width="1000" height="200" alt="epoch_0 (1)" src="https://github.com/user-attachments/assets/aec8634f-99c6-4b2f-afb8-ee9204d4fcb2" />
#### epoch_25
<img width="1000" height="200" alt="epoch_25" src="https://github.com/user-attachments/assets/77123825-6959-4d6c-a082-951c013a6853" />
#### epoch_49
<img width="1000" height="200" alt="epoch_49" src="https://github.com/user-attachments/assets/49a044b8-fa5b-481e-8871-ade6c47a9ce7" />



## Outputs:

Each saved image file contains a grid (8x8) of samples from the Generator at different training epochs. These help track how the quality of generated digits improves over time.

---


## Acknowledgments

- PyTorch documentation
- Deep learning community tutorials on GANs
