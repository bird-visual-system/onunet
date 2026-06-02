
# onunet

A U-Net based deep learning pipeline for semantic segmentation of optical nerve images. The purpose is to detect and mark axons constituting the optical nerve. The model is trained on grayscale microscopy images and produces pixel-level binary segmentation masks, using pixel-weighted binary cross-entropy to handle boundary detection for seprarating adjacent axons.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Organization](#data-organization)
- [Usage](#usage)
  - [Training with Docker (recommended)](#training-with-docker-recommended)
  - [Training locally](#training-locally)
  - [Prediction and evaluation](#prediction-and-evaluation)
- [Hyperparameters](#hyperparameters)
- [Training history](#training-history)
- [License](#license)

---

## Overview

**onunet** segments optical nerve structures from grayscale images using a fully-convolutional encoder-decoder network (U-Net). Key features:

- Pixel-weighted loss that down- or up-weights individual pixels during training, enabling better boundary detection between adjacent axons and, optionaly, handling  class imbalance between foreground and background.
- Data augmentation (rotation, shifts, shear, zoom, horizontal/vertical flips) applied consistently to images, masks, and weight maps.
- Configurable hyperparameters via environment variables for easy use in containerized environments.

---

## Architecture

Three model variants are defined in `model.py`:

| Variant | Description |
|---|---|
| `weighted_loss_unet` | Full-depth U-Net (64→1024 filters) with pixel-weighted binary cross-entropy. **Used for training.** |
| `unet` | Standard U-Net with plain binary cross-entropy loss. |
| `unet_minimal` | Lightweight U-Net (8→32 filters) for quick experiments. |

All variants share the same encoder-decoder structure with skip connections. The `weighted_loss_unet` accepts three inputs during training (image, mask, weight map) and a single input during inference.

**Loss:** pixel-weighted binary cross-entropy  
**Optimizer:** Adam (lr = 1e-4)  
**Metrics:** Binary accuracy, Mean IoU  
**Input size:** 512 × 512 × 1 (grayscale)

---

## Project Structure

```
onunet/
├── main.py               # Training entry point
├── model.py              # Model definitions and custom losses
├── data.py               # Data generators with augmentation
├── Dockerfile            # Container definition (TensorFlow 2.13 GPU)
├── Buildrun.bash         # Helper script to run the Docker container
├── requirements.txt      # Additional Python dependencies
├── trainUnet.ipynb       # Interactive training notebook
├── testOnunet.ipynb      # Evaluation and testing notebook
├── predictOnunet.ipynb   # Inference / prediction notebook
├── data/                 # Training and validation data (see below)
├── figures/              # Training history plots
└── old/                  # Archived experiments
```

---

## Requirements

Dependencies beyond the base TensorFlow image:

```
openpyxl==3.0.7
read-roi==1.6.0
Shapely==1.8a3
scikit-image==0.21.0
```

The Docker image (`tensorflow/tensorflow:2.13.0-gpu`) already includes TensorFlow, NumPy, and Keras. Install additional dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data Organization

The data directory must follow this structure:

```
data/optical_nerve/
├── train/
│   ├── images/     # Grayscale input images (.png / .tif)
│   ├── labels/     # Binary segmentation masks
│   └── weights/    # Per-pixel weight maps
└── validation/
    ├── images/
    ├── labels/
    └── weights/
```

Images and masks are normalised to [0, 1]. Masks are binarised at a threshold of 0.5. Weight maps are rescaled by a factor of `WEIGHTS_RESCALE_FACTOR` (default: 3.64749).

---

## Usage

### Training with Docker (recommended)

Pull or build the image:

```bash
docker build -t registry.gitlab.com/ptitmatheux/onunet .
```

Run training (mounts your home directory and sets hyperparameters):

```bash
docker run -it \
  -e BATCH_SIZE=4 \
  -e EPOCHS=50 \
  -e STEPS_PER_EPOCH=100 \
  -e VALIDATION_STEPS=20 \
  -v /home/$USER:/home/jovyan \
  registry.gitlab.com/ptitmatheux/onunet
```

See `Buildrun.bash` for machine-specific run commands.

### Training locally

Set the required environment variables, then run:

```bash
export BATCH_SIZE=4
export EPOCHS=50
export STEPS_PER_EPOCH=100
export VALIDATION_STEPS=20
python main.py
```

Trained weights are saved to `model_weights/optical_nerve_weights.hdf5` (best validation loss only). The training history CSV is written alongside it.

### Prediction and evaluation

Use the provided notebooks:

- **`predictOnunet.ipynb`** — load a trained model and run inference on new images.
- **`testOnunet.ipynb`** — evaluate model performance on a test set.
- **`trainUnet.ipynb`** — interactive training with live plots.

---

## Hyperparameters

Hyperparameters are passed as environment variables and logged to `optical_nerve_hyperparameters.txt` at the start of each run.

| Variable | Description |
|---|---|
| `BATCH_SIZE` | Number of samples per training step |
| `EPOCHS` | Total number of training epochs |
| `STEPS_PER_EPOCH` | Number of batches per epoch |
| `VALIDATION_STEPS` | Number of batches used for validation |

---

## Training history

Example training curves (50 epochs, 9 training examples):

![Training history](figures/optical_nerve_history_from_scratch_9examples.png)

---

## License

See [LICENSE.txt](LICENSE.txt).

