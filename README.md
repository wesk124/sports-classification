# Sports Image Classification

This project implements a deep learning pipeline for **multi-class sports image classification** using fine-tuned ResNet family models from `torchvision`.
You can copy the ipynb file from /demo and run the demo directly in Google Colab (recommended), or use the Python script provideded in the
repo to replicate this project.


---

## Project Overview

- **Task**: Classify images into 100 sports categories (e.g., tennis, basketball, golf, etc.).
- **Dataset**: [Sports Classification (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/sports-classification).
- **Model**: Pre-trained **ResNet-50** (ImageNet weights) with:
  - Backbone layers frozen
  - Final fully connected (FC) layer replaced to output 100 classes
- **Training**:
  - Loss: CrossEntropyLoss
  - Optimizer: Adam (on the new FC layer only)
  - Data augmentation: random resized crop, horizontal flip
  - Normalization: ImageNet mean/std

---

## Repository Structure

```text
├── README.md
├── requirements.txt
├── src/
│   ├── main.py        # Training & evaluation entry point
│   ├── utils.py       # Dataloaders, training loop, evaluation
│   ├── model.py       # ResNet model factory
│   ├── config.py      # Hyperparameters & paths
├── data/              # Downloaded dataset (Kaggle, auto-handled)
├── checkpoints/       # Saved model weights (.pth)
├── demo/
│   └── demo.py        # Simple inference demo on test samples
└── results/           # Training curves, demo predictions, etc.
```

---

## Setup Instructions

### 1. Clone & create environment

```bash
git clone https://github.com/wesk124/sports-classification.git
cd sports-classification

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---


## How to Run the Demo

1. Download the pre-trained model and place it in `checkpoints/` (see next section).
2. Run:

```bash
python demo/demo.py
```

The demo will:

- Load the pre-trained ResNet-<x> model
- Run inference on a batch of test images
- Print a few predictions
- Save predictions to `results/demo_predictions.json`

Example console output:

```text
Using device: cuda
Loading checkpoint from checkpoints/best_model_resnet50.pth...
Sample 0: true = 'basketball', predicted = 'basketball', confidence = 0.98
...
Demo predictions saved to results/demo_predictions.json
```

---

## How to Train from Scratch

```bash
python -m src.main --mode train
```

This will:

1. Download and extract the Kaggle sports dataset (if needed)
2. Build a ResNet-50 model with a new FC layer for `num_classes`
3. Train for `NUM_EPOCHS` (see `config.py`)
4. Save the best model (by validation accuracy) to:

   ```text
   checkpoints/best_model_resnet50.pth
   ```

To evaluate a trained model on the test set:

```bash
python -m src.main --mode eval
```

---

## Expected Output

During training, you’ll see per-epoch logs like:

```text
Epoch 1/15
----------
Train Loss: 1.2345 Acc: 0.6500
Val   Loss: 1.1000 Acc: 0.6800
New best model saved to checkpoints/best_model_resnet50.pth (Acc: 0.6800)
...
Training complete. Best val accuracy: 0.8000
```

After training, evaluation on the test set prints metrics:

```text
Evaluation metrics:
Accuracy: 0.78
Precision: 0.79
Recall: 0.78
F1: 0.78
```

The training history (loss/accuracy curves) is saved to:

```text
results/training_history_resnet50.json
```

The demo script outputs predictions to:

```text
results/demo_predictions.json
```

---

## Hyperparameters and Configuration

All key configuration (paths, hyperparameters) is centralized in `src/config.py`. For example:

- `MODEL_NAME = "resnet50"`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 15`
- `LEARNING_RATE = 1e-3`
- `BEST_MODEL_PATH = "checkpoints/best_model_resnet<x>.pth"`

You can tweak these values in `config.py` and re-run training.

---

## Acknowledgments

- **Dataset**: [Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification) by *gpiosenka* on Kaggle.
- **Pretrained Models**: [Torchvision ResNet models](https://pytorch.org/vision/stable/models.html).
- **Frameworks**: [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/), [scikit-learn](https://scikit-learn.org/).
