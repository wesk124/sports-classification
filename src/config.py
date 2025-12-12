"""Configuration file for the Sports Classification project."""

import os

# --------------------
# Dataset configuration
# --------------------

# Kaggle dataset handle used with kagglehub
DATASET_HANDLE = "gpiosenka/sports-classification"

# Local directory where the dataset will be extracted/copied
DATA_ROOT = os.path.join("data", "sports_classification_dataset")

# Subfolder names (ImageFolder-compatible)
TRAIN_DIR = "train"
VAL_DIR = "valid"
TEST_DIR = "test"

# --------------------
# Training configuration
# --------------------

MODEL_NAME = "resnet50"  # Using ResNet-50
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42

# --------------------
# Checkpoints / Results
# --------------------

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_FILENAME = "best_model_resnet50.pth"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_FILENAME)

RESULTS_DIR = "results"
TRAINING_HISTORY_FILE = os.path.join(RESULTS_DIR, "training_history_resnet50.json")
DEMO_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "demo_predictions.json")

# --------------------
# Image normalization (ImageNet stats)
# --------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
