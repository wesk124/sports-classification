"""Entry point for training and evaluation.

Usage:

  # Train from scratch (will download dataset via kagglehub)
  python -m src.main --mode train

  # Evaluate an existing checkpoint on the test set
  python -m src.main --mode eval
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src import config, model as model_module, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sports Classification Training/Eval")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Whether to train a model or evaluate a saved checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Basic setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # Ensure results/checkpoints directories exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Download dataset (if needed) and create dataloaders
    data_root = utils.download_and_prepare_dataset(
        dataset_handle=config.DATASET_HANDLE,
        extraction_dir=config.DATA_ROOT,
    )

    train_loader, val_loader, test_loader, class_names = utils.create_dataloaders(
        data_root=data_root,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    num_classes = len(class_names)

    # Build model
    model = model_module.create_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
    )
    model = model.to(device)

    if args.mode == "train":
        print("Starting training mode...")
        criterion = nn.CrossEntropyLoss().to(device)

        # Only train the final FC layer
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=config.LEARNING_RATE)

        # Train
        model, history = utils.train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloaders={"train": train_loader, "val": val_loader},
            device=device,
            num_epochs=config.NUM_EPOCHS,
            checkpoint_path=config.BEST_MODEL_PATH,
        )

        utils.save_history(history, config.TRAINING_HISTORY_FILE)

        # Evaluate best model on test set
        print("Evaluating best model on the test set...")
        metrics = utils.evaluate_model(model, test_loader, device)
        print("Test metrics:", metrics)

    else:  # eval mode
        print(f"Evaluation mode. Loading checkpoint from {config.BEST_MODEL_PATH}")
        if not os.path.exists(config.BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found at {config.BEST_MODEL_PATH}. "
                "Download it or train a model first."
            )

        utils.load_checkpoint(model, config.BEST_MODEL_PATH, device)
        metrics = utils.evaluate_model(model, test_loader, device)
        print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
