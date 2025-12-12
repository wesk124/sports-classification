"""Simple demo script for running inference with a pre-trained ResNet-50.

Run from repo root:

    python demo/demo.py
"""

import json
import os

import torch

from src import config, model as model_module, utils


def run_demo() -> None:
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Ensure dataset exists
    data_root = utils.download_and_prepare_dataset(
        dataset_handle=config.DATASET_HANDLE,
        extraction_dir=config.DATA_ROOT,
    )

    _, _, test_loader, class_names = utils.create_dataloaders(
        data_root=data_root,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    num_classes = len(class_names)

    # Build model architecture (weights loaded from checkpoint)
    model = model_module.create_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model = model.to(device)

    # Load checkpoint
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint file not found at {config.BEST_MODEL_PATH}. "
            "Download the pre-trained model and place it there, or train from scratch."
        )

    print(f"Loading checkpoint from {config.BEST_MODEL_PATH}...")
    state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded and set to eval mode.")

    # Take a single mini-batch from the test loader
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

    demo_results = []
    for idx in range(min(len(labels), 8)):
        true_idx = labels[idx].item()
        pred_idx = preds[idx].item()

        demo_results.append(
            {
                "index": idx,
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "confidence": float(confs[idx].item()),
            }
        )
        print(
            f"Sample {idx}: true = '{class_names[true_idx]}', "
            f"predicted = '{class_names[pred_idx]}', "
            f"confidence = {confs[idx].item():.4f}"
        )

    # Save predictions
    os.makedirs(os.path.dirname(config.DEMO_PREDICTIONS_FILE), exist_ok=True)
    with open(config.DEMO_PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(demo_results, f, indent=2)

    print(f"Demo predictions saved to {config.DEMO_PREDICTIONS_FILE}")


if __name__ == "__main__":
    run_demo()
