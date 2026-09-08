"""
Training entry point for the captioning model.
Owner: Person 2 (single-image) or Person 3 (multi/cross-image), depending on task.

Usage:
    python -m src.training.train_captioning --config configs/config.yaml
"""

import argparse
import yaml

from src.models.captioning_model import CaptioningModel, train_step
from src.preprocessing.dataset_loader import get_dataloader
from src.training.trainer_utils import (
    set_seed, get_device, build_optimizer, save_checkpoint, SimpleLogger,
)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    train_loader = get_dataloader(task="captioning", split="train", config=config,
                                   batch_size=config["training"]["batch_size"])
    val_loader = get_dataloader(task="captioning", split="val", config=config,
                                 batch_size=config["training"]["batch_size"])

    model = CaptioningModel(config=config)
    # For fine-tuning from a pretrained backbone, load it here before training:
    # model.load(config["models"]["caption"]["checkpoint"], device=device)
    optimizer = build_optimizer(model, config)
    logger = SimpleLogger(log_file=f"models/checkpoints/caption_train.log")

    step = 0
    for epoch in range(config["training"]["epochs"]):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, config)
            if step % config["training"]["log_every"] == 0:
                logger.log(step, {"loss": loss})
            step += 1

        if epoch % config["training"]["eval_every"] == 0:
            # TODO: run validation loop, compute task metric via
            # src/evaluation/metrics.py, log it.
            pass

        save_checkpoint(model, config["models"]["caption"]["checkpoint"], epoch, optimizer)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
