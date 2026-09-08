"""
Training entry point for the change model.
Owner: Person 2 (single-image) or Person 3 (multi/cross-image), depending on task.

Usage:
    python -m src.training.train_change --config configs/config.yaml
"""

import argparse
import yaml

from src.models.change_model import ChangeModel, train_step
from src.preprocessing.dataset_loader import get_dataloader
from src.training.trainer_utils import (
    set_seed, get_device, build_optimizer, save_checkpoint, SimpleLogger,
)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    train_loader = get_dataloader(task="change", split="train", config=config,
                                   batch_size=config["training"]["batch_size"])
    val_loader = get_dataloader(task="change", split="val", config=config,
                                 batch_size=config["training"]["batch_size"])

    model = ChangeModel(config=config)
    # For fine-tuning from a pretrained backbone, load it here before training:
    # model.load(config["models"]["change"]["checkpoint"], device=device)
    optimizer = build_optimizer(model, config)
    logger = SimpleLogger(log_file=f"models/checkpoints/change_train.log")

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

        save_checkpoint(model, config["models"]["change"]["checkpoint"], epoch, optimizer)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
