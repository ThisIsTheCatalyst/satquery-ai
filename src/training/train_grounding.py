"""
Training entry point for the grounding model (alternative to captioning).
Owner: Person 2 (Single-Image Models) — use this instead of train_captioning.py
if the team picks text-guided region grounding as the second single-image task.

Usage:
    python -m src.training.train_grounding --config configs/config.yaml
"""

import argparse
import yaml

from src.models.grounding_model import GroundingModel, train_step
from src.preprocessing.dataset_loader import get_dataloader
from src.training.trainer_utils import (
    set_seed, get_device, build_optimizer, save_checkpoint, SimpleLogger,
)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    train_loader = get_dataloader(task="grounding", split="train", config=config,
                                   batch_size=config["training"]["batch_size"])
    val_loader = get_dataloader(task="grounding", split="val", config=config,
                                 batch_size=config["training"]["batch_size"])

    model = GroundingModel(config=config)
    optimizer = build_optimizer(model, config)
    logger = SimpleLogger(log_file="models/checkpoints/grounding_train.log")

    step = 0
    for epoch in range(config["training"]["epochs"]):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer, config)
            if step % config["training"]["log_every"] == 0:
                logger.log(step, {"loss": loss})
            step += 1
        # TODO: validation loop + IoU metric via src/evaluation/metrics.py
        save_checkpoint(model, config["models"]["grounding"]["checkpoint"], epoch, optimizer)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
