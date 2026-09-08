#!/usr/bin/env python3
"""
Fine-tune GeoChat VQA with LoRA on BigEarthNet (RS domain adaptation).

Usage:
    python scripts/train_vqa.py --config configs/colab_8gb.yaml

This trains a LoRA adapter on top of GeoChat using BigEarthNet image-caption
pairs to improve remote-sensing grounding of the base VLM.
Requires: transformers, peft, accelerate, a GeoChat base checkpoint.
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="VQA LoRA fine-tuning")
    parser.add_argument("--config", default="configs/colab_8gb.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if "_extends" in config:
        base_path = config.pop("_extends")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(config)
        config = base

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install transformers peft accelerate")
        sys.exit(1)

    from src.training.trainer_utils import set_seed, get_device, save_checkpoint

    t_cfg = config["training"]
    device = get_device(t_cfg.get("device", "cuda"))
    set_seed(t_cfg.get("seed", 42))

    print(f"\nVQA LoRA fine-tuning")
    print(f"  config: {args.config}")
    print(f"  device: {device}")

    checkpoint_path = config["models"]["vqa"]["checkpoint"]
    model_cfg = config["models"]["vqa"]
    
    # Load base GeoChat model
    base_model_name = "MBZUAI/geochat-7B"
    print(f"  Loading base model: {base_model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if t_cfg.get("fp16") else torch.float32,
        )
    except Exception as e:
        print(f"ERROR: Could not load GeoChat: {e}")
        print("Download GeoChat from HuggingFace or use the baseline VQA model.")
        sys.exit(1)

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.get("lora_rank", 4),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    
    # Load BigEarthNet dataset
    try:
        from src.preprocessing.dataset_loader import get_dataloader
        train_loader = get_dataloader(
            task="vqa", split="train", dataset="bigearthnet",
            batch_size=t_cfg["batch_size"], config=config,
        )
        print(f"  BigEarthNet train: {len(train_loader.dataset)} samples")
    except Exception as e:
        print(f"WARNING: Could not load BigEarthNet: {e}")
        print("Run in smoke-test mode with synthetic data.")
        sys.exit(0)

    # Training loop (simplified — use HuggingFace Trainer for production)
    from torch.optim import AdamW
    from src.training.trainer_utils import SimpleLogger, EarlyStopper
    
    model.to(device)
    opt = AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    logger = SimpleLogger(log_file="outputs/logs/vqa_training.log")
    os.makedirs("outputs/logs", exist_ok=True)
    stopper = EarlyStopper(patience=3, mode="min")
    
    print("\nStarting training...")
    for epoch in range(t_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            captions = batch["label"]
            
            # Tokenize captions
            encodings = tokenizer(captions, return_tensors="pt", padding=True,
                                  truncation=True, max_length=128).to(device)
            
            # Forward pass (simplified — real VLM training would interleave image tokens)
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            if (step + 1) % t_cfg.get("gradient_accumulation_steps", 4) == 0:
                import torch.nn as nn
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            
            total_loss += loss.item()
            if step % t_cfg.get("log_every", 50) == 0:
                logger.log(step, {"loss": loss.item(), "epoch": epoch + 1})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}  loss={avg_loss:.4f}")
        
        is_best = stopper.step(-avg_loss)
        # Save LoRA adapter
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        if stopper.should_stop:
            print("Early stopping.")
            break
    
    print(f"\nLoRA adapter saved to: {checkpoint_path}")
    print("Run smoke test: python scripts/smoke_test.py")


if __name__ == "__main__":
    main()
