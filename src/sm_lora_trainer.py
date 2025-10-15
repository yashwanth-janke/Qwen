import os
import torch
import torch.distributed as dist
import yaml
import logging
import argparse
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get rank information
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# Setup logging
logging.basicConfig(level=logging.INFO if global_rank == 0 else logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Qwen3 model training")
    parser.add_argument("--config", type=str, default="/opt/ml/input/data/config/qwen3-4b.yaml")
    args = parser.parse_args()
    
    # Initialize distributed training
    is_distributed = world_size > 1
    is_main_process = global_rank == 0
    
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    # Load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Setup paths
    model_path = config.get("model_name_or_path", "/opt/ml/input/data/model_weight")
    train_path = os.path.join(
        config.get("train_dataset_path", "/opt/ml/input/data/training"),
        config.get("data", {}).get("train_path", "train_dataset.json")
    )
    output_dir = config.get("output_dir", "/opt/ml/checkpoints")
    
    # Create directories
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    
    # Set seed
    training_config = config.get("training", {})
    set_seed(training_config.get("seed", 42))
    
    # Precision settings
    use_bf16 = config.get("model", {}).get("use_bf16", True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        device_map=None
    )
    model = model.to("cuda")
    model.config.use_cache = False
    
    # Load and preprocess dataset
    raw_dataset = load_dataset("json", data_files=train_path, split="train")
    
    def preprocess_function(examples):
        text_field = "text" if "text" in examples else list(examples.keys())[0]
        texts = examples[text_field]
        if isinstance(texts, list):
            texts = [str(item) if not isinstance(item, str) else item for item in texts]
        
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.get("data", {}).get("max_seq_length", 2048),
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=1,
    )
    
    # LoRA configuration - more explicit setup
    lora_config = config.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("lora_r", 64),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias="none",
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]),
    )
    
    # Apply LoRA
    if is_main_process:
        logger.info(f"LoRA configuration: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
        logger.info(f"LoRA target modules: {peft_config.target_modules}")
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Print trainable parameters info
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        learning_rate=training_config.get("learning_rate", 2e-4),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        logging_steps=training_config.get("logging_steps", 10),
        warmup_steps=training_config.get("warmup_steps", 10),
        bf16=use_bf16,
        fp16=not use_bf16,
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        seed=training_config.get("seed", 42),
        dataloader_num_workers=0,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )
    
    try:
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Execute training
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            peft_config=None,  # Use already applied model
        )
        
        # Start training
        if is_main_process:
            logger.info("Starting training...")
        
        trainer.train()
        
        # Save model
        if is_main_process:
            logger.info("Saving model...")
            # Save only LoRA adapter
            model.save_pretrained(output_dir)
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
    
    finally:
        # Cleanup distributed training
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()