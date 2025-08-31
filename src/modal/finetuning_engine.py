# Fine-tuning Engine - Modal Function for GPT-OSS 20B fine-tuning with LoRA
import modal
import torch
import os
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import logging

# Core ML libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import bitsandbytes as bnb

# Unsloth for optimized training
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from .image_config import finetune_image

app = modal.App("gpt-oss-finetuning-engine")

# Volumes for model cache, datasets, and checkpoints
model_cache_vol = modal.Volume.from_name("model-cache", create_if_missing=True)
dataset_cache_vol = modal.Volume.from_name("dataset-cache", create_if_missing=True)
processed_cache_vol = modal.Volume.from_name("processed-data-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("checkpoint-storage", create_if_missing=True)

class TrainingProgressCallback:
    """Callback for tracking training progress and sending updates"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.start_time = time.time()
        self.step_times = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training metrics are logged"""
        if logs and self.progress_callback:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate ETA
            if state.global_step > 0:
                avg_step_time = elapsed_time / state.global_step
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = avg_step_time * remaining_steps
            else:
                eta_seconds = 0
            
            progress_data = {
                "step": state.global_step,
                "max_steps": state.max_steps,
                "progress_percentage": (state.global_step / state.max_steps) * 100,
                "elapsed_time": elapsed_time,
                "eta_seconds": eta_seconds,
                "logs": logs,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.progress_callback(progress_data)

@app.function(
    image=finetune_image,
    gpu="H100:1",  # Single H100 GPU
    volumes={
        "/root/.cache/huggingface": model_cache_vol,
        "/root/.cache/datasets": dataset_cache_vol,
        "/root/.cache/processed": processed_cache_vol,
        "/root/checkpoints": checkpoint_vol,
    },
    timeout=43200,  # 12 hours
    memory=32768,   # 32GB RAM
    cpu=8,          # 8 CPU cores
)
def finetune_gpt_oss_with_lora(
    dataset_path: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    lora_config: Dict[str, Any],
    job_id: str,
    dataset_index: int,
    progress_callback: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fine-tune GPT-OSS 20B model with LoRA on a specific dataset
    
    Args:
        dataset_path: Path to preprocessed dataset
        model_config: Model configuration
        training_config: Training parameters
        lora_config: LoRA configuration
        job_id: Unique job identifier
        dataset_index: Index of current dataset in queue
        progress_callback: URL for progress updates (if provided)
        
    Returns:
        Dictionary with training results
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting fine-tuning for job {job_id}, dataset {dataset_index}")
        
        # Set random seed for reproducibility
        set_seed(training_config.get('seed', 42))
        
        # Load model and tokenizer
        model_name = model_config.get('model_name', 'openai/gpt-oss-20b')
        logger.info(f"Loading model: {model_name}")
        
        if UNSLOTH_AVAILABLE and training_config.get('use_unsloth', True):
            logger.info("Using Unsloth for optimized training")
            model, tokenizer = load_model_with_unsloth(
                model_name=model_name,
                lora_config=lora_config,
                training_config=training_config
            )
        else:
            logger.info("Using standard transformers for training")
            model, tokenizer = load_model_standard(
                model_name=model_name,
                lora_config=lora_config,
                training_config=training_config
            )
        
        # Load preprocessed dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['validation']
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} validation samples")
        
        # Prepare datasets for training
        train_dataset = prepare_dataset_for_training(train_dataset, tokenizer, training_config)
        eval_dataset = prepare_dataset_for_training(eval_dataset, tokenizer, training_config)
        
        # Setup training arguments
        output_dir = f"/root/checkpoints/{job_id}_dataset_{dataset_index}"
        training_args = create_training_arguments(
            output_dir=output_dir,
            training_config=training_config,
            total_samples=len(train_dataset)
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create progress callback
        progress_cb = TrainingProgressCallback(
            progress_callback=lambda data: send_progress_update(progress_callback, data) if progress_callback else None
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[progress_cb]
        )
        
        # Start training
        logger.info("Starting training...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the LoRA adapter
        adapter_path = f"{output_dir}/lora_adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        # Evaluate the model
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        # Calculate final metrics
        final_metrics = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_results.get('eval_loss', 0),
            "perplexity": torch.exp(torch.tensor(eval_results.get('eval_loss', 0))).item(),
            "training_time_seconds": training_time,
            "total_steps": train_result.global_step,
        }
        
        # Generate sample outputs for quality check
        sample_outputs = generate_sample_outputs(model, tokenizer, eval_dataset, num_samples=3)
        
        result = {
            "job_id": job_id,
            "dataset_index": dataset_index,
            "status": "completed",
            "model_name": model_name,
            "adapter_path": adapter_path,
            "metrics": final_metrics,
            "sample_outputs": sample_outputs,
            "training_config": training_config,
            "lora_config": lora_config,
            "dataset_stats": {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset)
            },
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Save training results
        results_path = f"{output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Training completed successfully for dataset {dataset_index}")
        return result
        
    except Exception as e:
        error_msg = f"Error during fine-tuning: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "job_id": job_id,
            "dataset_index": dataset_index,
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def load_model_with_unsloth(model_name: str, lora_config: Dict, training_config: Dict):
    """Load model using Unsloth for optimized training"""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=training_config.get('max_seq_length', 2048),
        dtype=torch.float16,
        load_in_4bit=training_config.get('load_in_4bit', True),
        cache_dir="/root/.cache/huggingface"
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('rank', 16),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.1),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_config.get('seed', 42),
    )
    
    return model, tokenizer

def load_model_standard(model_name: str, lora_config: Dict, training_config: Dict):
    """Load model using standard transformers"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/root/.cache/huggingface"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=training_config.get('load_in_4bit', True),
        bnb_4bit_use_double_quant=training_config.get('bnb_4bit_use_double_quant', True),
        bnb_4bit_quant_type=training_config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=torch.float16,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir="/root/.cache/huggingface"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    peft_config = LoraConfig(
        r=lora_config.get('rank', 16),
        lora_alpha=lora_config.get('alpha', 32),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=lora_config.get('dropout', 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def prepare_dataset_for_training(dataset: Dataset, tokenizer, training_config: Dict) -> Dataset:
    """Prepare dataset for training by tokenizing"""
    
    max_length = training_config.get('max_seq_length', 2048)
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def create_training_arguments(output_dir: str, training_config: Dict, total_samples: int) -> TrainingArguments:
    """Create training arguments"""
    
    # Calculate steps if not provided
    max_steps = training_config.get('max_steps_per_dataset', 1000)
    batch_size = training_config.get('batch_size', 4)
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 8)
    
    # Estimate epochs
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, total_samples // effective_batch_size)
    num_train_epochs = max(1, max_steps // steps_per_epoch)
    
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=training_config.get('learning_rate', 2e-4),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        warmup_steps=training_config.get('warmup_steps', 100),
        logging_steps=training_config.get('logging_steps', 10),
        evaluation_strategy="steps",
        eval_steps=training_config.get('eval_steps', 50),
        save_steps=training_config.get('save_steps', 100),
        save_total_limit=3,
        fp16=training_config.get('fp16', True),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard by default
        push_to_hub=False,
        hub_model_id=None,
    )

def generate_sample_outputs(model, tokenizer, eval_dataset: Dataset, num_samples: int = 3) -> List[Dict]:
    """Generate sample outputs for quality assessment"""
    
    samples = []
    
    try:
        # Get random samples from eval dataset
        sample_indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
        
        for idx in sample_indices:
            sample = eval_dataset[idx]
            
            # Decode the input
            input_text = tokenizer.decode(sample['input_ids'][:100], skip_special_tokens=True)
            
            # Generate output
            with torch.no_grad():
                input_ids = torch.tensor([sample['input_ids'][:100]]).to(model.device)
                
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                samples.append({
                    "input": input_text,
                    "generated": generated_text,
                    "sample_index": idx
                })
    
    except Exception as e:
        samples.append({"error": f"Failed to generate samples: {str(e)}"})
    
    return samples

def send_progress_update(callback_url: str, progress_data: Dict):
    """Send progress update to callback URL"""
    try:
        import requests
        requests.post(callback_url, json=progress_data, timeout=5)
    except Exception as e:
        print(f"Failed to send progress update: {e}")

if __name__ == "__main__":
    print("Fine-tuning engine loaded successfully")