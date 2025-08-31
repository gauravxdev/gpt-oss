# Modal Image Configuration for GPT-OSS Fine-tuning
# This creates a custom container image with all dependencies for fine-tuning

import modal

# Create custom image with all fine-tuning dependencies
def create_finetune_image():
    """Create the Modal image with all required dependencies for fine-tuning GPT-OSS 20B"""
    
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        # Install system dependencies
        .run_commands([
            "apt-get update",
            "apt-get install -y git wget curl build-essential",
            "apt-get clean && rm -rf /var/lib/apt/lists/*"
        ])
        # Install PyTorch with CUDA support
        .pip_install([
            "torch>=2.1.0",
            "torchvision", 
            "torchaudio",
            "--extra-index-url https://download.pytorch.org/whl/cu121"
        ])
        # Install core ML/AI libraries
        .pip_install([
            "transformers>=4.36.0",
            "datasets>=2.15.0", 
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "bitsandbytes>=0.41.0",
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0"
        ])
        # Install Unsloth for optimized fine-tuning (2x faster, 70% less memory)
        .pip_install([
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        ])
        # Install additional dependencies
        .pip_install([
            "huggingface_hub>=0.19.0",
            "tokenizers>=0.15.0",
            "safetensors>=0.4.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
            "pandas>=2.1.0",
            "numpy>=1.24.0",
            "requests>=2.31.0",
            "tqdm>=4.66.0",
            "psutil>=5.9.0"
        ])
        # Install vLLM for inference support  
        .pip_install([
            "vllm>=0.2.5"
        ])
        # Set environment variables for optimization
        .env({
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",  # Disable by default, enable if needed
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        })
    )

# Alternative lightweight image for data processing only
def create_data_processing_image():
    """Create a lighter image for data processing tasks"""
    
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install([
            "pandas>=2.1.0",
            "numpy>=1.24.0", 
            "datasets>=2.15.0",
            "huggingface_hub>=0.19.0",
            "requests>=2.31.0",
            "tqdm>=4.66.0",
            "scikit-learn>=1.3.0"
        ])
        .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        })
    )

# Export the main image
finetune_image = create_finetune_image()
data_processing_image = create_data_processing_image()