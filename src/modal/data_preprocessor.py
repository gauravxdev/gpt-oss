# Data Preprocessor - Modal Function for preparing datasets for fine-tuning
import modal
import pandas as pd
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import logging

from .image_config import data_processing_image

app = modal.App("gpt-oss-data-preprocessor")

# Volume for caching datasets and preprocessed data
dataset_cache_vol = modal.Volume.from_name("dataset-cache", create_if_missing=True)
processed_cache_vol = modal.Volume.from_name("processed-data-cache", create_if_missing=True)

@app.function(
    image=data_processing_image,
    volumes={
        "/root/.cache/datasets": dataset_cache_vol,
        "/root/.cache/processed": processed_cache_vol
    },
    timeout=3600,  # 1 hour timeout
    memory=16384,  # 16GB memory for large dataset processing
)
def preprocess_conversation_dataset(
    dataset_path: str,
    dataset_config: Dict[str, Any],
    tokenizer_name: str = "openai/gpt-oss-20b",
    max_length: int = 2048,
    job_id: str = "default"
) -> Dict[str, Any]:
    """
    Preprocess the travel conversations dataset for fine-tuning
    
    Args:
        dataset_path: Path to the cached dataset
        dataset_config: Dataset configuration
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        job_id: Job identifier for caching
        
    Returns:
        Dictionary with preprocessing results
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting preprocessing for conversation dataset: {dataset_config['name']}")
        
        # Load the dataset
        dataset = Dataset.load_from_disk(dataset_path)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir="/root/.cache/datasets"
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Convert to pandas for easier processing
        df = dataset.to_pandas()
        
        # Clean and format conversations
        processed_conversations = []
        
        for idx, row in df.iterrows():
            try:
                conversation = str(row.get('conversation', ''))
                topic = str(row.get('topic', 'travel'))
                
                # Clean the conversation text
                cleaned_conversation = clean_conversation_text(conversation)
                
                # Skip if too short or too long
                if len(cleaned_conversation) < dataset_config.get('preprocessing', {}).get('min_length', 50):
                    continue
                    
                if len(cleaned_conversation) > dataset_config.get('preprocessing', {}).get('max_length', 2048):
                    cleaned_conversation = cleaned_conversation[:dataset_config.get('preprocessing', {}).get('max_length', 2048)]
                
                # Format as chat template
                formatted_text = format_conversation_for_training(
                    conversation=cleaned_conversation,
                    topic=topic,
                    tokenizer=tokenizer
                )
                
                # Tokenize to check length
                tokens = tokenizer.encode(formatted_text, truncation=True, max_length=max_length)
                
                if len(tokens) > 10:  # Minimum viable token length
                    processed_conversations.append({
                        'text': formatted_text,
                        'topic': topic,
                        'token_count': len(tokens),
                        'original_index': idx
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing conversation at index {idx}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_conversations)} valid conversations from {len(df)} original samples")
        
        # Create train/validation split
        train_split = dataset_config.get('train_split', 0.9)
        split_idx = int(len(processed_conversations) * train_split)
        
        train_data = processed_conversations[:split_idx]
        val_data = processed_conversations[split_idx:]
        
        # Convert back to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Save processed dataset
        output_path = f"/root/.cache/processed/{job_id}_conversations_processed"
        dataset_dict.save_to_disk(output_path)
        
        # Calculate statistics
        train_token_stats = calculate_token_statistics([item['token_count'] for item in train_data])
        val_token_stats = calculate_token_statistics([item['token_count'] for item in val_data])
        
        result = {
            "dataset_type": "conversations",
            "status": "success",
            "output_path": output_path,
            "statistics": {
                "total_samples": len(processed_conversations),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "train_split_ratio": train_split,
                "train_token_stats": train_token_stats,
                "validation_token_stats": val_token_stats,
                "dropped_samples": len(df) - len(processed_conversations)
            },
            "sample_data": {
                "train_sample": train_data[0] if train_data else None,
                "validation_sample": val_data[0] if val_data else None
            }
        }
        
        logger.info(f"Conversation dataset preprocessing completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Error preprocessing conversation dataset: {str(e)}"
        logger.error(error_msg)
        return {
            "dataset_type": "conversations",
            "status": "error",
            "error": error_msg
        }

@app.function(
    image=data_processing_image,
    volumes={
        "/root/.cache/datasets": dataset_cache_vol,
        "/root/.cache/processed": processed_cache_vol
    },
    timeout=3600,  # 1 hour timeout
    memory=16384,  # 16GB memory for large dataset processing
)
def preprocess_qa_dataset(
    dataset_path: str,
    dataset_config: Dict[str, Any],
    tokenizer_name: str = "openai/gpt-oss-20b",
    max_length: int = 2048,
    job_id: str = "default"
) -> Dict[str, Any]:
    """
    Preprocess the travel QA dataset for fine-tuning
    
    Args:
        dataset_path: Path to the cached dataset
        dataset_config: Dataset configuration
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        job_id: Job identifier for caching
        
    Returns:
        Dictionary with preprocessing results
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting preprocessing for QA dataset: {dataset_config['name']}")
        
        # Load the dataset
        dataset = Dataset.load_from_disk(dataset_path)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir="/root/.cache/datasets"
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Convert to pandas for easier processing
        df = dataset.to_pandas()
        
        # Clean and format QA pairs
        processed_qa_pairs = []
        
        for idx, row in df.iterrows():
            try:
                question = str(row.get('question', ''))
                answer = str(row.get('answer', ''))
                topic = str(row.get('topic', 'travel'))
                
                # Clean the text
                cleaned_question = clean_text(question)
                cleaned_answer = clean_text(answer)
                
                # Skip if too short
                min_length = dataset_config.get('preprocessing', {}).get('min_length', 20)
                if len(cleaned_question) < min_length or len(cleaned_answer) < min_length:
                    continue
                
                # Format as chat template for QA
                formatted_text = format_qa_for_training(
                    question=cleaned_question,
                    answer=cleaned_answer,
                    topic=topic,
                    tokenizer=tokenizer
                )
                
                # Tokenize to check length
                tokens = tokenizer.encode(formatted_text, truncation=True, max_length=max_length)
                
                if len(tokens) > 10:  # Minimum viable token length
                    processed_qa_pairs.append({
                        'text': formatted_text,
                        'question': cleaned_question,
                        'answer': cleaned_answer,
                        'topic': topic,
                        'token_count': len(tokens),
                        'original_index': idx
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing QA pair at index {idx}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_qa_pairs)} valid QA pairs from {len(df)} original samples")
        
        # Create train/validation split
        train_split = dataset_config.get('train_split', 0.9)
        split_idx = int(len(processed_qa_pairs) * train_split)
        
        train_data = processed_qa_pairs[:split_idx]
        val_data = processed_qa_pairs[split_idx:]
        
        # Convert back to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Save processed dataset
        output_path = f"/root/.cache/processed/{job_id}_qa_processed"
        dataset_dict.save_to_disk(output_path)
        
        # Calculate statistics
        train_token_stats = calculate_token_statistics([item['token_count'] for item in train_data])
        val_token_stats = calculate_token_statistics([item['token_count'] for item in val_data])
        
        result = {
            "dataset_type": "qa",
            "status": "success",
            "output_path": output_path,
            "statistics": {
                "total_samples": len(processed_qa_pairs),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "train_split_ratio": train_split,
                "train_token_stats": train_token_stats,
                "validation_token_stats": val_token_stats,
                "dropped_samples": len(df) - len(processed_qa_pairs)
            },
            "sample_data": {
                "train_sample": train_data[0] if train_data else None,
                "validation_sample": val_data[0] if val_data else None
            }
        }
        
        logger.info(f"QA dataset preprocessing completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Error preprocessing QA dataset: {str(e)}"
        logger.error(error_msg)
        return {
            "dataset_type": "qa",
            "status": "error",
            "error": error_msg
        }

def clean_conversation_text(text: str) -> str:
    """Clean conversation text"""
    if not text or pd.isna(text):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\'\"\(\)\-]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()

def clean_text(text: str) -> str:
    """Clean general text"""
    if not text or pd.isna(text):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special markdown and formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()

def format_conversation_for_training(conversation: str, topic: str, tokenizer) -> str:
    """Format conversation for training with chat template"""
    
    # GPT-OSS chat template format
    system_prompt = f"You are a helpful travel assistant specialized in {topic}. Provide informative and engaging responses about travel topics."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Let's discuss travel topics related to {topic}."},
        {"role": "assistant", "content": conversation}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=False
    )
    
    return formatted_text

def format_qa_for_training(question: str, answer: str, topic: str, tokenizer) -> str:
    """Format QA pair for training with chat template"""
    
    # GPT-OSS chat template format
    system_prompt = f"You are a helpful travel assistant specialized in {topic}. Answer travel questions accurately and helpfully."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return formatted_text

def calculate_token_statistics(token_counts: List[int]) -> Dict[str, float]:
    """Calculate statistics for token counts"""
    if not token_counts:
        return {}
    
    import numpy as np
    
    return {
        "mean": float(np.mean(token_counts)),
        "median": float(np.median(token_counts)),
        "std": float(np.std(token_counts)),
        "min": float(np.min(token_counts)),
        "max": float(np.max(token_counts)),
        "total": int(np.sum(token_counts))
    }

if __name__ == "__main__":
    # Test the preprocessor
    import asyncio
    
    async def test_preprocessor():
        # Test conversation preprocessing
        test_config = {
            "name": "Test Conversations",
            "preprocessing": {
                "min_length": 50,
                "max_length": 2048
            },
            "train_split": 0.9
        }
        
        # This would need an actual dataset path in practice
        result = await preprocess_conversation_dataset.remote.aio(
            dataset_path="/test/path",
            dataset_config=test_config,
            job_id="test_preprocess"
        )
        print(json.dumps(result, indent=2))
    
    print("Data preprocessor module loaded successfully")