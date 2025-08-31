# Dataset Loader - Modal Function for downloading and processing HuggingFace datasets
import modal
import pandas as pd
import os
import json
from typing import Dict, List, Any, Tuple
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
import logging

from .image_config import data_processing_image

# Create Modal app
app = modal.App("gpt-oss-dataset-loader")

# Create volumes for caching
dataset_cache_vol = modal.Volume.from_name("dataset-cache", create_if_missing=True)

@app.function(
    image=data_processing_image,
    volumes={"/root/.cache/datasets": dataset_cache_vol},
    timeout=3600,  # 1 hour timeout for large downloads
    memory=8192,   # 8GB memory for processing large CSVs
)
def download_and_validate_datasets(
    datasets_config: List[Dict[str, Any]],
    job_id: str
) -> Dict[str, Any]:
    """
    Download and validate the travel datasets from HuggingFace
    
    Args:
        datasets_config: List of dataset configurations
        job_id: Unique job identifier for tracking
        
    Returns:
        Dictionary with dataset information and validation results
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    results = {
        "job_id": job_id,
        "datasets": [],
        "total_datasets": len(datasets_config),
        "status": "success",
        "errors": []
    }
    
    try:
        for idx, dataset_config in enumerate(datasets_config):
            logger.info(f"Processing dataset {idx + 1}/{len(datasets_config)}: {dataset_config['name']}")
            
            dataset_result = {
                "dataset_id": f"dataset_{idx + 1}",
                "name": dataset_config["name"],
                "description": dataset_config.get("description", ""),
                "priority": dataset_config.get("priority", idx + 1),
                "status": "downloading",
                "file_info": {},
                "sample_data": [],
                "validation": {}
            }
            
            try:
                # Download the CSV file from HuggingFace
                file_path = hf_hub_download(
                    repo_id=dataset_config["dataset_name"],
                    filename=dataset_config["file_name"],
                    repo_type="dataset",
                    cache_dir="/root/.cache/datasets"
                )
                
                logger.info(f"Downloaded {dataset_config['file_name']} to {file_path}")
                
                # Get file info
                file_size = os.path.getsize(file_path)
                dataset_result["file_info"] = {
                    "path": file_path,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                }
                
                # Load and validate CSV
                logger.info(f"Loading CSV file: {file_path}")
                df = pd.read_csv(file_path)
                
                # Basic validation
                row_count = len(df)
                col_count = len(df.columns)
                
                dataset_result["validation"] = {
                    "row_count": row_count,
                    "column_count": col_count,
                    "columns": list(df.columns),
                    "has_data": row_count > 0,
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
                }
                
                # Get sample data (first 3 rows)
                sample_size = min(3, row_count)
                if sample_size > 0:
                    sample_df = df.head(sample_size)
                    dataset_result["sample_data"] = sample_df.to_dict('records')
                
                # Validate required columns based on dataset type
                if "conversation" in dataset_config["file_name"].lower():
                    # Travel conversations dataset
                    required_cols = ["conversation", "topic"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        dataset_result["validation"]["warnings"] = f"Missing columns: {missing_cols}"
                    else:
                        # Check for empty conversations
                        empty_conversations = df["conversation"].isnull().sum()
                        dataset_result["validation"]["empty_conversations"] = empty_conversations
                        
                elif "qa" in dataset_config["file_name"].lower():
                    # Travel QA dataset  
                    required_cols = ["question", "answer", "topic"]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        dataset_result["validation"]["warnings"] = f"Missing columns: {missing_cols}"
                    else:
                        # Check for empty Q&A pairs
                        empty_questions = df["question"].isnull().sum()
                        empty_answers = df["answer"].isnull().sum()
                        dataset_result["validation"]["empty_questions"] = empty_questions
                        dataset_result["validation"]["empty_answers"] = empty_answers
                
                dataset_result["status"] = "completed"
                logger.info(f"Successfully processed dataset: {dataset_config['name']}")
                
            except Exception as e:
                error_msg = f"Error processing dataset {dataset_config['name']}: {str(e)}"
                logger.error(error_msg)
                dataset_result["status"] = "error"
                dataset_result["error"] = error_msg
                results["errors"].append(error_msg)
            
            results["datasets"].append(dataset_result)
        
        # Overall status
        failed_datasets = [d for d in results["datasets"] if d["status"] == "error"]
        if failed_datasets:
            results["status"] = "partial_failure" if len(failed_datasets) < len(datasets_config) else "failed"
        
        logger.info(f"Dataset loading completed. Status: {results['status']}")
        
    except Exception as e:
        error_msg = f"Critical error in dataset loading: {str(e)}"
        logger.error(error_msg)
        results["status"] = "failed"
        results["errors"].append(error_msg)
    
    return results

@app.function(
    image=data_processing_image,
    volumes={"/root/.cache/datasets": dataset_cache_vol},
    timeout=1800,  # 30 minutes
    memory=4096,   # 4GB memory
)
def load_dataset_for_training(
    dataset_config: Dict[str, Any],
    job_id: str,
    dataset_index: int
) -> Dict[str, Any]:
    """
    Load a specific dataset and prepare it for training
    
    Args:
        dataset_config: Configuration for the specific dataset
        job_id: Unique job identifier
        dataset_index: Index of the dataset in the queue
        
    Returns:
        Dictionary with processed dataset information
    """
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading dataset for training: {dataset_config['name']}")
        
        # Download the file
        file_path = hf_hub_download(
            repo_id=dataset_config["dataset_name"],
            filename=dataset_config["file_name"],
            repo_type="dataset",
            cache_dir="/root/.cache/datasets"
        )
        
        # Load CSV
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Save the processed dataset
        cache_path = f"/root/.cache/datasets/{job_id}_dataset_{dataset_index}"
        dataset.save_to_disk(cache_path)
        
        result = {
            "dataset_id": f"dataset_{dataset_index}",
            "name": dataset_config["name"],
            "cache_path": cache_path,
            "row_count": len(df),
            "columns": list(df.columns),
            "status": "ready_for_training"
        }
        
        logger.info(f"Dataset {dataset_config['name']} prepared for training")
        return result
        
    except Exception as e:
        error_msg = f"Error loading dataset for training: {str(e)}"
        logger.error(error_msg)
        return {
            "dataset_id": f"dataset_{dataset_index}",
            "name": dataset_config.get("name", "Unknown"),
            "status": "error",
            "error": error_msg
        }

# Default dataset configurations
DEFAULT_DATASETS = [
    {
        "name": "Travel Conversations Dataset",
        "dataset_name": "soniawmeyer/travel-conversations-finetuning",
        "file_name": "conversational_sample_processed_with_topic.csv",
        "format": "csv",
        "priority": 1,
        "description": "680MB CSV - Conversational travel data with topics",
        "preprocessing": {
            "text_column": "conversation",
            "topic_column": "topic",
            "min_length": 50,
            "max_length": 2048
        }
    },
    {
        "name": "Travel QA Dataset",
        "dataset_name": "soniawmeyer/travel-conversations-finetuning", 
        "file_name": "travel_QA_processed_with_topic.csv",
        "format": "csv",
        "priority": 2,
        "description": "147MB CSV - Question-answer pairs for travel topics",
        "preprocessing": {
            "question_column": "question",
            "answer_column": "answer", 
            "topic_column": "topic",
            "min_length": 20,
            "max_length": 1024
        }
    }
]

if __name__ == "__main__":
    # Test the dataset loader
    import asyncio
    
    async def test_loader():
        result = await download_and_validate_datasets.remote.aio(
            datasets_config=DEFAULT_DATASETS,
            job_id="test_job_123"
        )
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_loader())