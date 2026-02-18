import logging
import os
import d3rlpy
import torch
import numpy as np
import utils
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer

def train_dt(config: dict) -> d3rlpy.algos.DiscreteDecisionTransformer:
    """
    Train the DT model.
    Args:
        config: The configuration dictionary.
    Returns:
        dt: The trained DT model.
    """
    logging.info("Starting DT Training...")
    
    # Validate configuration
    validate_config(config)
    
    # Get default paths
    dataset_path, save_path, log_dir = get_default_paths(config)
    
    # Load dataset and get DT configuration
    dataset = load_dataset(dataset_path)
    dt_config = config.get("dt", {})
    
    # # Set device
    # device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    # Set device (force CPU - GTX 1080 Ti CUDA 6.1 not supported by current PyTorch)
    device = "cpu:0"
    
    # Initialize DT model using d3rlpy v2.x Config API
    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=dt_config.get("batch_size", 64),
        context_size=dt_config.get("context_size", 20),
        learning_rate=dt_config.get("learning_rate", 1e-4),
    ).create(device=device)

    # Train DT model
    dt.fit(
        dataset,
        n_steps=dt_config.get("n_steps", 10000),
        n_steps_per_epoch=dt_config.get("n_steps_per_epoch", 1000),
        logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=log_dir),
        show_progress=True
    )
    
    # Save DT model
    save_dt(dt, save_path)

    return dt

def load_dt(save_path: str) -> d3rlpy.algos.DiscreteDecisionTransformer:
    """
    Load the DT model.
    Args:
        save_path: The path to load the model.
    Returns:
        dt: The loaded DT model.
    """
    dt = d3rlpy.load_learnable(save_path)
    logging.info(f"Loaded DT model from {save_path}")
    return dt

def load_dataset(dataset_path: str) -> d3rlpy.dataset.MDPDataset:
    """
    Load the dataset.
    Args:
        dataset_path: The path to load the dataset.
    Returns:
        dataset: The loaded dataset (ReplayBuffer).
    """
    buffer = InfiniteBuffer()
    dataset = ReplayBuffer.load(dataset_path, buffer)
    logging.info(f"Loaded dataset from {dataset_path}")
    return dataset

def get_default_paths(config: dict) -> tuple[str, str, str]:
    """
    Get default paths for DT training.
    Args:
        config: The configuration dictionary.
    Returns:
        tuple: (dataset_path, save_path, log_dir)
    """
    # Access the nested "dt" section of the config
    dt_config = config.get("dt", {})
    
    # Only use dynamic date/time as fallback if not specified in config
    date_str, time_str = utils.get_current_date_time_strings()
    
    dataset_path = dt_config.get("dataset_path", f"results/data/{date_str}/{time_str}/expert_data.h5")
    save_path = dt_config.get("save_path", f"results/models/DT/{date_str}/{time_str}/dt_model.d3")
    log_dir = dt_config.get("log_dir", f"logs/dt/{date_str}/{time_str}")
    return dataset_path, save_path, log_dir

def validate_config(config: dict) -> bool:
    """
    Validate the configuration for DT training.
    Args:
        config: The configuration dictionary.
    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    try:
        dataset_path, save_path, log_dir = get_default_paths(config)
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset not found at {dataset_path}. Cannot train DT.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating configuration: {e}")
        return False

def save_dt(dt: d3rlpy.algos.DiscreteDecisionTransformer, save_path: str) -> None:
    """
    Save the DT model.
    Args:
        dt: The DT model.
        save_path: The path to save the model.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dt.save(save_path)
    logging.info(f"Saved DT model to {save_path}")
