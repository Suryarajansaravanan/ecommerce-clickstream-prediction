# =============================================================================
# Utility Functions
# =============================================================================

import random
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across random, numpy, and other libraries.
    
    Parameters
    ----------
    seed : int
        Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set seeds for optional libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def memory_usage(df: pd.DataFrame) -> str:
    """
    Calculate and return the memory usage of a DataFrame in human-readable format.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to measure
        
    Returns
    -------
    str
        Human-readable memory usage string
    """
    mem_bytes = df.memory_usage(deep=True).sum()
    
    if mem_bytes < 1024:
        return f"{mem_bytes:.2f} B"
    elif mem_bytes < 1024 ** 2:
        return f"{mem_bytes / 1024:.2f} KB"
    elif mem_bytes < 1024 ** 3:
        return f"{mem_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{mem_bytes / (1024 ** 3):.2f} GB"
