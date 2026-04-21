"""
src - Clickstream Purchase Prediction Source Package
=====================================================

This package contains reusable modules for the clickstream purchase
prediction project. The modules handle data loading, feature engineering,
and utility functions.

Modules:
--------
- load_data: Functions for loading and standardizing raw clickstream data
- features: Feature engineering using prefix-based approach to avoid leakage

Usage:
------
    from src.load_data import load_raw_data, standardize_columns
    from src.features import build_prefix_intent_dataset
"""

from .load_data import load_raw_data, standardize_columns
from .features import build_prefix_intent_dataset

__all__ = ["load_raw_data", "standardize_columns", "build_prefix_intent_dataset"]
