"""
load_data.py - Data Loading and Preprocessing Module
=====================================================

This module handles loading raw clickstream data from CSV files and
standardizing column names for downstream analysis.

Key Functions:
- load_raw_data(): Load semicolon-separated CSV files
- standardize_columns(): Normalize column names and create synthetic timestamps

Dataset Context:
----------------
The raw data comes from a clothing e-commerce store and contains user
browsing sessions with click-level events. The dataset uses semicolon
as delimiter and has columns like session_id, year, month, day, order, etc.

Note: This dataset does NOT contain explicit purchase events. The 'order'
column represents click sequence within a session, not purchase orders.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load semicolon-separated clickstream CSV.
    
    Parameters
    ----------
    path : str
        File path to the raw CSV data
        
    Returns
    -------
    pd.DataFrame
        Raw clickstream data with original column names
    """
    return pd.read_csv(path, sep=";")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns and create event_time based on date + click sequence 'order'.
    
    This dataset has no true timestamps, so we reconstruct event_time to preserve
    ordering within sessions. The 'order' column indicates click sequence (1st, 2nd, etc.)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw clickstream DataFrame from load_raw_data()
        
    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with:
        - Lowercase column names with underscores
        - Synthetic event_time column for ordering
        - event_type column (all 'view' since no purchase labels exist)
        
    Raises
    ------
    ValueError
        If required columns (session_id, year, month, day, order) are missing
    """

    # Normalize column names: strip whitespace, lowercase, replace spaces
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Validate required columns exist
    required = {"session_id", "year", "month", "day", "order"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build synthetic event_time: combine date components + click order as seconds
    # This preserves chronological ordering within sessions
    df["event_date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["event_time"] = df["event_date"] + pd.to_timedelta(df["order"], unit="s")

    # Sort by session and click order for consistency
    df = df.sort_values(["session_id", "order"]).reset_index(drop=True)

    # Event type: This dataset does not contain explicit conversion labels.
    # We treat all clicks as views; purchase intent is derived at session-level
    # using behavioral features, not from per-click purchase events.
    df["event_type"] = "view"

    return df