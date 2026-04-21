"""
features.py - Feature Engineering Module
=========================================

This module builds session-level features from raw clickstream data for
purchase intent prediction. It implements a prefix-based approach to avoid
data leakage.

Key Function:
- build_prefix_intent_dataset(): Creates features from early session behavior

Methodology:
------------
The prefix-based approach only uses the first N clicks of a session to
predict whether the user will complete their session quickly (indicating
purchase intent). This avoids the circular logic of using full session
data to predict session outcomes.

Features Generated:
- Price statistics (mean, std, min, max) from early clicks
- Categorical preferences (mode of categories viewed)
- Variety metrics (nunique of categories explored)
"""

import numpy as np
import pandas as pd


# Categorical columns that represent user preferences
CATEGORICAL_CODE_COLS = [
    "country",                      # User's geographic location
    "page_1_(main_category)",       # Main product category (e.g., shirts, pants)
    "page_2_(clothing_model)",      # Specific product model
    "colour",                       # Product color preference
    "model_photography",            # Photography style of product images
]


def _mode(series: pd.Series):
    """
    Get the mode (most frequent value) of a series.
    
    Falls back to first non-null value if no mode exists.
    
    Parameters
    ----------
    series : pd.Series
        Input series to find mode of
        
    Returns
    -------
    scalar
        Most frequent value or first non-null value
    """
    m = series.mode(dropna=True)
    if len(m) == 0:
        return series.dropna().iloc[0] if series.dropna().shape[0] > 0 else np.nan
    return m.iloc[0]


def build_prefix_intent_dataset(
    df: pd.DataFrame,
    prefix_n: int = 5,
    horizon_k: int = 5
) -> pd.DataFrame:
    """
    Build a non-leaky purchase intent dataset using prefix-based features.

    Problem Definition:
    -------------------
    Using only the first `prefix_n` interactions of a session,
    predict whether the session will end within the next `horizon_k` interactions.
    
    Short sessions (quick decisions) are labeled as positive (purchase intent),
    while long sessions (extensive browsing) are labeled as negative.

    Target Logic:
    -------------
    target = 1  if total_session_length <= prefix_n + horizon_k (quick decision)
           = 0  otherwise (extended browsing)

    This approach avoids:
    - Using future information to predict past behavior
    - Session-length features that directly encode the target
    - Circular label definitions

    Parameters
    ----------
    df : pd.DataFrame
        Standardized clickstream DataFrame with session_id, order, and feature columns
    prefix_n : int, default=5
        Number of early clicks to use for feature extraction
    horizon_k : int, default=5
        Future window to define "quick purchase" threshold

    Returns
    -------
    pd.DataFrame
        Session-level dataset with:
        - session_id: Session identifier
        - session_len: Total session length (for analysis only, not a feature)
        - target: Binary label (1 = quick decision, 0 = extended browsing)
        - *_pfx features: Features derived from first prefix_n clicks only
    """

    # ---- Validate required columns ----
    if "session_id" not in df.columns or "order" not in df.columns:
        raise ValueError("Expected columns 'session_id' and 'order'.")

    # ---- Calculate total session length (USED ONLY FOR TARGET, NOT AS FEATURE) ----
    session_len = df.groupby("session_id").size().rename("session_len")

    # ---- Define target: quick decision = session ends within horizon ----
    # Sessions that complete quickly suggest decisive purchase intent
    target = (session_len <= (prefix_n + horizon_k)).astype(int).rename("target")

    # ---- Extract prefix slice (first N clicks only for features) ----
    df_sorted = df.sort_values(["session_id", "order"])
    prefix = df_sorted[df_sorted["order"] <= prefix_n].copy()

    # ---- Build numeric price features from early behavior ----
    numeric_aggs = {}

    if "price" in prefix.columns:
        numeric_aggs.update({
            "price_mean_pfx": ("price", "mean"),   # Average price of viewed products
            "price_std_pfx": ("price", "std"),     # Price variation (comparison shopping)
            "price_min_pfx": ("price", "min"),     # Lowest price viewed
            "price_max_pfx": ("price", "max"),     # Highest price viewed
        })

    if "price_2" in prefix.columns:
        numeric_aggs.update({
            "price2_mean_pfx": ("price_2", "mean"),  # Secondary price metric
            "price2_std_pfx": ("price_2", "std"),
        })

    num_df = (
        prefix.groupby("session_id").agg(**numeric_aggs)
        if numeric_aggs else
        pd.DataFrame(index=session_len.index)
    )

    # ---- Build categorical preference features ----
    # Mode = dominant category/color preference in early clicks
    # Nunique = variety of exploration (high = browsing, low = focused)
    cat_mode = {}
    cat_nunique = {}

    for col in CATEGORICAL_CODE_COLS:
        if col in prefix.columns:
            cat_mode[f"{col}_mode_pfx"] = (col, _mode)        # Most frequent value
            cat_nunique[f"{col}_nunique_pfx"] = (col, "nunique")  # Variety count

    mode_df = (
        prefix.groupby("session_id").agg(**cat_mode)
        if cat_mode else
        pd.DataFrame(index=session_len.index)
    )

    nunique_df = (
        prefix.groupby("session_id").agg(**cat_nunique)
        if cat_nunique else
        pd.DataFrame(index=session_len.index)
    )

    # ---- Combine all features into final dataset ----
    out = (
        pd.DataFrame(index=session_len.index)
        .join(session_len)   # Keep for analysis only (not used as feature)
        .join(target)        # Binary target variable
        .join(num_df)        # Numeric price features
        .join(mode_df)       # Categorical mode features
        .join(nunique_df)    # Categorical variety features
        .reset_index()
        .rename(columns={"index": "session_id"})
    )

    # ---- Handle missing values in numeric features ----
    # std can be NaN if only 1 value exists; fill with 0
    for c in out.columns:
        if c.endswith("_pfx") and out[c].dtype.kind in "fc":
            out[c] = out[c].fillna(0)

    return out