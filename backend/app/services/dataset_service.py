"""
Dataset processing, validation, health checking, and synthetic generation for MuleNet AI.
"""

import pandas as pd
import numpy as np
import io
import logging
from typing import Dict, Any, Tuple, Optional
from app.services.bank_features import BANK_FINALIZED_FEATURES, TARGET_VARIABLE

logger = logging.getLogger(__name__)

# Global in-memory storage for current dataset
_CURRENT_DATASET: Optional[pd.DataFrame] = None
_DATASET_METADATA: Dict[str, Any] = {
    "filename": "Synthetic Demonstration Data",
    "is_demo": True,
    "rows": 0,
    "columns": 0,
    "target_found": True,
    "health_score": 98.0,
    "legitimate_count": 0,
    "suspicious_count": 0,
    "mule_rate": 0.0,
    "imbalance_ratio": "1:7"
}

def generate_synthetic_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic bank transaction data with continuous feature overlap across risk spectrum."""
    np.random.seed(seed)
    
    # Continuous latent risk intensity z_i across Beta spectrum
    z = np.random.beta(a=0.6, b=1.6, size=n_samples)
    
    # Specific demonstration anchor profiles with varied behaviors
    z[5] = 0.04   # ACC-100005 (LOW ~ 4%)
    z[12] = 0.16  # ACC-100012 (LOW ~ 16%)
    z[20] = 0.35  # ACC-100020 (MEDIUM ~ 35%)
    z[35] = 0.65  # ACC-100035 (HIGH ~ 65%)
    z[99] = 0.94  # ACC-100099 (CRITICAL ~ 94%)
    z[98] = 0.88  # ACC-100098 (CRITICAL ~ 88%)
    
    # Target variable y_i is sampled probabilistically based on z_i
    y = (np.random.rand(n_samples) < (z**1.2)).astype(int)
    
    data = {}
    
    # 1. Bank Finalized Features scaling smoothly with continuous risk propensity z_i
    data["F115"] = np.round(np.clip(0.3 + z * 5.5 + np.random.normal(0, 0.5, n_samples), 0, None), 4)
    data["F321"] = np.round(np.clip(0.4 + z * 6.0 + np.random.normal(0, 0.6, n_samples), 0, None), 4)
    data["F527"] = np.round(np.clip(0.5 + z * 6.5 + np.random.normal(0, 0.7, n_samples), 0, None), 4)
    data["F531"] = np.round(np.clip(0.5 + z * 7.0 + np.random.normal(0, 0.8, n_samples), 0, None), 4)
    data["F670"] = np.round(np.clip(1.0 + z * 28.0 + np.random.normal(0, 3.0, n_samples), 0, None), 4)
    data["F1692"] = np.round(np.clip(0.8 + z * 18.0 + np.random.normal(0, 2.0, n_samples), 0, None), 4)
    data["F2082"] = np.round(np.clip(0.5 + z * 16.0 + np.random.normal(0, 1.8, n_samples), 0, None), 4)
    data["F2122"] = np.round(np.clip(0.5 + z * 14.0 + np.random.normal(0, 1.5, n_samples), 0, None), 4)
    data["F2582"] = np.round(np.clip(200.0 + z * 52000.0 + np.random.normal(0, 5000, n_samples), 0, None), 4)
    data["F2678"] = np.round(np.clip(500.0 + z * 105000.0 + np.random.normal(0, 10000, n_samples), 0, None), 4)
    data["F2737"] = np.round(np.clip(400.0 + z * 82000.0 + np.random.normal(0, 8000, n_samples), 0, None), 4)
    data["F2956"] = np.round(np.clip(0.2 + z * 28.0 + np.random.normal(0, 3.0, n_samples), 0, None), 4)
    data["F3043"] = np.round(np.clip(0.1 + z * 20.0 + np.random.normal(0, 2.5, n_samples), 0, None), 4)
    data["F3836"] = np.round(np.clip(55000.0 * (1.05 - z) + np.random.normal(0, 4000, n_samples), 100, None), 4)
    data["F3887"] = np.round(np.clip(1800.0 * (1.05 - z) + np.random.normal(0, 150, n_samples), 1, None), 4)
    data["F3889"] = np.round(np.clip(9.0 * (1.05 - z) + np.random.normal(0, 1.0, n_samples), 1, 10), 4)
    data["F3891"] = np.random.randint(1, 10, n_samples)
    data["F3894"] = np.round(np.clip(52.0 - z * 28.0 + np.random.normal(0, 4.0, n_samples), 18, 75), 4)

    # 2. Add extra synthetic features F1 to F100 scaling with z_i
    signal_map = {
        "F1": (2.0, 30.0),
        "F10": (1.0, 22.0),
        "F11": (0.8, 18.0),
        "F12": (0.5, 14.0),
        "F13": (1.0, 20.0),
        "F14": (0.2, 16.0),
        "F20": (0.3, 10.0),
        "F30": (0.4, 14.0),
        "F50": (5.0, 90.0),
        "F60": (10.0, 85.0),
        "F70": (0.5, 18.0),
        "F80": (1.0, 25.0)
    }

    for i in range(1, 101):
        fname = f"F{i}"
        if fname not in data:
            if fname in signal_map:
                base_v, scale_v = signal_map[fname]
                data[fname] = np.round(np.clip(base_v + z * scale_v + np.random.normal(0, scale_v * 0.15, n_samples), 0, None), 4)
            else:
                data[fname] = np.round(np.clip(np.random.normal(10.0, 3.0, n_samples), 0, None), 4)

    account_ids = [f"ACC-{100000 + i}" for i in range(n_samples)]
    df = pd.DataFrame(data)
    df.insert(0, "ACCOUNT_ID", account_ids)
    df[TARGET_VARIABLE] = y
    
    return df

def get_current_dataset() -> pd.DataFrame:
    """Returns currently loaded dataset or initializes synthetic fallback."""
    global _CURRENT_DATASET
    if _CURRENT_DATASET is None:
        _CURRENT_DATASET = generate_synthetic_dataset()
        update_dataset_metadata(_CURRENT_DATASET, filename="Synthetic Demonstration Data", is_demo=True)
    return _CURRENT_DATASET

def update_dataset_metadata(df: pd.DataFrame, filename: str, is_demo: bool = False):
    """Calculates dataset statistics and health score."""
    global _DATASET_METADATA
    
    rows = len(df)
    cols = len(df.columns)
    target_found = TARGET_VARIABLE in df.columns
    
    legit_count = 0
    mule_count = 0
    mule_rate = 0.0
    imbalance_ratio = "N/A"
    
    if target_found:
        legit_count = int((df[TARGET_VARIABLE] == 0).sum())
        mule_count = int((df[TARGET_VARIABLE] == 1).sum())
        mule_rate = round((mule_count / rows) * 100, 2) if rows > 0 else 0.0
        if mule_count > 0:
            ratio_val = round(legit_count / mule_count, 1)
            imbalance_ratio = f"1:{ratio_val}"
            
    missing_pct = float(df.isnull().sum().sum() / (rows * cols)) * 100 if rows * cols > 0 else 0.0
    
    _DATASET_METADATA = {
        "filename": filename,
        "is_demo": is_demo,
        "rows": rows,
        "columns": cols,
        "target_found": target_found,
        "health_score": round(max(0.0, 100.0 - missing_pct), 1),
        "legitimate_count": legit_count,
        "suspicious_count": mule_count,
        "mule_rate": mule_rate,
        "imbalance_ratio": imbalance_ratio
    }

def process_uploaded_csv(file_bytes: bytes, filename: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Processes uploaded CSV file, validates columns, and sets active dataset."""
    global _CURRENT_DATASET
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        if len(df) < 5:
            return False, "Dataset must contain at least 5 records.", {}
            
        if "ACCOUNT_ID" not in df.columns:
            df.insert(0, "ACCOUNT_ID", [f"ACC-{100000 + i}" for i in range(len(df))])
            
        _CURRENT_DATASET = df
        update_dataset_metadata(df, filename=filename, is_demo=False)
        logger.info(f"Successfully loaded uploaded dataset '{filename}' with {len(df)} rows.")
        return True, f"Dataset '{filename}' uploaded successfully with {len(df)} records.", _DATASET_METADATA
    except Exception as e:
        logger.error(f"Failed to process CSV upload: {str(e)}")
        return False, f"CSV parsing error: {str(e)}", {}
