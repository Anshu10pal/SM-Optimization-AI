"""
utils.py — Shared helpers: text preprocessing, embedding generation,
BM25 indexing, label encoding, confidence labelling, and logging.
"""

import re
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import LabelEncoder

from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM,
    EMBEDDINGS_CACHE_PATH, EMBEDDINGS_CACHE_INDEX,
    BM25_INDEX_PATH, LABEL_ENCODERS_PATH,
    TEXT_COLS, LOGS_DIR, INFER_LOG,
    CONFIDENCE_AUTO_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(INFER_LOG), rotation="10 MB", retention="30 days", level="INFO")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: Any) -> str:
    """
    Normalise free-text incident fields:
    - lower-case
    - remove special characters, extra whitespace
    - strip leading/trailing spaces
    """
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.\,\-\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text_fields(row: pd.Series, cols: List[str] = TEXT_COLS) -> str:
    """
    Concatenate multiple text fields into one string, separated by ' [SEP] '.
    Missing / null fields are silently skipped.
    """
    parts = []
    for col in cols:
        val = row.get(col, "")
        cleaned = clean_text(val)
        if cleaned:
            parts.append(cleaned)
    return " [SEP] ".join(parts) if parts else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to all TEXT_COLS in-place and add a 'combined_text' column.
    Handles missing columns gracefully.
    """
    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)
        else:
            df[col] = ""
            logger.warning(f"Column '{col}' not found — filled with empty strings.")
    df["combined_text"] = df.apply(combine_text_fields, axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SENTENCE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

_embedding_model: Optional[SentenceTransformer] = None   # module-level singleton


def get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer once and cache it in memory."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _cache_key(texts: List[str]) -> str:
    """SHA-256 fingerprint of a list of texts — used to detect stale caches."""
    content = "|||".join(texts[:100])   # fingerprint first 100 for speed
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_embeddings(
    texts: List[str],
    use_cache: bool = True,
    cache_prefix: str = "train",
) -> np.ndarray:
    """
    Generate sentence embeddings in batches (CPU-friendly).
    Caches results to disk to avoid recomputation on subsequent runs.

    Args:
        texts:        List of raw / preprocessed text strings.
        use_cache:    Whether to load/save from the disk cache.
        cache_prefix: Distinguishes train vs inference caches.

    Returns:
        np.ndarray of shape (N, EMBEDDING_DIM).
    """
    cache_path  = Path(str(EMBEDDINGS_CACHE_PATH).replace(".npy", f"_{cache_prefix}.npy"))
    index_path  = Path(str(EMBEDDINGS_CACHE_INDEX).replace(".pkl", f"_{cache_prefix}.pkl"))
    key         = _cache_key(texts)

    # ── Try loading from cache ─────────────────────────────────────────────
    if use_cache and cache_path.exists() and index_path.exists():
        with open(index_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("key") == key and meta.get("n") == len(texts):
            logger.info(f"Loading embeddings from cache ({cache_prefix}).")
            return np.load(str(cache_path))

    # ── Compute embeddings ─────────────────────────────────────────────────
    model = get_embedding_model()
    logger.info(f"Encoding {len(texts):,} texts in batches of {EMBEDDING_BATCH_SIZE} …")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-norm for cosine similarity
    )

    # ── Save to cache ──────────────────────────────────────────────────────
    if use_cache:
        np.save(str(cache_path), embeddings)
        with open(index_path, "wb") as f:
            pickle.dump({"key": key, "n": len(texts)}, f)
        logger.info(f"Embeddings cached to {cache_path}.")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# BM25 INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """Tokenise texts and build a BM25Okapi index."""
    tokenised = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenised)
    logger.info(f"BM25 index built over {len(texts):,} documents.")
    return bm25


def save_bm25_index(bm25: BM25Okapi, texts: List[str]) -> None:
    """Persist the BM25 index and its source texts."""
    payload = {"bm25": bm25, "texts": texts}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"BM25 index saved → {BM25_INDEX_PATH}")


def load_bm25_index() -> Tuple[Optional[BM25Okapi], List[str]]:
    """Load a saved BM25 index.  Returns (None, []) if not found."""
    if not BM25_INDEX_PATH.exists():
        logger.warning("BM25 index not found.")
        return None, []
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["texts"]


def bm25_retrieve(
    query: str,
    bm25: BM25Okapi,
    texts: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve the top-k most similar historical incidents for a query.
    Returns a list of dicts with 'text' and 'score'.
    """
    tokenised_query = query.split()
    scores = bm25.get_scores(tokenised_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"text": texts[i], "score": float(scores[i])} for i in top_indices]


# ══════════════════════════════════════════════════════════════════════════════
# LABEL ENCODERS
# ══════════════════════════════════════════════════════════════════════════════

def fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Fit a LabelEncoder for each target column.
    Returns a dict: {column_name: fitted_LabelEncoder}.
    """
    from config import TARGET_RESOLUTION_CODE, TARGET_SUBCATEGORY, TARGET_FAULT_CATEGORY
    encoders: Dict[str, LabelEncoder] = {}
    for col in [TARGET_RESOLUTION_CODE, TARGET_SUBCATEGORY, TARGET_FAULT_CATEGORY]:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].dropna().astype(str))
            encoders[col] = le
            logger.info(f"LabelEncoder for '{col}': {len(le.classes_)} classes.")
    return encoders


def save_label_encoders(encoders: Dict[str, LabelEncoder]) -> None:
    with open(LABEL_ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Label encoders saved → {LABEL_ENCODERS_PATH}")


def load_label_encoders() -> Dict[str, LabelEncoder]:
    if not LABEL_ENCODERS_PATH.exists():
        raise FileNotFoundError(f"Label encoders not found at {LABEL_ENCODERS_PATH}")
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_label(confidence: float) -> str:
    """Map a float probability to a human-readable confidence tier."""
    if confidence >= CONFIDENCE_AUTO_THRESHOLD:
        return "Auto"
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Needs Review"


def evaluate_confidence_buckets(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Compute calibration statistics per confidence bucket.
    Useful for understanding where the model is over/under-confident.

    Args:
        confidences: 1-D array of predicted max-class probabilities.
        correct:     1-D boolean array — True if prediction was correct.
        n_buckets:   Number of equally-spaced buckets.

    Returns:
        DataFrame with columns: bucket_min, bucket_max, count, accuracy, avg_confidence.
    """
    bins = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for i in range(n_buckets):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket_min":       round(lo, 2),
            "bucket_max":       round(hi, 2),
            "count":            int(mask.sum()),
            "accuracy":         round(float(correct[mask].mean()), 4),
            "avg_confidence":   round(float(confidences[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_main_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess the main incident dataset."""
    from config import MAIN_DATA_PATH
    fpath = path or MAIN_DATA_PATH
    if not fpath.exists():
        raise FileNotFoundError(f"Main dataset not found at {fpath}")
    df = pd.read_csv(fpath, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {fpath.name}.")
    return preprocess_dataframe(df)


def load_feedback_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load user-corrected feedback rows (may not exist yet)."""
    from config import FEEDBACK_PATH
    fpath = path or FEEDBACK_PATH
    if not fpath.exists():
        logger.warning("No feedback file found — returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(fpath, low_memory=False)
    logger.info(f"Loaded {len(df):,} feedback rows from {fpath.name}.")
    return preprocess_dataframe(df)


def load_sop_rules(path: Optional[Path] = None) -> pd.DataFrame:
    """Load SOP keyword → label rules from Excel."""
    from config import SOP_RULES_PATH
    fpath = path or SOP_RULES_PATH
    if not fpath.exists():
        logger.warning("SOP rules file not found — rule engine disabled.")
        return pd.DataFrame()
    return pd.read_excel(fpath)


def filter_rare_classes(
    df: pd.DataFrame,
    col: str,
    min_samples: int,
) -> pd.DataFrame:
    """
    Remove rows whose class label in `col` has fewer than `min_samples`
    training examples (prevents degenerate classifiers).
    """
    counts = df[col].value_counts()
    valid  = counts[counts >= min_samples].index
    before = len(df)
    df     = df[df[col].isin(valid)].copy()
    logger.info(
        f"'{col}': dropped {before - len(df):,} rows with < {min_samples} samples. "
        f"Remaining classes: {len(valid)}."
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model: Any, path: Path) -> None:
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model