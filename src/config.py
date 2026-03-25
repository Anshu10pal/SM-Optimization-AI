"""
config.py — Central configuration for Telecom Incident Classifier.

All paths, model hyper-parameters, and decision thresholds live here.
Import this module everywhere; never hard-code paths in other files.
"""

from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # telecom_incident_classifier/

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
MAIN_DATA_PATH  = DATA_DIR / "incident_enriched.csv"
FEEDBACK_PATH   = DATA_DIR / "feedback_learning.csv"
SOP_RULES_PATH  = DATA_DIR / "SOP_rules.xlsx"

# ─── Model Artefacts ──────────────────────────────────────────────────────────
MODELS_DIR              = ROOT_DIR / "models"
RESOLUTION_MODEL_PATH   = MODELS_DIR / "resolution_code_mlp.pkl"
FAULT_MODEL_PATH        = MODELS_DIR / "fault_category_mlp.pkl"
SUBCATEGORY_MODELS_DIR  = MODELS_DIR / "subcategory_models"
BM25_INDEX_PATH         = MODELS_DIR / "bm25_index.pkl"
LABEL_ENCODERS_PATH     = MODELS_DIR / "label_encoders.pkl"
EMBEDDINGS_CACHE_PATH   = MODELS_DIR / "embeddings_cache.npy"
EMBEDDINGS_CACHE_INDEX  = MODELS_DIR / "embeddings_cache_index.pkl"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGS_DIR   = ROOT_DIR / "logs"
TRAIN_LOG  = LOGS_DIR / "training.log"
INFER_LOG  = LOGS_DIR / "inference.log"

# ─── Sentence-Transformer ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256   # CPU-safe batch size for ~85 k rows
EMBEDDING_DIM        = 384   # all-MiniLM-L6-v2 output dimension

# ─── Input Columns ────────────────────────────────────────────────────────────
TEXT_COLS = ["short_description", "close_notes", "task_text"]

TARGET_RESOLUTION_CODE = "close_code"
TARGET_SUBCATEGORY     = "u_resolution_sub_category"
TARGET_FAULT_CATEGORY  = "u_fault_category"

# ─── MLP Hyper-parameters ─────────────────────────────────────────────────────
MLP_PARAMS = {
    "hidden_layer_sizes":   (512, 256, 128),
    "activation":           "relu",
    "solver":               "adam",
    "alpha":                1e-4,           # L2 regularisation — prevents overfitting
    "batch_size":           "auto",
    "learning_rate":        "adaptive",
    "learning_rate_init":   1e-3,
    "max_iter":             300,
    "early_stopping":       True,
    "validation_fraction":  0.1,
    "n_iter_no_change":     15,
    "random_state":         42,
    "verbose":              False,
}

# ─── Confidence Thresholds ────────────────────────────────────────────────────
# confidence ≥ 0.85        → Auto prediction
# 0.65 ≤ confidence < 0.85 → Medium confidence (flag for review)
# confidence < 0.65        → Needs Review
CONFIDENCE_AUTO_THRESHOLD   = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.65

# ─── BM25 ─────────────────────────────────────────────────────────────────────
BM25_TOP_K = 5   # retrieve top-5 similar historical incidents

# ─── Training Splits ──────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TEST_SIZE             = 0.15
MIN_SAMPLES_PER_CLASS = 5   # drop classes with fewer training examples

# ─── Feedback Loop ────────────────────────────────────────────────────────────
FEEDBACK_RETRAIN_THRESHOLD = 50   # trigger incremental retrain after N rows

# ─── Bootstrap directories (safe to call at import time) ──────────────────────
for _d in [DATA_DIR, MODELS_DIR, SUBCATEGORY_MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)