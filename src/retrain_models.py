import pandas as pd
import numpy as np
import re
import os
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

# =========================
# PATHS
# =========================
DATA_DIR = r"D:\ML\SM Modeling\Data"
RAW_DIR = DATA_DIR + r"\Raw"
MASTER_DIR = DATA_DIR + r"\Master"

INCIDENT_FILE = RAW_DIR + r"\incident_enriched.csv"
FEEDBACK_FILE = RAW_DIR + r"\feedback_learning.csv"

# Output models
RESCODE_LR = MASTER_DIR + r"\rescode_logistic.pkl"
RESCODE_MLP = MASTER_DIR + r"\rescode_mlp.pkl"

SUBCAT_MODEL = MASTER_DIR + r"\subcat_logistic.pkl"
FAULTCAT_MODEL = MASTER_DIR + r"\faultcat_logistic.pkl"

# =========================
# UTILS
# =========================
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_text(df):
    return (
        df["short_description"] + " " +
        df["close_notes"] + " " +
        (df["task_text"] + " ") * 3
    ).apply(normalize)

# =========================
# LOAD BASE DATA (IMMUTABLE)
# =========================
print("Loading enriched incidents...")
df_base = pd.read_csv(INCIDENT_FILE, encoding="latin1")
df_base.columns = df_base.columns.str.strip()

for c in ["short_description", "close_notes", "task_text"]:
    df_base[c] = df_base[c].fillna("").astype(str)

df_base["text"] = build_text(df_base)

print("Base incidents:", len(df_base))

# =========================
# LOAD FEEDBACK (OVERRIDES)
# =========================
if os.path.exists(FEEDBACK_FILE):
    print("Loading feedback data...")
    fb = pd.read_csv(FEEDBACK_FILE)
    fb.columns = fb.columns.str.strip()
    fb["text"] = fb["text"].fillna("").astype(str).apply(normalize)
else:
    print("No feedback file found — training only on base data.")
    fb = pd.DataFrame()

# =========================
# -------- RESOLUTION CODE
# =========================
print("\nTraining Resolution Code models...")

train_res_base = df_base[["text", "close_code"]].rename(
    columns={"close_code": "label"}
)
train_res_base = train_res_base[train_res_base["label"].notna()]

train_res_fb = pd.DataFrame()
if not fb.empty and "correct_res_code" in fb.columns:
    train_res_fb = fb[fb["correct_res_code"].notna()][["text", "correct_res_code"]] \
        .rename(columns={"correct_res_code": "label"})

train_res = pd.concat([train_res_base, train_res_fb], ignore_index=True)
train_res = train_res[train_res["label"].str.len() > 0]

X = train_res["text"]
y = train_res["label"]

# Logistic (calibrated)
base_lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1500))
])

rescode_lr = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
rescode_lr.fit(X, y)

# MLP (secondary validator)
rescode_mlp = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2)
    )),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=20,
        random_state=42
    ))
])

rescode_mlp.fit(X, y)

joblib.dump(rescode_lr, RESCODE_LR)
joblib.dump(rescode_mlp, RESCODE_MLP)

print("✅ Resolution Code models saved")

# =========================
# -------- SUB CATEGORY
# =========================
print("\nTraining Sub-Category model...")

train_sub_base = df_base[df_base["u_resolution_sub_category"].notna()].copy()
train_sub_base["label"] = train_sub_base["u_resolution_sub_category"]

train_sub_fb = pd.DataFrame()
if not fb.empty and "correct_subcat" in fb.columns:
    train_sub_fb = fb[fb["correct_subcat"].notna()][["text", "correct_subcat"]] \
        .rename(columns={"correct_subcat": "label"})

train_sub = pd.concat([
    train_sub_base[["text", "label"]],
    train_sub_fb
], ignore_index=True)

train_sub = train_sub[train_sub["label"].str.len() > 0]

subcat_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1500))
])

subcat_model.fit(train_sub["text"], train_sub["label"])
joblib.dump(subcat_model, SUBCAT_MODEL)

print("✅ Sub-Category model saved")

# =========================
# -------- FAULT CATEGORY
# =========================
print("\nTraining Fault Category model...")

train_fault_base = df_base[df_base["u_fault_category"].notna()].copy()
train_fault_base["label"] = train_fault_base["u_fault_category"]

train_fault_fb = pd.DataFrame()
if not fb.empty and "correct_faultcat" in fb.columns:
    train_fault_fb = fb[fb["correct_faultcat"].notna()][["text", "correct_faultcat"]] \
        .rename(columns={"correct_faultcat": "label"})

train_fault = pd.concat([
    train_fault_base[["text", "label"]],
    train_fault_fb
], ignore_index=True)

train_fault = train_fault[train_fault["label"].str.len() > 0]

faultcat_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=7000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1500))
])

faultcat_model.fit(train_fault["text"], train_fault["label"])
joblib.dump(faultcat_model, FAULTCAT_MODEL)

print("✅ Fault Category model saved")

# =========================
# DONE
# =========================
print("\n======================================")
print("🎉 RETRAINING COMPLETED SUCCESSFULLY")
print("Timestamp:", datetime.now())
print("======================================")

