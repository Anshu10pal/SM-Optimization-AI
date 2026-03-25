# 📡 Telecom Incident Classifier

A production-ready, CPU-optimised AI system that classifies telecom incidents into:

- **Resolution Code** (`close_code`)
- **Resolution Sub-Category** (`u_resolution_sub_category`)
- **Fault Category** (`u_fault_category`)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
│                                                             │
│  Input Text (short_description + close_notes + task_text)  │
│         │                                                   │
│  Layer 1 ── Rule Engine (SOP keyword match)                 │
│         │   ↓ miss                                          │
│  Layer 2 ── BM25 Retrieval (top-k similar incidents)        │
│         │                                                   │
│  Layer 3 ── Sentence Embeddings (MiniLM-L6-v2, 384-dim)    │
│         │                                                   │
│  Layer 4 ── MLP Classifier (calibrated, primary model)      │
│         │                                                   │
│  Layer 5 ── Confidence Calibration (Platt scaling)          │
│         │                                                   │
│  Layer 6 ── Decision Engine                                 │
│              ├─ conf ≥ 0.85  → Auto                         │
│              ├─ conf ≥ 0.65  → Medium                       │
│              └─ conf < 0.65  → Needs Review                 │
└─────────────────────────────────────────────────────────────┘

Hierarchical Modelling:
  Stage 1: Text → Resolution Code
  Stage 2: Text + Resolution Code → Sub-Category  (per-code model)
  Stage 3: Text + Resolution Code + Sub-Category → Fault Category
```

---

## 📂 Project Structure

```
telecom_incident_classifier/
├── data/
│   ├── incident_enriched.csv       ← main training data
│   ├── feedback_learning.csv       ← user corrections (auto-created)
│   └── SOP_rules.xlsx              ← keyword rules
│
├── models/
│   ├── resolution_code_mlp.pkl
│   ├── fault_category_mlp.pkl
│   ├── label_encoders.pkl
│   ├── bm25_index.pkl
│   └── subcategory_models/
│       └── <code>.pkl
│
├── notebooks/
│   ├── 01_resolution_embeddings.ipynb
│   ├── 02_resolution_mlp.ipynb
│   ├── 03_subcategory_models.ipynb
│   ├── 04_fault_category.ipynb
|   └── 05_prediction_engine.ipynb
│
├── src/
│   ├── config.py              ← all paths & hyper-parameters
│   ├── utils.py               ← preprocessing, embeddings, BM25, helpers
│   ├── rule_engine.py         ← Layer 1: SOP keyword rules
│   ├── train_models.py        ← full training pipeline
│   ├── retrain_models.py      ← feedback-driven retraining
│   └── inference_pipeline.py  ← 6-layer prediction engine
│
├── ui_app.py                  ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / copy this project

```bash
cd telecom_incident_classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Prepare data

Place your files in the `data/` directory:
```
data/incident_enriched.csv
data/SOP_rules.xlsx          (optional — enables rule engine)
```

**Required CSV columns:**
| Column | Description |
|--------|-------------|
| `short_description` | Brief incident summary |
| `close_notes` | Resolution notes |
| `task_text` | Task details |
| `close_code` | Target: Resolution Code |
| `u_resolution_sub_category` | Target: Sub-Category |
| `u_fault_category` | Target: Fault Category |

**Optional SOP_rules.xlsx columns:**
`keyword | resolution_code | sub_category | fault_category | priority`

### Step 2 — Train models

```bash
cd src
python train_models.py
```

Training will:
1. Preprocess and combine text fields
2. Generate MiniLM-L6-v2 embeddings (cached after first run)
3. Build BM25 index
4. Train Stage 1 (Resolution Code) MLP
5. Train per-code Stage 2 (Sub-Category) MLPs
6. Train Stage 3 (Fault Category) MLP
7. Save all models to `models/`

> **Note:** First run downloads the ~90MB MiniLM model. Subsequent runs use the cache.
> Expected time: ~15–30 min for 85k rows on a modern CPU.

### Step 3 — Launch the UI

```bash
# From project root:
streamlit run ui_app.py
```

Open your browser at `http://localhost:8501`

---

## 🔁 Retraining

### Manual trigger

```bash
cd src
python retrain_models.py --force
```

### Automatic trigger (via UI)
When the feedback pool reaches **50 rows**, the sidebar shows a **Retrain Now** button.

### Threshold-based trigger

```bash
cd src
python retrain_models.py   # only retrains if ≥ 50 feedback rows
```

---

## 📊 Confidence Levels

| Label | Confidence | Action |
|-------|-----------|--------|
| ✅ Auto | ≥ 85% | Apply prediction automatically |
| ⚠️ Medium | 65–85% | Flag for human review |
| 🔴 Needs Review | < 65% | Require human decision |

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_resolution_embeddings.ipynb` | EDA, UMAP visualisation |
| `02_resolution_mlp.ipynb` | Stage 1 training & calibration analysis |
| `03_subcategory_models.ipynb` | Stage 2 per-code model evaluation |
| `04_fault_category.ipynb` | Stage 3 + end-to-end pipeline evaluation |

---

## 🔧 Configuration

All settings are in `src/config.py`:

```python
# Confidence thresholds
CONFIDENCE_AUTO_THRESHOLD   = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.65

# MLP architecture
MLP_PARAMS = {
    "hidden_layer_sizes": (512, 256, 128),
    "max_iter":           300,
    "early_stopping":     True,
    ...
}

# Feedback retraining trigger
FEEDBACK_RETRAIN_THRESHOLD = 50
```

---

## 🛡 Production Notes

- **CPU-only**: No GPU dependencies; all models run on standard hardware.
- **Reproducible**: Fixed `RANDOM_STATE=42` across all splits and models.
- **Modular**: Each layer is independently testable and replaceable.
- **Calibrated**: `CalibratedClassifierCV` with Platt scaling ensures reliable probability estimates.
- **Fault-tolerant**: Missing columns, unseen labels, and empty files are handled gracefully.
- **Feedback loop**: Corrections saved to CSV, merged before retraining.

---

## 📋 Requirements

- Python 3.9+
- 8 GB RAM (16 GB recommended for 85k rows)
- ~2 GB disk (model weights + embedding cache)

---

## 📄 License

Internal use — Telecom Operations AI Team.
