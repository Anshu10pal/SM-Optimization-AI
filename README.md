# 🧠 SM Optimization AI

AI-based system to predict:
- Resolution Code
- Resolution Sub-Category
- Fault Category

## 🚀 Features
- Embedding-based semantic understanding
- MLP classification
- Confidence-based decisioning
- Needs Review fallback
- Bulk Excel prediction
- Feedback learning loop

## 🏗 Architecture
Rule Engine → BM25 → Embeddings → MLP → Confidence → Decision

## 📊 Goal
- 95%+ accuracy on high-confidence predictions
- Reduce manual correction effort

## ⚙️ Tech Stack
- Python
- Sentence Transformers
- Scikit-learn
- Streamlit

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run src/ui_app.py
