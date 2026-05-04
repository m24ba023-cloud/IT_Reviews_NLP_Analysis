# IT Customer Reviews — NLP Analytics Dashboard

## Live Dashboard
🚀 **[Open Dashboard](YOUR_STREAMLIT_LINK_HERE)**

## Project Overview
Complete NLP pipeline for IT customer review sentiment analysis
with an interactive Streamlit dashboard.

## Pipeline (Updated)
| Phase | Components |
|-------|-----------|
| Phase 1 | Data loading + SMOTE balancing |
| Phase 2 | Lemmatization + Bigrams/N-grams |
| Phase 3 | TextBlob + VADER + CSS score |
| Phase 4 | Sentence-BERT + SMOTE + HDBSCAN |
| Phase 5 | Word clouds + Charts + Scatter |
| Phase 6 | Word2Vec + RoBERTa fine-tuned |

## Model Results
| Model | Accuracy |
|-------|----------|
| TextBlob | 42% |
| VADER | 40% |
| LSTM | 44.33% |
| BiLSTM (baseline) | 67% |
| BiLSTM + GloVe + Aug | 74.85% |
| **RoBERTa fine-tuned** | **87%+** |

## Key Improvements
- ✅ Stemming removed — Only lemmatization
- ✅ Bigrams + Trigrams for better features
- ✅ TF-IDF replaced by Sentence-BERT embeddings
- ✅ K-Means replaced by HDBSCAN clustering
- ✅ SMOTE for class imbalance
- ✅ RoBERTa fine-tuned on IT reviews

## Tech Stack
Python • PyTorch • HuggingFace Transformers
Sentence-BERT • HDBSCAN • SMOTE
Streamlit • Plotly • NLTK • Gensim

## How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
