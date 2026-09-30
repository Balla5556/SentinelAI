# SentinelAI — Autonomous Fraud Detection & Forensic Investigation Platform

SentinelAI is a production-style decision system that bridges the gap between predictive machine learning and agentic forensic investigation. Instead of relying on a "black box" fraud score, SentinelAI uses an ensemble of ML models to detect anomalies and a LangGraph-powered agent to investigate the "why" behind each alert.

## 🏛️ Architecture

The system follows a **Predict $\rightarrow$ Explain $\rightarrow$ Investigate $\rightarrow$ Recommend** pipeline:

1. **Predictive ML Engine**: A hybrid ensemble of **XGBoost** (supervised fraud detection) and **Isolation Forest** (unsupervised anomaly detection) calculates a fraud probability score.
2. **Explainability Layer**: **SHAP** values translate the model's numerical output into human-readable feature contributions.
3. **Agentic Investigator**: A **LangGraph** orchestrator takes the SHAP explanations and coordinates a series of forensic tools:
    - **Evidence Retriever**: Hybrid RAG (Postgres + Vector DB) to fetch customer history and transaction patterns.
    - **Policy Agent**: Cross-references the behavior against corporate compliance and fraud policy documents.
    - **Decision Agent**: Synthesizes all evidence into a structured investigation report.
4. **Human-in-the-Loop (HITL)**: The system presents a recommended action (e.g., "Hold Account") to a human analyst for final approval.

## 🛠️ Tech Stack (V1)

- **Language**: Python 3.10+
- **ML**: XGBoost, Scikit-learn, SHAP
- **LLM Orchestration**: LangGraph, LangChain
- **Vector Database**: FAISS / Qdrant
- **API**: FastAPI
- **UI**: Streamlit / React
- **MLOps**: MLflow, Docker

## 📈 Key Metrics for Evaluation

- **ML Performance**: ROC-AUC, PR-AUC, False Positive Rate (FPR).
- **Agent Fidelity**: Citation accuracy, Hallucination rate (via RAGAS), Tool-call success rate.
- **System Efficiency**: End-to-end investigation latency vs. manual analyst triage time.
