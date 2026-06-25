# Kart AI

A locally-deployed, hallucination-resistant recommendation system for e-commerce. Kart AI uses a multi-agent architecture to combine LLM reasoning with RAG and structured data analysis, ensuring every recommendation maps to a real product in the inventory.

---

## The Problem

Generative models frequently "hallucinate" — recommending products that don't exist. Kart AI addresses this by routing queries through specialized agents that only return verified, inventory-backed results.

---

## Features

- **Multi-agent routing** — classifies queries as search, analytical, or conversational (99.4% accuracy)
- **Zero hallucination** — all recommendations are grounded in real inventory data
- **Hybrid search** — semantic RAG for descriptive queries, Pandas-based filtering for price/category
- **Fully local** — no cloud APIs, runs entirely on your own hardware

---

## Architecture

Four core components:

- **LLM Router** — classifies intent and formats the final response
- **RAG Search Agent** — runs semantic similarity search over the vector database
- **Data Analyst Agent** — handles calculations, rankings, and price filtering via CSV
- **Product Knowledge Base** — stores both vector embeddings and tabular product data

---

## Tech Stack

- **LLM Runtime:** Ollama
- **Models:** Qwen 3.5 (4B) for reasoning, nomic-embed-text-v2-moe for embeddings
- **Vector DB:** ChromaDB
- **Data processing:** Pandas
- **UI:** Streamlit

---

## Setup

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com) installed.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the required models
ollama pull qwen3.5:4b
ollama pull nomic-embed-text-v2-moe

# 4. Ingest the product dataset into ChromaDB
python knowledge_base.py

# 5. Run the app
streamlit run app.py
```

---

## Citation

Developed as a Bachelor's Thesis at the German University in Cairo (GUC).  
**Author:** Hesham Amr Mohamed El-Nabawy

