# knowledge_base.py
# Handles ChromaDB ingestion and semantic search
# Satisfies: FR2 (Grounded Product Search), NFR1 (Hallucination Resistance)

import pandas as pd
import chromadb
import ollama
from tqdm import tqdm
from config import (
    CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL,
    DATA_PATH, TOP_K_RESULTS
)

# ── ChromaDB client (no embedding function — we pass raw vectors) ──────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── Embed a batch of texts directly via ollama (no timeout issues) ─────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response.embeddings


# ── One-time ingestion ─────────────────────────────────────────────────────
def ingest_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["product_name", "description"])
    df = df.fillna("")

    collection = get_collection()

    existing = collection.count()
    if existing >= len(df):
        print(f"Knowledge base already populated ({existing} products). Skipping.")
        return

    # If partially ingested, get already-stored IDs and skip them
    already_done = set()
    if existing > 0:
        print(f"Resuming from {existing} already ingested products...")
        stored = collection.get(include=[])
        already_done = set(stored["ids"])

    print(f"Ingesting {len(df)} products into ChromaDB...")

    documents = []
    metadatas = []
    ids = []

    print("Preparing documents...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building text"):
        uid = str(idx)
        if uid in already_done:
            continue

        combined_text = (
            f"Product: {row['product_name']}\n"
            f"Category: {row['primary_category']}\n"
            f"Brand: {row['brand']}\n"
            f"Description: {row['description']}\n"
            f"Specs: {row['specifications_text']}"
        )

        metadata = {
            "product_name": str(row["product_name"]),
            "brand": str(row["brand"]),
            "main_category": str(row["primary_category"]),
            "retail_price": float(row["retail_price"]) if row["retail_price"] != "" else 0.0,
            "discounted_price": float(row["discounted_price"]) if row["discounted_price"] != "" else 0.0,
        }

        documents.append(combined_text)
        metadatas.append(metadata)
        ids.append(uid)

    if not documents:
        print("Nothing new to ingest.")
        return

    # Embed + upsert in small batches of 25
    batch_size = 25
    total = len(documents)

    with tqdm(total=total, desc="Embedding & uploading", unit="products") as pbar:
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)

            batch_docs = documents[i:batch_end]
            batch_meta = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]

            # Retry logic
            for attempt in range(5):
                try:
                    embeddings = embed_batch(batch_docs)
                    collection.upsert(
                        documents=batch_docs,
                        embeddings=embeddings,
                        metadatas=batch_meta,
                        ids=batch_ids,
                    )
                    break
                except Exception as e:
                    if attempt < 4:
                        tqdm.write(f"  Retry {attempt+1}/5 on batch {i}-{batch_end}: {e}")
                    else:
                        tqdm.write(f"  Skipping batch {i}-{batch_end} after 5 attempts: {e}")

            pbar.update(batch_end - i)

    print(f"\nIngestion complete. {collection.count()} products stored.")


# ── Semantic search ────────────────────────────────────────────────────────
def search_products(query: str, n_results: int = TOP_K_RESULTS) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_batch([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    products = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        products.append({
            "product_name": meta.get("product_name", "Unknown"),
            "brand": meta.get("brand", ""),
            "main_category": meta.get("main_category", ""),
            "retail_price": meta.get("retail_price", 0.0),
            "discounted_price": meta.get("discounted_price", 0.0),
            "relevance_score": round(1 - results["distances"][0][i], 3),
        })

    return products


if __name__ == "__main__":
    ingest_data()