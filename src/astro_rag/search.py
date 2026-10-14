"""Module for Vector Search using Qdrant and FastEmbed embeddings."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COLLECTION_NAME = "astro_rag_chunks"
DEFAULT_DB_PATH = Path("data/qdrant_db")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions


class VectorStore:
    """Manages local Qdrant collection and semantic retrieval."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        model_name: str = DEFAULT_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.model_name = model_name

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.db_path))
        self.embed_model = TextEmbedding(model_name=self.model_name)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if not already existing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # all-MiniLM-L6-v2 vector dimension is 384
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )

    def index_gold_chunks(self, gold_path: Path = Path("data/gold/chunks.parquet")) -> int:
        """Read chunks from Gold Parquet, compute embeddings, and insert into Qdrant."""
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold Parquet not found: {gold_path}")

        query = f"""
            SELECT 
                chunk_id, 
                text, 
                document_id, 
                document_version, 
                page_start, 
                page_end, 
                chunking_strategy_version
            FROM read_parquet('{gold_path.as_posix()}')
        """
        rows = duckdb.sql(query).fetchall()
        if not rows:
            return 0

        texts = [r[1] for r in rows]
        print(f"Gerando embeddings para {len(texts)} chunks usando '{self.model_name}'...")
        embeddings = list(self.embed_model.embed(texts))

        points = []
        for idx, row in enumerate(rows):
            chunk_id, text, doc_id, doc_ver, p_start, p_end, strat_ver = row
            vector = embeddings[idx].tolist()
            payload = {
                "chunk_id": chunk_id,
                "text": text,
                "document_id": doc_id,
                "document_version": doc_ver,
                "page_start": p_start,
                "page_end": p_end,
                "chunking_strategy_version": strat_ver,
            }
            points.append(
                models.PointStruct(
                    id=idx + 1,  # Qdrant accepts unsigned int or UUID
                    vector=vector,
                    payload=payload,
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        print(f"Indexação concluída! {len(points)} vetores armazenados no Qdrant.")
        return len(points)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform semantic vector search using cosine similarity."""
        query_vector = list(self.embed_model.embed([query]))[0].tolist()

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for point in search_result:
            results.append({
                "score": round(point.score, 4),
                "chunk_id": point.payload.get("chunk_id"),
                "page_start": point.payload.get("page_start"),
                "page_end": point.payload.get("page_end"),
                "text": point.payload.get("text"),
                "document_id": point.payload.get("document_id"),
            })
        return results

    def close(self) -> None:
        """Close Qdrant local client cleanly."""
        try:
            self.client.close()
        except Exception:
            pass


def run_demo_search():
    """Build chunks, index into Qdrant, and run a demonstration query."""
    from astro_rag.chunking import ChunkingPipeline

    # 1. Chunk Silver to Gold
    chunker = ChunkingPipeline()
    chunker.process_silver_to_gold()

    # 2. Index into Qdrant
    vstore = VectorStore()
    vstore.index_gold_chunks()

    # 3. Demonstration inspection queries
    sample_queries = [
        "Quais são os principais métodos para detectar exoplanetas?",
        "Qual é o problema ou objetivo central abordado neste trabalho?",
    ]

    for q in sample_queries:
        print("\n" + "=" * 70)
        print(f"PERGUNTA: '{q}'")
        print("=" * 70)
        hits = vstore.search(q, top_k=2)
        for i, hit in enumerate(hits, 1):
            print(f"\n[Trecho #{i}] (Score de Similaridade: {hit['score']})")
            print(f"Origem: {hit['chunk_id']} | Página: {hit['page_start']}")
            print(f"Texto recuperado:\n{hit['text']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        vstore = VectorStore()
        hits = vstore.search(query, top_k=3)
        print(f"\nResultados para: '{query}':\n")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] Score: {h['score']} | Pág: {h['page_start']} | ID: {h['chunk_id']}")
            print(f"{h['text']}\n" + "-" * 50)
        vstore.close()
    else:
        run_demo_search()
