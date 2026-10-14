from pathlib import Path
import pytest
from astro_rag.chunking import ChunkingPipeline
from astro_rag.search import VectorStore


def test_chunking_pipeline(tmp_path):
    silver_sample = Path("data/silver/pages.parquet")
    if not silver_sample.exists():
        pytest.skip("Silver parquet not generated yet")

    gold_dir = tmp_path / "gold"
    pipeline = ChunkingPipeline(
        silver_path=silver_sample,
        gold_dir=gold_dir,
        chunk_size=500,
        chunk_overlap=50,
        strategy_version="test_v1",
    )
    chunks = pipeline.process_silver_to_gold()
    assert len(chunks) > 0
    first = chunks[0]
    assert first.chunk_id.startswith("doc_")
    assert first.page_start >= 1
    assert first.chunking_strategy_version == "test_v1"
    assert (gold_dir / "chunks.parquet").exists()


def test_vector_search_hit():
    qdrant_dir = Path("data/qdrant_db")
    if not qdrant_dir.exists():
        pytest.skip("Qdrant db not initialized yet")

    vstore = VectorStore(db_path=qdrant_dir)
    hits = vstore.search("exoplanetas e trânsito planetário", top_k=1)
    assert len(hits) == 1
    assert hits[0]["score"] > 0.4
    assert "exoplaneta" in hits[0]["text"].lower() or "trânsito" in hits[0]["text"].lower()
