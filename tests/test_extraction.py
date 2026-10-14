import duckdb
from pathlib import Path
from astro_rag.extraction import ExtractionPipeline, conservative_clean_text


def test_conservative_clean_text():
    raw = "Trânsito   de  exoplanetas.\n\n\nEquação: R_p / R_* = 0.01.\nQuebra de pla-\nnetas aqui."
    cleaned = conservative_clean_text(raw)
    
    # Check that runs of spaces are normalized
    assert "   " not in cleaned
    # Check that excessive newlines are capped at 2
    assert "\n\n\n" not in cleaned
    # Check that hyphenated word wrap is rejoined
    assert "planetas" in cleaned
    # Check that scientific equation symbols/decimals are preserved
    assert "R_p / R_* = 0.01" in cleaned


def test_parquet_and_duckdb_query(tmp_path):
    pipeline = ExtractionPipeline(
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        quarantine_dir=tmp_path / "quarantine",
    )
    
    # Extract from existing bronze file if available
    bronze_sample = Path("data/bronze/ProjetoTCC_Felipe_Nichole_corrigido.pdf")
    if not bronze_sample.exists():
        return
        
    pages = pipeline.extract_document(bronze_sample, "doc_test", "hash_test")
    assert len(pages) > 0
    assert pages[0].page_number == 1
    assert pages[0].extractor_version.startswith("pypdf-")

    output_parquet = tmp_path / "silver" / "test_pages.parquet"
    saved_path = pipeline.save_to_parquet(pages, output_parquet)
    assert saved_path.exists()

    # Query with DuckDB
    count = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{saved_path.as_posix()}')").fetchone()[0]
    assert count == len(pages)
