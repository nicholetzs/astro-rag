import json
import shutil
from pathlib import Path
import pytest

from astro_rag.ingestion import IngestionPipeline, calculate_sha256


@pytest.fixture
def temp_pipeline(tmp_path):
    incoming = tmp_path / "incoming"
    bronze = tmp_path / "bronze"
    manifest = bronze / "manifest.json"
    return IngestionPipeline(
        incoming_dir=incoming,
        bronze_dir=bronze,
        manifest_path=manifest,
    )


def test_calculate_sha256(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Astronomical Data RAG", encoding="utf-8")
    
    hash1 = calculate_sha256(sample_file)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex string length


def test_first_ingestion(temp_pipeline, tmp_path):
    # Create mock PDF in incoming
    test_pdf = temp_pipeline.incoming_dir / "transit.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 Mock Planetary Transit Document")

    records = temp_pipeline.ingest_incoming()
    assert len(records) == 1
    record = records[0]

    assert record.status == "ingested"
    assert record.original_filename == "transit.pdf"
    assert record.document_id.startswith("doc_")
    assert len(record.content_hash) == 64

    # Verify Bronze has the preserved file
    bronze_file = temp_pipeline.bronze_dir / "transit.pdf"
    assert bronze_file.exists()
    assert bronze_file.read_bytes() == test_pdf.read_bytes()

    # Verify manifest exists and recorded the document
    assert temp_pipeline.manifest_path.exists()
    with open(temp_pipeline.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert record.content_hash in manifest


def test_idempotence_duplicate_ingestion(temp_pipeline):
    # Ingest file first time
    test_pdf = temp_pipeline.incoming_dir / "transit.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 Mock Planetary Transit Document")

    first_records = temp_pipeline.ingest_incoming()
    assert first_records[0].status == "ingested"

    # Ingest same file a second time
    second_records = temp_pipeline.ingest_incoming()
    assert len(second_records) == 1
    second_record = second_records[0]

    # Verify idempotency
    assert second_record.status == "skipped_already_exists"
    assert second_record.content_hash == first_records[0].content_hash
    assert second_record.document_id == first_records[0].document_id

    # Verify only ONE copy of the document remains in bronze
    bronze_files = list(temp_pipeline.bronze_dir.glob("*.pdf"))
    assert len(bronze_files) == 1
