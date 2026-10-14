"""Module to extract per-page text from Bronze PDFs into Silver Parquet format."""

import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pypdf


@dataclass
class ExtractedPage:
    document_id: str
    content_hash: str
    original_filename: str
    page_number: int  # 1-indexed
    text: str
    char_count: int
    is_empty: bool
    extractor_version: str
    extracted_at: str


def conservative_clean_text(raw_text: str) -> str:
    """Conservatively clean extracted text.

    Preserves scientific symbols, signs, formulas, exponents, and units.
    Only normalizes excessive whitespace runs and irregular line wraps.
    """
    if not raw_text:
        return ""

    # Replace null bytes or non-printable control chars, keeping standard whitespace
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_text)

    # Normalize Windows and Mac line endings to standard Unix \n
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin hyphenated line breaks (e.g., 'pla-\nnetas' -> 'planetas')
    # only when it's clearly word-hyphenation (letters on both sides)
    cleaned = re.sub(r"([a-zA-Zá-úÁ-Ú])-\n([a-zA-Zá-úÁ-Ú])", r"\1\2", cleaned)

    # Normalize runs of horizontal spaces (spaces and tabs) to a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Normalize excessive consecutive newlines (more than 2 newlines -> 2 newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


class ExtractionPipeline:
    """Pipeline that reads bronze PDFs, extracts pages conservatively, and saves to Silver Parquet."""

    def __init__(
        self,
        bronze_dir: Path = Path("data/bronze"),
        silver_dir: Path = Path("data/silver"),
        quarantine_dir: Path = Path("data/quarantine"),
    ):
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.quarantine_dir = Path(quarantine_dir)

        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_version = f"pypdf-{pypdf.__version__}"

    def extract_document(
        self,
        pdf_path: Path,
        document_id: str,
        content_hash: str,
    ) -> List[ExtractedPage]:
        """Extract pages from a PDF. If extraction fails, send to quarantine."""
        pages_data: List[ExtractedPage] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                except Exception:
                    raw_text = ""

                cleaned_text = conservative_clean_text(raw_text)
                is_empty = len(cleaned_text.strip()) == 0

                pages_data.append(
                    ExtractedPage(
                        document_id=document_id,
                        content_hash=content_hash,
                        original_filename=pdf_path.name,
                        page_number=page_num,
                        text=cleaned_text,
                        char_count=len(cleaned_text),
                        is_empty=is_empty,
                        extractor_version=self.extractor_version,
                        extracted_at=now,
                    )
                )

        except Exception as err:
            # Route failure to quarantine
            self._quarantine_document(pdf_path, document_id, content_hash, str(err))
            raise

        return pages_data

    def _quarantine_document(
        self, pdf_path: Path, document_id: str, content_hash: str, error_msg: str
    ) -> None:
        """Log failure and move/copy issue description into quarantine."""
        log_file = self.quarantine_dir / f"error_{document_id}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Document ID: {document_id}\n")
            f.write(f"Original File: {pdf_path.name}\n")
            f.write(f"Content Hash: {content_hash}\n")
            f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")

    def save_to_parquet(
        self, pages: List[ExtractedPage], output_file: Optional[Path] = None
    ) -> Path:
        """Save extracted pages into a columnar Parquet file."""
        if not pages:
            raise ValueError("No pages to save.")

        if output_file is None:
            output_file = self.silver_dir / "pages.parquet"

        # Define explicit PyArrow schema
        schema = pa.schema([
            ("document_id", pa.string()),
            ("content_hash", pa.string()),
            ("original_filename", pa.string()),
            ("page_number", pa.int32()),
            ("text", pa.string()),
            ("char_count", pa.int32()),
            ("is_empty", pa.bool_()),
            ("extractor_version", pa.string()),
            ("extracted_at", pa.string()),
        ])

        data = {
            "document_id": [p.document_id for p in pages],
            "content_hash": [p.content_hash for p in pages],
            "original_filename": [p.original_filename for p in pages],
            "page_number": [p.page_number for p in pages],
            "text": [p.text for p in pages],
            "char_count": [p.char_count for p in pages],
            "is_empty": [p.is_empty for p in pages],
            "extractor_version": [p.extractor_version for p in pages],
            "extracted_at": [p.extracted_at for p in pages],
        }

        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, output_file, compression="SNAPPY")
        return output_file


def run_silver_pipeline():
    """Main execution of the Bronze to Silver extraction pipeline."""
    import json

    manifest_path = Path("data/bronze/manifest.json")
    if not manifest_path.exists():
        print("Manifesto data/bronze/manifest.json não encontrado. Execute a ingestão primeiro.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = ExtractionPipeline()
    all_pages: List[ExtractedPage] = []

    for content_hash, doc_info in manifest.items():
        bronze_path = Path(doc_info["bronze_path"])
        if not bronze_path.exists():
            print(f"Aviso: Arquivo {bronze_path} não encontrado na Bronze.")
            continue

        print(f"Extraindo: {doc_info['original_filename']} ({doc_info['document_id']})...")
        pages = pipeline.extract_document(
            pdf_path=bronze_path,
            document_id=doc_info["document_id"],
            content_hash=content_hash,
        )
        all_pages.extend(pages)
        print(f"  -> {len(pages)} páginas processadas.")

    if all_pages:
        parquet_path = pipeline.save_to_parquet(all_pages)
        print(f"\nSalvo com sucesso na Silver: {parquet_path}")

        # DuckDB summary query
        print("\n=== Resumo DuckDB da Camada Silver ===")
        duckdb.sql(f"""
            SELECT 
                COUNT(*) as total_paginas,
                SUM(CASE WHEN is_empty THEN 1 ELSE 0 END) as paginas_vazias,
                SUM(CASE WHEN NOT is_empty THEN 1 ELSE 0 END) as paginas_com_texto,
                AVG(char_count)::INT as media_caracteres_por_pagina,
                MIN(char_count) as min_caracteres,
                MAX(char_count) as max_caracteres
            FROM read_parquet('{parquet_path.as_posix()}')
        """).show()


if __name__ == "__main__":
    run_silver_pipeline()
