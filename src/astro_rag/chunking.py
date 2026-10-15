"""Module to chunk silver layer text into gold layer chunks for retrieval."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    document_id: str
    document_version: str
    page_start: int
    page_end: int
    chunking_strategy_version: str
    char_count: int


DEFAULT_METADATA_HEADER = (
    "[Documento: ORBITALIA - TCC Ciência da Computação (FAESA, 2026) | "
    "Autores/Desenvolvedores: Felipe Pereira Umpierre e Nichole da Silva Nicolini | "
    "Orientador: Prof. Howard Cruz Roatti]"
)


class ChunkingPipeline:
    """Chunks silver pages into manageable units with metadata preservation."""

    def __init__(
        self,
        silver_path: Path = Path("data/silver/pages.parquet"),
        gold_dir: Path = Path("data/gold"),
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        strategy_version: str = "contextual_metadata_v1",
        metadata_header: Optional[str] = DEFAULT_METADATA_HEADER,
    ):
        self.silver_path = Path(silver_path)
        self.gold_dir = Path(gold_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_version = strategy_version
        self.metadata_header = metadata_header

        self.gold_dir.mkdir(parents=True, exist_ok=True)

    def _split_page_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks respecting sentence or paragraph boundaries."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:].strip())
                break

            # Try to break on newline or sentence end within overlap window
            break_point = -1
            search_window = text[start + self.chunk_size - self.chunk_overlap : end]
            
            # Prefer paragraph break
            p_break = search_window.rfind("\n\n")
            if p_break != -1:
                break_point = (start + self.chunk_size - self.chunk_overlap) + p_break + 2
            else:
                # Prefer sentence period
                s_break = search_window.rfind(". ")
                if s_break != -1:
                    break_point = (start + self.chunk_size - self.chunk_overlap) + s_break + 2
                else:
                    # Prefer space
                    sp_break = search_window.rfind(" ")
                    if sp_break != -1:
                        break_point = (start + self.chunk_size - self.chunk_overlap) + sp_break + 1

            if break_point == -1 or break_point <= start:
                break_point = end

            chunk_text = text[start:break_point].strip()
            if chunk_text:
                chunks.append(chunk_text)

            start = break_point - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def process_silver_to_gold(self) -> List[TextChunk]:
        """Read silver Parquet, produce chunks, and write gold Parquet."""
        if not self.silver_path.exists():
            raise FileNotFoundError(f"Arquivo Silver não encontrado: {self.silver_path}")

        # Fetch non-empty pages ordered by page_number
        query = f"""
            SELECT document_id, content_hash, page_number, text 
            FROM read_parquet('{self.silver_path.as_posix()}')
            WHERE NOT is_empty
            ORDER BY page_number ASC
        """
        rows = duckdb.sql(query).fetchall()

        all_chunks: List[TextChunk] = []
        chunk_idx = 1

        for doc_id, doc_ver, page_num, page_text in rows:
            page_chunks = self._split_page_text(page_text)
            for ch_text in page_chunks:
                chunk_id = f"{doc_id}_c{chunk_idx:04d}"
                final_text = f"{self.metadata_header}\n{ch_text}" if self.metadata_header else ch_text
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=final_text,
                        document_id=doc_id,
                        document_version=doc_ver,
                        page_start=page_num,
                        page_end=page_num,
                        chunking_strategy_version=self.strategy_version,
                        char_count=len(final_text),
                    )
                )
                chunk_idx += 1

        # Save to Parquet
        output_path = self.gold_dir / "chunks.parquet"
        self._save_to_parquet(all_chunks, output_path)
        return all_chunks

    def _save_to_parquet(self, chunks: List[TextChunk], output_file: Path) -> Path:
        schema = pa.schema([
            ("chunk_id", pa.string()),
            ("text", pa.string()),
            ("document_id", pa.string()),
            ("document_version", pa.string()),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("chunking_strategy_version", pa.string()),
            ("char_count", pa.int32()),
        ])

        data = {
            "chunk_id": [c.chunk_id for c in chunks],
            "text": [c.text for c in chunks],
            "document_id": [c.document_id for c in chunks],
            "document_version": [c.document_version for c in chunks],
            "page_start": [c.page_start for c in chunks],
            "page_end": [c.page_end for c in chunks],
            "chunking_strategy_version": [c.chunking_strategy_version for c in chunks],
            "char_count": [c.char_count for c in chunks],
        }

        table = pa.Table.from_pydict(data, schema=schema)
        pq.write_table(table, output_file, compression="SNAPPY")
        return output_file


if __name__ == "__main__":
    pipeline = ChunkingPipeline()
    print("Iniciando divisão de páginas em chunks (Silver -> Gold)...")
    chunks = pipeline.process_silver_to_gold()
    print(f"Total de chunks gerados: {len(chunks)}")
    print(f"Arquivo salvo: data/gold/chunks.parquet")
    print(f"Estratégia: {pipeline.strategy_version}")
