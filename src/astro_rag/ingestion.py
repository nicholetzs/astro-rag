"""Module for controlled data ingestion into the bronze layer of astro-rag."""

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IngestionRecord:
    document_id: str
    content_hash: str
    source_url: Optional[str]
    original_filename: str
    ingested_at: str
    status: str
    bronze_path: Optional[str] = None


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file's content."""
    sha256_func = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256_func.update(chunk)
    return sha256_func.hexdigest()


class IngestionPipeline:
    """Pipeline to ingest raw incoming documents into the Bronze layer idempotently."""

    def __init__(
        self,
        incoming_dir: Path = Path("data/incoming"),
        bronze_dir: Path = Path("data/bronze"),
        manifest_path: Optional[Path] = None,
    ):
        self.incoming_dir = Path(incoming_dir)
        self.bronze_dir = Path(bronze_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else self.bronze_dir / "manifest.json"
        )

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, dict] = self._load_registry()

    def _load_registry(self) -> Dict[str, dict]:
        """Load the manifest of ingested documents keyed by content_hash."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_registry(self) -> None:
        """Persist the manifest to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)

    def is_already_ingested(self, content_hash: str) -> bool:
        """Check if a document with this content hash already exists in bronze."""
        return content_hash in self._registry

    def ingest_file(
        self, file_path: Path, source_url: Optional[str] = None
    ) -> IngestionRecord:
        """Ingest a single file into the bronze layer with idempotency check."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_hash = calculate_sha256(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()
        original_filename = file_path.name

        # Idempotency check: verify if content already exists
        if self.is_already_ingested(content_hash):
            existing_data = self._registry[content_hash]
            return IngestionRecord(
                document_id=existing_data["document_id"],
                content_hash=content_hash,
                source_url=existing_data.get("source_url") or source_url,
                original_filename=existing_data.get("original_filename", original_filename),
                ingested_at=timestamp,
                status="skipped_already_exists",
                bronze_path=existing_data.get("bronze_path"),
            )

        # Generate unique document_id
        document_id = f"doc_{content_hash[:12]}"

        # Copy to bronze layer (preserve original filename or add prefix if collision)
        dest_filename = original_filename
        dest_path = self.bronze_dir / dest_filename
        if dest_path.exists():
            # If filename exists with different hash, prefix short hash to preserve both
            dest_filename = f"{content_hash[:8]}_{original_filename}"
            dest_path = self.bronze_dir / dest_filename

        shutil.copy2(file_path, dest_path)

        record = IngestionRecord(
            document_id=document_id,
            content_hash=content_hash,
            source_url=source_url,
            original_filename=original_filename,
            ingested_at=timestamp,
            status="ingested",
            bronze_path=str(dest_path.as_posix()),
        )

        # Update registry
        self._registry[content_hash] = asdict(record)
        self._save_registry()

        return record

    def ingest_incoming(
        self, extension: Optional[str] = ".pdf"
    ) -> List[IngestionRecord]:
        """Scan data/incoming and ingest valid documents."""
        results: List[IngestionRecord] = []
        for item in sorted(self.incoming_dir.iterdir()):
            if not item.is_file() or item.name.startswith("."):
                continue
            if extension and item.suffix.lower() != extension.lower():
                continue

            record = self.ingest_file(item)
            results.append(record)
        return results


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    print("Iniciando ingestão da pasta data/incoming...")
    records = pipeline.ingest_incoming()
    if not records:
        print("Nenhum arquivo elegível encontrado em data/incoming.")
    else:
        for r in records:
            print(f"- [{r.status.upper()}] {r.original_filename}")
            print(f"  ID: {r.document_id}")
            print(f"  Hash SHA-256: {r.content_hash}")
            print(f"  Bronze: {r.bronze_path}")
            print(f"  Data: {r.ingested_at}\n")
