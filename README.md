# 🪐 Astro-RAG: Pipeline de Recuperação e Busca Semântica Vetorial para Astronomia

Pipeline de Engenharia de Dados e Recuperação Aumentada por Geração (RAG) desenvolvido sobre documentos astronômicos e Trabalhos de Conclusão de Curso (TCC) voltados à detecção de exoplanetas por curvas de luz (Projeto **Orbitalia**).

---

## 🏛️ Arquitetura em Camadas (Medallion)

O pipeline foi estruturado seguindo as melhores práticas de dados:

```text
astro-rag/
├── data/
│   ├── incoming/      # Ponto de entrada de documentos brutos (PDFs)
│   ├── bronze/        # Preservação idempotente com hash SHA-256 e manifest.json
│   ├── silver/        # Texto extraído por página em formato Parquet (consultável via DuckDB SQL)
│   ├── gold/          # Chunks estruturados com metadados para busca vetorial
│   └── quarantine/    # Registros e logs de documentos com falhas de processamento
├── src/astro_rag/     # Módulos Python (ingestion, extraction, chunking, search)
├── tests/             # Suíte de testes automatizados com pytest
└── query_silver.sql   # Consultas analíticas SQL via DuckDB
```

---

## 🚀 Funcionalidades

1. **Ingestão Idempotente (Camada Bronze):**
   - Cálculo de hash SHA-256 para evitar duplicatas.
   - Registro detalhado em manifesto (`document_id`, `content_hash`, `ingested_at`, `status`).
2. **Extração Conservadora (Camada Silver):**
   - Extração página por página preservando símbolos matemáticos, unidades e fórmulas.
   - Gravação colunar em **Parquet**.
   - Consultas SQL instantâneas sem servidor utilizando **DuckDB**.
3. **Chunking e Busca Semântica Vetorial (Camada Gold + Qdrant):**
   - Divisão inteligente de texto respeitando frases e parágrafos.
   - Geração de embeddings vetoriais com **FastEmbed** (`all-MiniLM-L6-v2`).
   - Armazenamento e busca semântica por similaridade de cosseno com **Qdrant**.

---

## 🛠️ Instalação e Uso

### 1. Ambiente Virtual e Dependências
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pypdf pyarrow duckdb qdrant-client fastembed
```

### 2. Executar o Pipeline

```powershell
$env:PYTHONPATH="src"

# 1. Ingestão Bronze
python -m astro_rag.ingestion

# 2. Extração Silver (Parquet)
python -m astro_rag.extraction

# 3. Chunking Gold e Indexação no Qdrant
python -m astro_rag.search
```

### 3. Consultas SQL com DuckDB
```powershell
python query.py
```

### 4. Busca Semântica no TCC
```powershell
$env:PYTHONPATH="src"
python -m astro_rag.search "o que é a plataforma orbitalia?"
```

---

## 🧪 Testes Automatizados
```powershell
$env:PYTHONPATH="src"
pytest tests
```
