from unittest.mock import MagicMock
from astro_rag.rag import OllamaClient, RAGPipeline


def test_rag_build_context_prompt():
    pipeline = RAGPipeline(vector_store=MagicMock(), ollama_client=MagicMock())
    chunks = [
        {"chunk_id": "c1", "page_start": 3, "page_end": 3, "score": 0.85, "text": "Texto da página 3."},
        {"chunk_id": "c2", "page_start": 4, "page_end": 5, "score": 0.75, "text": "Texto das páginas 4 e 5."},
    ]
    prompt = pipeline._build_context_prompt(chunks)
    assert "Página 3" in prompt
    assert "Páginas 4-5" in prompt
    assert "Texto da página 3." in prompt


def test_rag_answer_grounded():
    mock_vstore = MagicMock()
    mock_vstore.search.return_value = [
        {"chunk_id": "c1", "page_start": 7, "page_end": 7, "score": 0.9, "text": "Método de trânsito."}
    ]

    mock_ollama = MagicMock()
    mock_ollama.generate_chat.return_value = "O método é o de trânsito [Página 7]."

    rag = RAGPipeline(vector_store=mock_vstore, ollama_client=mock_ollama)
    result = rag.answer("Qual método?", top_k=1, stream=False)

    assert result["query"] == "Qual método?"
    assert "trânsito" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_start"] == 7
