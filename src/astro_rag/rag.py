"""RAG Pipeline: Connects Qdrant vector retrieval with local Ollama generator (LLM)."""

import json
import os
import sys
from typing import Any, Dict, Generator, List, Optional

import requests

from astro_rag.search import VectorStore

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


class OllamaClient:
    """Client for local Ollama REST API."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        stream: bool = False,
    ) -> Any:
        """Send chat prompt to Ollama."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": stream,
        }

        if stream:
            resp = requests.post(url, json=payload, stream=True, timeout=120)
            resp.raise_for_status()

            def response_generator() -> Generator[str, None, None]:
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content

            return response_generator()
        else:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


class RAGPipeline:
    """End-to-end RAG orchestrator combining VectorStore and Ollama LLM."""

    SYSTEM_PROMPT = """Você é um assistente acadêmico e técnico especializado em analisar documentos e TCCs.
Sua missão é responder à pergunta do usuário baseando-se ESTRITAMENTE nas evidências apresentadas no contexto fornecido.

Regras e Guardrails:
1. Responda apenas com informações contidas no contexto abaixo. NÃO invente nem deduza fatos que não constem no texto.
2. Se a informação necessária para responder NÃO estiver presente nos trechos, responda honestamente:
   "Com base no documento fornecido, não encontrei informações suficientes para responder a essa pergunta."
3. Sempre cite as páginas de referência dos trechos utilizados para fundamentar sua resposta (exemplo: "[Página 12]").
4. Responda em português, de forma clara, objetiva e bem estruturada.
"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.ollama = ollama_client or OllamaClient()

    def _build_context_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "Nenhum trecho relevante foi encontrado no documento."

        context_blocks = []
        for i, c in enumerate(chunks, 1):
            p_start = c.get("page_start", "?")
            p_end = c.get("page_end", p_start)
            page_info = f"Página {p_start}" if p_start == p_end else f"Páginas {p_start}-{p_end}"
            score = c.get("score", 0.0)

            context_blocks.append(
                f"--- [Trecho #{i} | {page_info} | Similaridade: {score}] ---\n"
                f"{c.get('text', '').strip()}"
            )
        return "\n\n".join(context_blocks)

    def answer(
        self,
        query: str,
        top_k: int = 4,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Perform retrieval and generate grounded response."""
        # 1. Retrieval
        retrieved_chunks = self.vector_store.search(query, top_k=top_k)

        # 2. Context assembly
        context_text = self._build_context_prompt(retrieved_chunks)
        user_prompt = f"""Contexto extraído do documento:
{context_text}

Pergunta do usuário:
{query}
"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 3. Generation
        if stream:
            stream_gen = self.ollama.generate_chat(messages, stream=True)
            return {
                "query": query,
                "sources": retrieved_chunks,
                "stream": stream_gen,
            }
        else:
            answer_text = self.ollama.generate_chat(messages, stream=False)
            return {
                "query": query,
                "answer": answer_text,
                "sources": retrieved_chunks,
            }

    def close(self) -> None:
        """Close resources cleanly."""
        if self.vector_store:
            self.vector_store.close()


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "O que é a plataforma Orbitalia e quais são seus objetivos?"

    print("\n" + "=" * 70)
    print(f"🚀 Astro-RAG com Ollama (Llama 3.1)")
    print(f"❓ Pergunta: {query}")
    print("=" * 70)

    rag = RAGPipeline()

    if not rag.ollama.is_available():
        print(f"\n❌ Erro: Ollama não está acessível em {rag.ollama.base_url}")
        print("Certifique-se de que o Ollama está rodando no seu computador.")
        return

    print("\n🔍 1. Recuperando trechos relevantes no Qdrant...")
    result = rag.answer(query, top_k=3, stream=True)

    print("\n🤖 2. Resposta gerada pelo Llama 3.1:\n")
    for token in result["stream"]:
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")

    print("=" * 70)
    print("📚 Fontes consultadas:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] Página {src.get('page_start')} (Score: {src.get('score')}) - ID: {src.get('chunk_id')}")
    print("=" * 70)

    rag.close()


if __name__ == "__main__":
    main()
