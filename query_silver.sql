-- Consulta páginas extraídas na camada Silver do astro-rag
-- Exibe metadados, contagem de caracteres e primeiras linhas do texto

SELECT 
    page_number,
    char_count,
    is_empty,
    extractor_version,
    SUBSTRING(REPLACE(text, chr(10), ' '), 1, 100) AS preview_texto
FROM read_parquet('data/silver/pages.parquet')
ORDER BY page_number ASC
LIMIT 10;
