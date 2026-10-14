import sys
from pathlib import Path
import duckdb

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    sql_file = Path("query_silver.sql")
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    elif sql_file.exists():
        query = sql_file.read_text(encoding="utf-8")
    else:
        query = "SELECT * FROM read_parquet('data/silver/pages.parquet') LIMIT 5"

    try:
        print("\n=== Resultado da Consulta DuckDB SQL ===")
        duckdb.sql(query).show()
        print("=" * 40)
    except Exception as e:
        print(f"Erro ao executar SQL: {e}")

if __name__ == "__main__":
    main()
