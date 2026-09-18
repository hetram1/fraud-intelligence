from __future__ import annotations

from pathlib import Path

from src.graph.neo4j_client import get_driver


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs" / "graph_schema.cypher"


def load_statements() -> list[str]:
    text = SCHEMA_PATH.read_text(encoding="utf-8")

    statements = []
    for statement in text.split(";"):
        cleaned = statement.strip()

        if cleaned:
            statements.append(cleaned)

    return statements


def main() -> None:
    statements = load_statements()

    if not statements:
        raise RuntimeError("No Neo4j schema statements found")

    driver = get_driver()

    try:
        with driver.session() as session:
            for statement in statements:
                session.run(statement).consume()
                print("[OK]", statement.replace("\n", " ")[:100])

    finally:
        driver.close()

    print(f"\nApplied {len(statements)} Neo4j schema statements.")
    print("GRAPH SCHEMA INITIALIZATION: PASS")


if __name__ == "__main__":
    main()
