from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD is not set in .env")


def get_driver() -> Driver:
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )


def verify_connection() -> str:
    driver = get_driver()

    try:
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection successful' AS message")
            record = result.single()

            if record is None:
                raise RuntimeError("Neo4j returned no result")

            return record["message"]
    finally:
        driver.close()


if __name__ == "__main__":
    print(verify_connection())
