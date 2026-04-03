import os
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


@lru_cache(maxsize=1)
def get_pgvector_connection_string() -> str:
    raw_connection_string = os.getenv("DATABASE_URL")
    if not raw_connection_string:
        raise RuntimeError("DATABASE_URL is missing")

    if raw_connection_string.startswith("postgresql://"):
        return raw_connection_string.replace("postgresql://", "postgresql+psycopg://", 1)

    return raw_connection_string


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
    )

@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name="historical_tickets",
        connection=get_pgvector_connection_string(),
        use_jsonb=True,
    )
