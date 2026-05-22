"""Qdrant vector store operations for NexusOps."""
from __future__ import annotations
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from nexusops.config import settings

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Create collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def store_incident_report(
    incident_id: str,
    root_cause: str,
    report: str,
    metadata: dict,
    embedder,
) -> None:
    """Embed and store a resolved incident for future RAG lookups."""
    client = get_client()
    ensure_collection(client)
    text = f"Root cause: {root_cause}\n\nReport: {report}"
    vector = embedder.embed_query(text)
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"incident_id": incident_id, "root_cause": root_cause,
                     "report": report, **metadata},
        )],
    )


def search_similar_incidents(query: str, embedder, top_k: int = 3) -> list[dict]:
    """Semantic search over past incidents. Returns empty list if no data yet."""
    client = get_client()
    ensure_collection(client)
    count = client.count(collection_name=settings.QDRANT_COLLECTION).count
    if count == 0:
        return []
    vector = embedder.embed_query(query)
    # qdrant-client ≥1.10 removed .search() — use .query_points() instead
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": r.score, "root_cause": r.payload.get("root_cause", ""),
         "report": r.payload.get("report", ""), "incident_id": r.payload.get("incident_id", "")}
        for r in response.points
    ]
