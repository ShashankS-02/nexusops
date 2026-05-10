"""LLM Factory — Budget-Aware Model Selection"""
from __future__ import annotations
from langchain_core.language_models import BaseChatModel
from nexusops.config import settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return LLM based on LLM_PROVIDER env var. Defaults to gpt-4o-mini."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.LLM_MODEL or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.LLM_MODEL or "gpt-4o-mini",
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
    )


def get_embedding_model():
    """Free local embeddings via sentence-transformers (all-MiniLM-L6-v2, 384 dims)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
