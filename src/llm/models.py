"""
LLM Configuration and Setup
============================

Language model configurations for different buckets.
"""

from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    BIG_LLM_MODEL
)


def get_small_llm():
    """
    Get Small LLM for BUCKET_B (RAG queries)
    
    Returns:
        ChatGroq: Groq LLM instance
    """
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        groq_api_key=GROQ_API_KEY
    )


def get_big_llm():
    """
    Get Big LLM for BUCKET_C (escalation)
    
    Returns:
        ChatGroq: Groq LLM instance
    """
    return ChatGroq(
        model=BIG_LLM_MODEL,
        temperature=0.3,
        max_tokens=1000,
        groq_api_key=GROQ_API_KEY
    )


class LLMFactory:
    """Factory for creating LLM instances based on bucket"""
    
    _small_llm = None
    _big_llm = None
    
    @classmethod
    def get_llm_for_bucket(cls, bucket: str):
        """
        Get appropriate LLM for bucket
        
        Args:
            bucket: BUCKET_A, BUCKET_B, or BUCKET_C
            
        Returns:
            ChatOpenAI instance or None for BUCKET_A
        """
        if bucket == 'BUCKET_A':
            return None  # No LLM needed
        
        elif bucket == 'BUCKET_B':
            if cls._small_llm is None:
                cls._small_llm = get_small_llm()
            return cls._small_llm
        
        elif bucket == 'BUCKET_C':
            if cls._big_llm is None:
                cls._big_llm = get_big_llm()
            return cls._big_llm
        
        else:
            raise ValueError(f"Unknown bucket: {bucket}")
    
    @classmethod
    def reset(cls):
        """Reset cached LLM instances"""
        cls._small_llm = None
        cls._big_llm = None
