"""
Modelos Pydantic para a API de Flashcards

Este módulo contém todos os modelos de dados utilizados na API:
- FlashcardAdvanced: Modelo individual de flashcard
- FlashcardsResponseAdvanced: Resposta com múltiplos flashcards
- FlashcardAdvancedGenerationRequest: Dados de entrada para gerar flashcards
- ErrorResponse: Modelo para respostas de erro
- Enums: DifficultyLevel, LanguageCode
"""

from .flashcard import (
    FlashcardAdvanced,
    FlashcardsResponseAdvanced,
    DifficultyLevel
)

from .request import (
    FlashcardAdvancedGenerationRequest,
    ErrorResponse,
    LanguageCode
)

__all__ = [
    "FlashcardAdvanced",
    "FlashcardsResponseAdvanced", 
    "FlashcardAdvancedGenerationRequest",
    "ErrorResponse",
    "DifficultyLevel",
    "LanguageCode"
]