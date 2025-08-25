"""
Serviços da API de Flashcards

Este módulo contém os serviços de integração com IA e outras funcionalidades:
- AIService: Integração com modelos de IA para geração de flashcards
"""

from .ai_service import AIServiceAdvanced, ai_service

__all__ = [
    "AIServiceAdvanced",
    "ai_service"
]