from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class LanguageCode(str, Enum):
    """Códigos de idioma suportados"""
    PORTUGUESE = "pt"
    ENGLISH = "en"

class FlashcardRequest(BaseModel):
    """Modelo para requisição de geração de flashcards"""
    
    topic: str = Field(
        ...,
        description="Tópico ou assunto para gerar os flashcards",
        min_length=3,
        max_length=100,
        examples=["Fiber Architecture em React", "Machine Learning Basics", "Python Decorators"]
    )
    
    quantity: int = Field(
        default=5,
        description="Quantidade de flashcards a serem gerados",
        ge=1,  # greater or equal (>=1)
        le=20  # less or equal (<=20)
    )
    
    
    language: Optional[LanguageCode] = Field(
        default=LanguageCode.PORTUGUESE,
        description="Idioma para gerar os flashcards"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Fiber Architecture em React",
                "quantity": 10,          
                "language": "pt"
            }
        }
    }

class ErrorResponse(BaseModel):
    """Modelo para respostas de erro"""
    
    error: str = Field(
        ..., 
        description="Tipo do erro"
    )
    
    message: str = Field(
        ..., 
        description="Mensagem descritiva do erro"
    )
    
    details: Optional[dict] = Field(
        default=None,
        description="Detalhes adicionais sobre o erro"
    )
    
    timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp do erro em formato ISO"
    )