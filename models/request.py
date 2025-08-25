from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum
from .flashcard import DifficultyLevel

class LanguageCode(str, Enum):
    """Códigos de idioma suportados"""
    PORTUGUESE = "pt"
    ENGLISH = "en"
    SPANISH = "es"

class FlashcardAdvancedGenerationRequest(BaseModel):
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
    
    difficulty: Optional[DifficultyLevel] = Field(
        default=DifficultyLevel.INTERMEDIATE,
        description="Nível de dificuldade desejado para os flashcards"
    )
    
    language: Optional[LanguageCode] = Field(
        default=LanguageCode.PORTUGUESE,
        description="Idioma para gerar os flashcards"
    )
    
    context: Optional[str] = Field(
        default=None,
        description="Contexto adicional ou instruções específicas",
        max_length=500
    )
    
    include_explanations: Optional[bool] = Field(
        default=True,
        description="Se deve incluir explicações adicionais nos flashcards"
    )
    
    focus_areas: Optional[List[str]] = Field(
        default=[],
        description="Áreas específicas para focar dentro do tópico",
        max_items=5
    )
    
    @validator('topic')
    def validate_topic(cls, v):
        """Valida e limpa o tópico"""
        if not v or v.isspace():
            raise ValueError('Tópico não pode estar vazio')
        
        # Remove espaços extras
        v = ' '.join(v.split())
        
        # Verifica se não contém apenas números
        if v.isdigit():
            raise ValueError('Tópico deve conter texto, não apenas números')
            
        return v
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        """Valida as áreas de foco"""
        if v:
            # Remove itens vazios e duplicados
            v = list(set([area.strip() for area in v if area and area.strip()]))
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Fiber Architecture em React",
                "quantity": 10,
                "difficulty": "intermediate",
                "language": "pt",
                "context": "Focado em conceitos práticos para desenvolvedores front-end",
                "include_explanations": True,
                "focus_areas": ["reconciliation", "performance", "virtual DOM"]
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