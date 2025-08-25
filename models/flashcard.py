from pydantic import BaseModel, Field
from typing import List, Optional

class Flashcard(BaseModel):  
    front: str = Field(
        ..., 
        description="Pergunta ou conceito no front do flashcard",
        min_length=5,
        max_length=200
    )
    
    back: str = Field(
        ..., 
        description="Resposta ou explicação no verso do flashcard",
        min_length=10,
        max_length=500
    )
    
    difficulty: str = Field(
        description="Nível de dificuldade do flashcard"
    )
    
    suggestions: Optional[str] = Field(
        default=None,
        description="Sugestão de conteudos para ser estudado",
        max_length=300
    )



class FlashcardsResponse(BaseModel):
    topic: str = Field(
        ..., 
        description="Tópico dos flashcards gerados"
    )
    
    total_generated: int = Field(
        ..., 
        description="Número total de flashcards gerados"
    )
    
    flashcards: List[Flashcard] = Field(
        ..., 
        description="Lista de flashcards gerados"
    )
    
    generation_time_ms: Optional[int] = Field(
        default=None,
        description="Tempo de geração em milissegundos"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Fiber Architecture em React",
                "total_generated": 2,
                "flashcards": [
                    {
                        "front": "O que é Fiber Architecture no React?",
                        "back": "Fiber é a nova arquitetura de reconciliação do React que permite renderização incremental e priorização de atualizações.",
                        "difficulty": "fácil",
                        "suggestion": "Introduzido no React 16, permite melhor performance em aplicações complexas."
                    },
                    {
                        "front": "Qual é o principal benefício do Fiber?",
                        "back": "Permite que o React pause, aborte ou reutilize trabalho, resultando em melhor experiência do usuário.",
                    }
                ],
                "generation_time_ms": 1500
            }
        }
    }