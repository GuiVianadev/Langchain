from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from datetime import datetime

# Importa nossos modelos e serviços
from models import (
    FlashcardAdvancedGenerationRequest,
    FlashcardsResponseAdvanced,
    ErrorResponse,
    FlashcardAdvanced,
    DifficultyLevel
)
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

from services import ai_service


# Cria a instância da aplicação FastAPI
app = FastAPI(
    title="Flashcards API",
    description="API para gerar flashcards personalizados usando IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rota de health check
@app.get("/")
async def root():
    return {
        "message": "Flashcards API está funcionando!",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Rota de health check mais detalhada
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "flashcards-api",
        "version": "1.0.0",
        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
    }

@app.post("/flashcards/generate", response_model=FlashcardsResponseAdvanced)
async def generate_flashcards(request: FlashcardAdvancedGenerationRequest):
    """
    Gera flashcards personalizados usando IA baseado nos parâmetros fornecidos.
    
    Esta é a rota principal da API que usa o serviço de IA para gerar
    flashcards educativos sobre qualquer tópico.
    """
    try:
        # Chama o serviço de IA
        result = await ai_service.generate_flashcards(
            topic=request.topic,
            quantity=request.quantity,
            difficulty=request.difficulty.value,
            language=request.language.value,
            context=request.context,
            focus_areas=request.focus_areas
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Erro na geração: {result.get('error', 'Erro desconhecido')}"
            )
        
        # Converte os flashcards da IA para nosso modelo
        flashcards = []
        for ai_flashcard in result["flashcards"]:
            flashcard = FlashcardAdvanced(
                front=ai_flashcard.front,
                back=ai_flashcard.back,
                difficulty=DifficultyLevel(ai_flashcard.difficulty),
                tags=ai_flashcard.tags,
                explanation=ai_flashcard.explanation
            )
            flashcards.append(flashcard)
        
        return FlashcardsResponseAdvanced(
            topic=request.topic,
            total_generated=len(flashcards),
            flashcards=flashcards,
            generation_time_ms=result["generation_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )

# Rota para verificar status do modelo de IA
@app.get("/ai/status")
async def ai_status():
    """
    Verifica o status do serviço de IA configurado
    """
    try:
        model_info = ai_service.get_model_info()
        return {
            "ai_service": "online",
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "ai_service": "error", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Rota para validar apenas a estrutura da requisição
@app.post("/flashcards/validate")
async def validate_request(request: FlashcardsResponseAdvanced):
    """
    Valida se a requisição está corretamente formatada
    """
    return {
        "valid": True,
        "message": "Requisição válida!",
        "received_data": {
            "topic": request.topic,
            "quantity": request.quantity,
            "difficulty": request.difficulty,
            "language": request.language
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Pega as configurações do .env ou usa valores padrão
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug  # Auto-reload durante desenvolvimento
    )