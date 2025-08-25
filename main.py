from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from services.ai_service import ai_service
from datetime import datetime
import os

from models import (
    FlashcardRequest,
    FlashcardsResponse,
    Flashcard,
)

load_dotenv()




app = FastAPI(
    title="Flashcards API",
    description="API para gerar flashcards personalizados usando IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Flashcards API está funcionando!",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "flashcards-api",
        "version": "1.0.0",
        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
    }


@app.post("/flashcards/generate", response_model=FlashcardsResponse)
async def generate_flashcards(request: FlashcardRequest):
    """
    Gera flashcards personalizados usando IA baseado nos parâmetros fornecidos.
    """
    try:
        result = await ai_service.generate_flashcards_with_metadata(
            topic=request.topic,
            quantity=request.quantity,
            language=request.language.value, 
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Erro na geração: {result.get('error', 'Erro desconhecido')}"
            )
  
        flashcards = []
        for ai_flashcard in result["flashcards"]:
            flashcard = Flashcard(
                front=ai_flashcard.front,
                back=ai_flashcard.back,
                difficulty=ai_flashcard.difficulty,
                suggestions=ai_flashcard.suggestions  
            )
            flashcards.append(flashcard)
        
        return FlashcardsResponse(
            topic=request.topic,
            total_generated=len(flashcards),
            flashcards=flashcards,
            generation_time_ms=result["generation_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"ERRO DETALHADO: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )
    
if __name__ == "__main__":
    import uvicorn
    
    
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug 
    )