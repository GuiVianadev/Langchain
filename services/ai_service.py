import os
import asyncio
from typing import List, Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlashcardAI(BaseModel):
    """Modelo para um flashcard gerado pela IA"""
    front: str = Field(description="Pergunta ou conceito no front do flashcard")
    back: str = Field(description="Resposta ou explicação no verso do flashcard")
    difficulty: str = Field(description="Nível de dificuldade: beginner, intermediate, advanced")
    suggestions: Optional[str] = Field(description="Sugestão de estudo", default=None)

class FlashcardsGeneration(BaseModel):
    """Modelo para múltiplos flashcards gerados pela IA"""
    flashcards: List[FlashcardAI] = Field(description="Lista de flashcards gerados")


class AIService:
    """Serviço para gerar flashcards com IA"""

    def _initialize_model(self):
        """Inicializa o modelo de IA baseado no provider configurado"""
        try:
            if self.provider == "openai":
                api_key = load_dotenv()
                if not api_key:
                    raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")
                
                return init_chat_model(
                    "gpt-4o-mini",  
                    model_provider="openai",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            raise ValueError(f"Provider {self.provider} não suportado")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {e}")
            raise
    
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "openai")
        self.model_name = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1500"))
        self.max_flashcards = int(os.getenv("MAX_FLASHCARDS_PER_REQUEST", "15"))
        self.min_flashcards = int(os.getenv("MIN_FLASHCARDS_PER_REQUEST", "1"))

        self.llm = self._initialize_model()

        if self.llm:
            self.structured_llm = self.llm.with_structured_output(FlashcardsGeneration)
        else:
            self.structured_llm = None

    
    def _create_prompt(self, topic: str, quantity: int, language: str) -> ChatPromptTemplate:
        """Cria o prompt otimizado para geração de flashcards"""
        
        # Mapeia códigos de idioma para nomes
        language_map = {
            "pt": "português brasileiro",
            "en": "Inglês americano"
        }
        
        language_name = language_map.get(language, "português brasileiro")
        
        system_message = f"""
Você é um especialista em educação. Gere EXATAMENTE 

Gere EXATAMENTE {quantity} flashcards sobre "{topic}" em {language_name}.

FORMATO OBRIGATÓRIO (JSON estruturado em lista de objetos):
[
  {{{{  # duas chaves para escapar
    "front": "Pergunta clara e objetiva (máx 150 chars, sem pontuação desnecessária)",
    "back": "Resposta completa, didática e precisa (máx 400 chars, evitar redundância)",
    "difficulty": "Fácil | Intermediário | Difícil",
    "suggestion": "Sugestão breve e útil de estudo ou memorização (máx 200 chars)"
  }}}}
]

REGRAS E DIRETRIZES:
1. Varie as dificuldades: pelo menos 30% Fácil, 40% Intermediário e 30% Difícil.
2. Varie os tipos de flashcards:
   - Definições (conceitos e termos)
   - Comparações (semelhanças/diferenças)
   - Aplicações (exemplos práticos, casos de uso, problemas)
3. Seja educativo, preciso e confiável. Evite ambiguidades.
4. Evite repetições de perguntas, respostas ou sugestões.
5. A linguagem deve ser clara, objetiva e adequada para estudo individual.
6. Não use listas, bullet points ou parágrafos longos nas respostas — apenas frases corridas.
7. Todas as respostas devem estar no idioma solicitado: {language_name}.
8. Garanta consistência no formato JSON: não inclua texto fora do JSON.
9. Não invente informações falsas. Se não for aplicável ao tópico, não crie o flashcard.
10. Certifique-se de que cada flashcard é autoexplicativo (não depende de outro para fazer sentido).
11. "Fácil": "Use linguagem simples e conceitos básicos.",
            "Intermediário": "Use terminologia técnica apropriada com exemplos.",
            "Difícil": "Use conceitos complexos e terminologia avançada."

REGRAS ANTI-ALUCINAÇÃO (OBRIGATÓRIAS):
1) Use apenas fatos amplamente conhecidos e verificáveis.
2) Evite números específicos, datas, estatísticas e nomes de funções/APIs pouco conhecidos.
3) Se houver incerteza, responda de forma genérica porém correta (sem inventar detalhes).
4) Para bibliotecas/tecnologias, foque em conceitos fundamentais e usos comuns.
5) Não inclua nada fora do formato solicitado. Nenhum texto fora do JSON.

RESPONDA SOMENTE COM O JSON, SEM EXPLICAÇÕES ADICIONAIS.
"""

        human_message = f"Tópico: {topic}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def generate_flashcards(self, topic: str, quantity: int = 5, language: str = "pt") -> dict:
        """Gera flashcards de forma simples, sem validações extras."""

        # Cria o prompt
        prompt = self._create_prompt(
            topic=topic,
            quantity=quantity,
            language=language
        )

        # Cria a chain
        chain = prompt | self.structured_llm

        # Executa a geração
        result = await chain.ainvoke({})

        # Retorna direto os flashcards
        return {
            "flashcards": result.flashcards
        }
    async def generate_flashcards_with_metadata(self, topic: str, quantity: int = 5, language: str = "pt") -> dict:
        """Gera flashcards com metadados para compatibilidade com FastAPI"""
        try:
            start_time = time.time()
            
            # Usar o método existente
            result = await self.generate_flashcards(topic, quantity, language)
            
            end_time = time.time()
            generation_time_ms = int((end_time - start_time) * 1000)
            
            return {
                "success": True,
                "flashcards": result["flashcards"],
                "generation_time_ms": generation_time_ms,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "flashcards": [],
                "generation_time_ms": 0,
                "error": str(e)
            }


ai_service = AIService()