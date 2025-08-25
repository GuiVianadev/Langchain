import os
import asyncio
from typing import List, Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlashcardAI(BaseModel):
    """Modelo para um flashcard gerado pela IA"""
    front: str = Field(description="Pergunta ou conceito no front do flashcard")
    back: str = Field(description="Resposta ou explicação no verso do flashcard")
    difficulty: str = Field(description="Nível de dificuldade: beginner, intermediate, advanced")
    tags: List[str] = Field(description="Tags para categorizar o flashcard", default=[])
    explanation: Optional[str] = Field(description="Explicação adicional", default=None)

class FlashcardsGeneration(BaseModel):
    """Modelo para múltiplos flashcards gerados pela IA"""
    flashcards: List[FlashcardAI] = Field(description="Lista de flashcards gerados")

class AIServiceAdvanced:
    """Serviço para integração com modelos de IA"""
    
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "openai")
        self.model_name = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        self.temperature = float(os.getenv("TEMPERATURE", "0.5"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1500"))
        self.max_flashcards = int(os.getenv("MAX_FLASHCARDS_PER_REQUEST", "15"))
        self.min_flashcards = int(os.getenv("MIN_FLASHCARDS_PER_REQUEST", "1"))
        
        # Inicializa o modelo
        self.llm = self._initialize_model()
        
        # Configura o structured output
        if self.llm:
            self.structured_llm = self.llm.with_structured_output(FlashcardsGeneration)
        else:
            self.structured_llm = None
        
    def _initialize_model(self):
        """Inicializa o modelo de IA baseado no provider configurado"""
        try:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")
                
                return init_chat_model(
                    "gpt-3.5-turbo",  
                    model_provider="openai",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Provider {self.provider} não suportado")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {e}")
            raise
    
    def _create_prompt(self, topic: str, quantity: int, difficulty: str, 
                      language: str, context: str = None, 
                      focus_areas: List[str] = None) -> ChatPromptTemplate:
        """Cria o prompt otimizado para geração de flashcards"""
        
        # Mapeia códigos de idioma para nomes
        language_map = {
            "pt": "português brasileiro",
        }
        
        language_name = language_map.get(language, "português brasileiro")
        
        # Instruções mais concisas por nível
        difficulty_map = {
            "beginner": "Use linguagem simples e conceitos básicos.",
            "intermediate": "Use terminologia técnica apropriada com exemplos.",
            "advanced": "Use conceitos complexos e terminologia avançada."
        }
        
        # Constrói contexto de forma mais eficiente
        context_parts = []
        if context:
            context_parts.append(f"Contexto: {context}")
        if focus_areas:
            context_parts.append(f"Foque em: {', '.join(focus_areas[:3])}")  # Limita a 3 áreas
        
        additional_context = f"\n{'. '.join(context_parts)}" if context_parts else ""
        
        # Template otimizado (mais conciso)
        system_message = f"""Gere EXATAMENTE {quantity} flashcards sobre "{topic}" em {language_name}.

FORMATO OBRIGATÓRIO para cada flashcard:
- Front: Pergunta clara (máx 150 chars)
- Back: Resposta completa (máx 400 chars)  
- Difficulty: {difficulty}
- Tags: 2-3 palavras-chave
- Explanation: Contexto adicional (máx 200 chars)

REGRAS:
- {difficulty_map.get(difficulty, '')}
- Varie tipos: definições, comparações, aplicações
- Seja educativo e preciso
- Evite repetições{additional_context}

RESPONDA APENAS COM O JSON ESTRUTURADO."""

        human_message = f"Tópico: {topic}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    async def generate_flashcards(self, topic: str, quantity: int = 5, 
                                difficulty: str = "intermediate", language: str = "pt",
                                context: str = None, focus_areas: List[str] = None) -> dict:
        """Gera flashcards usando IA com fallback e retry logic"""
        
        start_time = datetime.now()
        
        # Validações
        if not self.structured_llm:
            return {
                "success": False,
                "error": "Serviço de IA não configurado corretamente",
                "flashcards": [],
                "generation_time_ms": 0
            }
        
        # Ajusta quantidade se necessário
        original_quantity = quantity
        quantity = max(self.min_flashcards, min(quantity, self.max_flashcards))
        
        if quantity != original_quantity:
            logger.warning(f"Quantidade ajustada de {original_quantity} para {quantity}")
        
        # Tenta geração com estratégia de fallback
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Tentativa {attempt + 1}: Gerando {quantity} flashcards sobre '{topic}' (dificuldade: {difficulty})")
                
                # Ajusta quantidade na última tentativa se falhou
                if attempt == max_retries and quantity > 5:
                    quantity = min(5, quantity)
                    logger.info(f"Última tentativa: reduzindo para {quantity} flashcards")
                
                # Cria o prompt
                prompt = self._create_prompt(
                    topic=topic,
                    quantity=quantity, 
                    difficulty=difficulty,
                    language=language,
                    context=context,
                    focus_areas=focus_areas
                )
                
                # Cria a chain com structured output
                chain = prompt | self.structured_llm
                
                # Executa a geração com timeout
                result = await asyncio.wait_for(
                    chain.ainvoke({}), 
                    timeout=30.0  # 30 segundos timeout
                )
                
                # Valida resultado
                if not result or not hasattr(result, 'flashcards') or not result.flashcards:
                    raise ValueError("Resultado vazio ou inválido")
                
                # Calcula tempo de geração
                generation_time = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ {len(result.flashcards)} flashcards gerados com sucesso em {generation_time:.0f}ms")
                
                return {
                    "success": True,
                    "flashcards": result.flashcards,
                    "generation_time_ms": int(generation_time),
                    "model_used": f"{self.provider}/{self.model_name}",
                    "attempts_made": attempt + 1,
                    "quantity_requested": original_quantity,
                    "quantity_generated": len(result.flashcards)
                }
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout na tentativa {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # Wait before retry
                    continue
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Erro na tentativa {attempt + 1}: {error_msg}")
                
                # Se é erro de token limit, tenta com menos flashcards
                if "length limit" in error_msg.lower() or "token" in error_msg.lower():
                    if quantity > 3 and attempt < max_retries:
                        quantity = max(3, quantity // 2)
                        logger.info(f"Reduzindo quantidade para {quantity} devido a limite de tokens")
                        await asyncio.sleep(0.5)
                        continue
                
                # Para outros erros, tenta novamente se não é a última tentativa
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                
                # Se chegou aqui, falhou em todas as tentativas
                generation_time = (datetime.now() - start_time).total_seconds() * 1000
                return {
                    "success": False,
                    "error": f"Falha após {max_retries + 1} tentativas: {error_msg}",
                    "flashcards": [],
                    "generation_time_ms": int(generation_time),
                    "attempts_made": attempt + 1
                }
        
        # Fallback final - nunca deveria chegar aqui
        return {
            "success": False,
            "error": "Erro inesperado no sistema de retry",
            "flashcards": [],
            "generation_time_ms": 0
        }
    
    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo configurado"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "status": "configured" if self.llm else "error"
        }



# Instância global do serviço
ai_service = AIServiceAdvanced()
