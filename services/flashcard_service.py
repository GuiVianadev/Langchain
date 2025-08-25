# services/flashcard_service.py
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Modelo para um flashcard individual
class Flashcard(BaseModel):
    question: str = Field(description="Pergunta do flashcard")
    answer: str = Field(description="Resposta do flashcard")
    difficulty: str = Field(description="Nível de dificuldade: easy, medium, hard")
    category: str = Field(description="Categoria ou tópico do flashcard")

# Modelo para a lista de flashcards
class FlashcardSet(BaseModel):
    flashcards: List[Flashcard] = Field(description="Lista de flashcards gerados")
    total_cards: int = Field(description="Número total de flashcards")
    source_info: Optional[str] = Field(description="Informações sobre a fonte", default=None)

class FlashcardGeneratorService:
    def __init__(
        self, 
        llm: BaseChatModel, 
        text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
        max_flashcards: int = 25
    ):
        self.llm = llm
        self.structured_llm = llm.with_structured_output(schema=FlashcardSet)
        self.max_flashcards = max_flashcards
        
        # Text splitter (usa o fornecido ou cria um padrão)
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=300,
            length_function=len,
        )
        
        # Template para gerar flashcards
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "Você é um especialista em criação de conteúdo educacional, focado em flashcards de alta qualidade. "
                "Crie flashcards seguindo estas diretrizes:\n"
                "- Faça perguntas claras e específicas\n"
                "- Forneça respostas concisas mas completas\n"
                "- Varie os níveis de dificuldade (easy, medium, hard)\n"
                "- Cubra conceitos-chave, definições, exemplos e aplicações\n"
                "- Categorize por tópico quando possível\n"
                "- Foque no valor educacional e retenção\n"
                "- Limite a {max_cards} flashcards no máximo\n"
                "\nInstruções personalizadas: {custom_prompt}"
            ),
            (
                "human",
                "Conteúdo para criar flashcards:\n\n{content}\n\n"
                "Por favor, gere flashcards baseados neste conteúdo."
            )
        ])
    
    def _load_pdf_content(self, pdf_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
        """Carrega e processa o conteúdo do PDF"""
        # Salva temporariamente o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # Carrega o PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Adiciona metadados
            for doc in documents:
                doc.metadata["source_file"] = filename
            
            # Divide em chunks se necessário
            if len(documents) > 1 or (documents and len(documents[0].page_content) > 6000):
                documents = self.text_splitter.split_documents(documents)
            
            logger.info(f"PDF processado: {len(documents)} chunks gerados")
            return documents
            
        except Exception as e:
            logger.error(f"Erro ao processar PDF: {e}")
            raise
        finally:
            # Remove o arquivo temporário
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _combine_content(self, documents: List[Document], max_chunks: int = 3) -> str:
        """Combina o conteúdo dos documentos em uma string"""
        # Limita chunks para evitar tokens excessivos
        limited_docs = documents[:max_chunks] if len(documents) > max_chunks else documents
        
        content_parts = []
        for i, doc in enumerate(limited_docs):
            page_info = doc.metadata.get("page", i + 1)
            content_parts.append(f"--- Página {page_info} ---\n{doc.page_content}")
        
        return "\n\n".join(content_parts)
    
    def generate_flashcards(
        self, 
        pdf_bytes: bytes, 
        filename: str = "document.pdf",
        custom_prompt: str = "",
        max_chunks: int = 3
    ) -> FlashcardSet:
        """
        Gera flashcards a partir de um PDF
        
        Args:
            pdf_bytes: Conteúdo do PDF em bytes
            filename: Nome do arquivo PDF
            custom_prompt: Prompt personalizado para guiar a geração
            max_chunks: Máximo de chunks para processar
        
        Returns:
            FlashcardSet com os flashcards gerados
        """
        try:
            # Carrega e processa o PDF
            documents = self._load_pdf_content(pdf_bytes, filename)
            
            if not documents:
                raise ValueError("Não foi possível extrair conteúdo do PDF")
            
            # Combina o conteúdo
            content = self._combine_content(documents, max_chunks)
            
            # Prompt padrão se não fornecido
            if not custom_prompt:
                custom_prompt = (
                    f"Gere entre 10-{self.max_flashcards} flashcards cobrindo os principais conceitos, "
                    "definições e detalhes importantes deste conteúdo. "
                    "Foque em informações que seriam úteis para estudo e memorização."
                )
            
            # Monta o prompt final
            prompt = self.prompt_template.invoke({
                "content": content,
                "custom_prompt": custom_prompt,
                "max_cards": self.max_flashcards
            })
            
            # Gera os flashcards
            logger.info("Gerando flashcards com IA...")
            result = self.structured_llm.invoke(prompt)
            
            # Valida e retorna
            flashcard_set = FlashcardSet.model_validate(result)
            flashcard_set.total_cards = len(flashcard_set.flashcards)
            flashcard_set.source_info = f"Gerado de: {filename}"
            
            logger.info(f"Gerados {flashcard_set.total_cards} flashcards")
            return flashcard_set
            
        except Exception as e:
            logger.error(f"Erro na geração de flashcards: {e}")
            raise
    
    def generate_flashcards_with_focus(
        self, 
        pdf_bytes: bytes, 
        focus_areas: List[str],
        filename: str = "document.pdf",
        cards_per_area: int = 3
    ) -> FlashcardSet:
        """
        Gera flashcards focando em áreas específicas
        
        Args:
            pdf_bytes: Conteúdo do PDF
            focus_areas: Lista de áreas/tópicos para focar
            filename: Nome do arquivo
            cards_per_area: Número de cards por área
        """
        total_cards = min(len(focus_areas) * cards_per_area, self.max_flashcards)
        
        custom_prompt = (
            f"Foque nestas áreas específicas: {', '.join(focus_areas)}. "
            f"Gere aproximadamente {cards_per_area} flashcards para cada área, "
            f"totalizando no máximo {total_cards} flashcards. "
            "Certifique-se de cobrir os conceitos-chave dentro de cada área de foco."
        )
        
        return self.generate_flashcards(pdf_bytes, filename, custom_prompt)
    
    def generate_difficulty_focused(
        self,
        pdf_bytes: bytes,
        difficulty_distribution: dict = None,
        filename: str = "document.pdf"
    ) -> FlashcardSet:
        """
        Gera flashcards com distribuição específica de dificuldade
        
        Args:
            pdf_bytes: Conteúdo do PDF
            difficulty_distribution: Dict como {"easy": 40, "medium": 40, "hard": 20} (percentuais)
            filename: Nome do arquivo
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 30, "medium": 50, "hard": 20}
        
        easy_cards = int(self.max_flashcards * difficulty_distribution.get("easy", 30) / 100)
        medium_cards = int(self.max_flashcards * difficulty_distribution.get("medium", 50) / 100)
        hard_cards = self.max_flashcards - easy_cards - medium_cards
        
        custom_prompt = (
            f"Gere flashcards com esta distribuição de dificuldade: "
            f"{easy_cards} fáceis (conceitos básicos e definições), "
            f"{medium_cards} médios (aplicações e relações), "
            f"{hard_cards} difíceis (análise crítica e síntese). "
            f"Total de {self.max_flashcards} flashcards."
        )
        
        return self.generate_flashcards(pdf_bytes, filename, custom_prompt)