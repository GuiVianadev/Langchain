from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
import tempfile
import os

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

class FlashcardGeneratorService:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.structured_llm = llm.with_structured_output(schema=FlashcardSet)
        
        # Template para gerar flashcards
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert educational content creator specializing in flashcard generation. "
                "Create high-quality flashcards from the provided content following these guidelines:\n"
                "- Make questions clear and specific\n"
                "- Provide concise but complete answers\n"
                "- Vary difficulty levels (easy, medium, hard)\n"
                "- Cover key concepts, definitions, examples, and applications\n"
                "- Categorize by topic when possible\n"
                "- Aim for educational value and retention\n"
                "\nCustom instructions: {custom_prompt}"
            ),
            (
                "human",
                "Content to create flashcards from:\n\n{content}\n\n"
                "Please generate flashcards based on this content."
            )
        ])
        
        # Configuração do text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Tamanho do chunk
            chunk_overlap=200,  # Sobreposição entre chunks
            length_function=len,
        )
    
    def _load_pdf_content(self, pdf_bytes: bytes) -> List[Document]:
        """Carrega e processa o conteúdo do PDF"""
        # Salva temporariamente o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # Carrega o PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Divide em chunks se necessário
            if len(documents) > 1 or len(documents[0].page_content) > 8000:
                documents = self.text_splitter.split_documents(documents)
            
            return documents
        finally:
            # Remove o arquivo temporário
            os.unlink(temp_path)
    
    def _combine_content(self, documents: List[Document]) -> str:
        """Combina o conteúdo dos documentos em uma string"""
        return "\n\n".join([
            f"--- Página {i+1} ---\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
    
    def generate_flashcards(
        self, 
        pdf_bytes: bytes, 
        custom_prompt: str = "",
        max_chunks: int = 3
    ) -> FlashcardSet:
        """
        Gera flashcards a partir de um PDF
        
        Args:
            pdf_bytes: Conteúdo do PDF em bytes
            custom_prompt: Prompt personalizado para guiar a geração
            max_chunks: Máximo de chunks para processar (evita PDFs muito grandes)
        
        Returns:
            FlashcardSet com os flashcards gerados
        """
        # Carrega e processa o PDF
        documents = self._load_pdf_content(pdf_bytes)
        
        # Limita o número de chunks se necessário
        if len(documents) > max_chunks:
            documents = documents[:max_chunks]
        
        # Combina o conteúdo
        content = self._combine_content(documents)
        
        # Prompt padrão se não fornecido
        if not custom_prompt:
            custom_prompt = (
                "Generate 10-15 flashcards covering the main concepts, "
                "definitions, and important details from this content."
            )
        
        # Monta o prompt final
        prompt = self.prompt_template.invoke({
            "content": content,
            "custom_prompt": custom_prompt
        })
        
        # Gera os flashcards
        result = self.structured_llm.invoke(prompt)
        
        # Valida e retorna
        flashcard_set = FlashcardSet.model_validate(result)
        flashcard_set.total_cards = len(flashcard_set.flashcards)
        
        return flashcard_set
    
    def generate_flashcards_with_focus(
        self, 
        pdf_bytes: bytes, 
        focus_areas: List[str],
        cards_per_area: int = 5
    ) -> FlashcardSet:
        """
        Gera flashcards focando em áreas específicas
        
        Args:
            pdf_bytes: Conteúdo do PDF
            focus_areas: Lista de áreas/tópicos para focar
            cards_per_area: Número de cards por área
        """
        custom_prompt = (
            f"Focus on these specific areas: {', '.join(focus_areas)}. "
            f"Generate approximately {cards_per_area} flashcards for each area. "
            "Make sure to cover the key concepts within each focus area."
        )
        
        return self.generate_flashcards(pdf_bytes, custom_prompt)