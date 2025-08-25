from dotenv import load_dotenv
from ..utils.config import OPENAI_API_KEY
from langchain.chat_models import init_chat_model

load_dotenv()

def get_llm(model: str="gpt-3.5-turbo"):
    return init_chat_model(model=model, model_provider="openai")
