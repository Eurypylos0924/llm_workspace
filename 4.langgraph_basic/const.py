import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()

API_KEY = os.getenv("NVIDIA")
MODEL = os.getenv("NVIDIA_MODEL")
EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def create_model():
    model = ChatNVIDIA(
        model=MODEL,
        api_key=API_KEY,
        # streaming=True,
        max_completion_tokens=2048,
        timeout=600,
        # chat_template_kwargs={"enable_thinking": True},
    )

    return model


def create_embedding():
    embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY)
    return embeddings