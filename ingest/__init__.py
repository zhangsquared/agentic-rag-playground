from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from config import EMBED_MODEL_NAME, LLM_MODEL_NAME

load_dotenv()  # take environment variables from .env file

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm_model = GoogleGenAI(model_name=LLM_MODEL_NAME)
