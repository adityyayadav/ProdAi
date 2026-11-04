from typing import NamedTuple
import os
from dotenv import load_dotenv
from src.logger import logger

class Config(NamedTuple):
    groq_api_key: str

_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        load_dotenv()
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in .env file. Please add your Groq API key.")
            
        _config = Config(groq_api_key=groq_key)
        logger.info("Configuration loaded successfully (Groq)")
    return _config
