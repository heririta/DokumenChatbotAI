import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")

    # Cohere API configuration for embeddings
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # Model configurations
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")

    # Processing parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # Paths
    DB_PATH = os.getenv("DB_PATH", "chatbot.db")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

    # Provider selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # Groq only
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")  # Cohere embeddings

def validate_config():
    """Validate configuration - Both API keys are required"""
    errors = []

    # Groq API key is required for LLM
    if not Config.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is required. Please set it in your .env file.")

    # Cohere API key is required for embeddings
    if not Config.COHERE_API_KEY:
        errors.append("COHERE_API_KEY is required for embeddings. Please set it in your .env file.")

    if errors:
        raise ValueError("\n".join(errors))

    return True

def test_connection():
    """Test connection to API providers"""
    results = {}

    # Test LLM provider (Groq)
    try:
        from langchain_groq import ChatGroq
        if Config.GROQ_BASE_URL:
            llm = ChatGroq(
                model=Config.LLM_MODEL,
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL
            )
        else:
            llm = ChatGroq(
                model=Config.LLM_MODEL,
                api_key=Config.GROQ_API_KEY
            )
        test_response = llm.invoke("test")
        results["llm"] = (True, "Groq connection successful")
    except Exception as e:
        results["llm"] = (False, f"Groq connection failed: {str(e)}")

    # Test embedding provider (Cohere)
    try:
        from langchain_cohere import CohereEmbeddings
        embeddings = CohereEmbeddings(
            model=Config.EMBEDDING_MODEL,
            cohere_api_key=Config.COHERE_API_KEY
        )
        test_result = embeddings.embed_query("test")
        results["embedding"] = (True, "embeddings connection successful")
    except Exception as e:
        results["embedding"] = (False, f"embeddings connection failed: {str(e)}")

    return results