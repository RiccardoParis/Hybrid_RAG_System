import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Qdrant Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Groq Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY is not set in the environment variables.")

# AWS Bedrock Mantle Config (OpenAI Compatible)
AWS_MANTLE_API_KEY = os.getenv("AWS_MANTLE_API_KEY")
AWS_MANTLE_REGION = os.getenv("AWS_MANTLE_REGION", "us-east-1")
AWS_MANTLE_BASE_URL = f"https://bedrock-mantle.{AWS_MANTLE_REGION}.api.aws/v1"
