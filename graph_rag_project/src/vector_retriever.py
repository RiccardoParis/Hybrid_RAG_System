from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY

class VectorRetriever:
    def __init__(self, collection_name: str = "hybrid_rag", model_name: str = "llama3"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Inizializza Qdrant Client
        api_key = QDRANT_API_KEY if QDRANT_API_KEY and QDRANT_API_KEY != "your_qdrant_api_key_here" else None
        self.client = QdrantClient(url=QDRANT_URL, api_key=api_key)
        
        # Configura il Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, documents):
        """Aggiunge documenti al vector store."""
        return self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 4):
        """Esegue una ricerca di similarità."""
        return self.vector_store.similarity_search(query, k=k)
