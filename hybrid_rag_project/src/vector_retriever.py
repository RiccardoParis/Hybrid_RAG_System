import os
import re
import json
import random
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY

class VectorRetriever:
    def __init__(self, collection_name: str = "hybrid_rag", model_name: str = "intfloat/multilingual-e5-base"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize Qdrant Client
        api_key = QDRANT_API_KEY if QDRANT_API_KEY and QDRANT_API_KEY != "your_qdrant_api_key_here" else None
        self.client = QdrantClient(url=QDRANT_URL, api_key=api_key)
        
        # Configure Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, documents):
        """Adds documents to the vector store."""
        return self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 4):
        """Executes a similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def generate_and_save_metadata(self, documents: list):
        """Uses a lightweight LLM to dynamically update VectorDB metadata, merging old and new."""
        if not documents:
            return
            
        print(f"[VectorDB] Automatic generation/update of metadata in progress...")
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        meta_path = os.path.join(data_dir, "vector_metadata.json")
        
        # 1. Retrieve existing metadata (if any)
        existing_detailed_meta = "No previous context."
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    existing_detailed_meta = old_data.get("detailed", "No previous context.")
            except Exception:
                pass

        # 2. Sample new documents
        sample_size = min(10, len(documents))
        sample_docs = random.sample(documents, sample_size)
        sample_text = "\n\n---\n\n".join([doc.page_content[:500] for doc in sample_docs])
        
        # 3. Initialize LLM and Prompt for integration
        model_id = "openai/gpt-oss-20b"
        if "anthropic" in model_id or "meta.llama" in model_id:
            aws_api_key = os.getenv("AWS_MANTLE_API_KEY")
            aws_region = os.getenv("AWS_MANTLE_REGION", "us-east-1")
            llm = ChatOpenAI(
                model=model_id,
                temperature=0,
                api_key=aws_api_key,
                base_url=f"https://bedrock-mantle.{aws_region}.api.aws/v1"
            )
        else:
            llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=model_id)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Data Steward for a vector database. The database already contained documents described in the 'Previous Context'. Now, 'New Documents' are about to be inserted.
Your task is to merge the themes: you must generate a new description that includes BOTH the old topics AND the new ones.

Output EXCLUSIVELY a valid JSON dictionary with two keys:
- "compact": A very short string (max 15 words) starting with "UNSTRUCTURED TEXT:" followed by the unified domain.
- "detailed": A rich description (max 40 words) explaining exactly what the entire vector database currently contains.
No additional markdown, only the JSON code."""),
            ("user", "Previous Context:\n{existing_meta}\n\nExtracts from New Documents:\n{text}")
        ])
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({"existing_meta": existing_detailed_meta, "text": sample_text})
            content = response.content.strip()
            
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                metadata = json.loads(match.group(0))
            else:
                metadata = json.loads(content)
                
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            print(f"[VectorDB] Metadata successfully integrated and saved in {meta_path}")
            
        except Exception as e:
            print(f"[VectorDB] Error updating metadata: {e}")