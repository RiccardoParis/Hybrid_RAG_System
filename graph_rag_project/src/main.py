from config import QDRANT_URL, NEO4J_URI, GROQ_API_KEY

def main():
    print("Project initialized successfully.")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Groq API Key Configured: {'Yes' if GROQ_API_KEY else 'No'}")

if __name__ == "__main__":
    main()
