import openai  # Import the OpenAI Python client library
import os      # OS module to interact with environment variables
from dotenv import load_dotenv  # Load environment variables from .env file
import time    # For timestamp-based IDs
import uuid    # For generating valid UUID point IDs

# LangChain & Qdrant imports
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_qdrant import Qdrant            # Updated import
from qdrant_client import QdrantClient                      # Qdrant client
from qdrant_client.http.models import Distance, VectorParams  # Qdrant vector config

# ——————————————————————————————————————————————
# 1. Bootstrap environment and clients
# ——————————————————————————————————————————————
load_dotenv()  # Load variables from .env

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = openai.Client()  # OpenAI client for chat completions

# Embedding model via LangChain-openai (updated)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# Qdrant vector DB client setup
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "journals"
EMBEDDING_DIM = 1536  # Dimension for text-embedding-ada-002

qdrant = QdrantClient(url=QDRANT_URL)
collections = qdrant.get_collections().collections
# Create collection if missing
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

# LangChain Qdrant VectorStore
vectorstore = Qdrant(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)

# ——————————————————————————————————————————————
# 2. Define AI feedback function
# ——————————————————————————————————————————————
def get_ai_feedback(journal_text: str) -> str:
    """
    Send a journal entry to the OpenAI API and receive structured feedback.
    """
    system_prompt = """
You are a compassionate AI trained in CBT, Stoic philosophy, and emotional intelligence.
The user will input a personal journal entry.

Your task:
1. Detect their mood
2. Analyze emotional clarity and articulation
3. Offer 1 insight using CBT or Stoic wisdom
4. Suggest 1 small action for tomorrow

Respond in this JSON format:
{
  "mood": "...",
  "clarityScore": 0-10,
  "summary": "...",
  "insight": "...",
  "suggestedAction": "..."
}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": journal_text}
        ]
    )
    return response.choices[0].message.content

# ——————————————————————————————————————————————
# 3. Define function to store journal entries in Qdrant
# ——————————————————————————————————————————————
def store_journal_entry(user_id: str, journal_text: str) -> str:
    """
    Embed a journal entry and upsert into Qdrant with metadata.
    Returns the UUID string of the stored journal point.
    """
    point_id = str(uuid.uuid4())  # Generate a valid UUID
    vectorstore.add_texts(
        texts=[journal_text],
        metadatas=[{"user_id": user_id, "timestamp": int(time.time()), "type": "journal"}],
        ids=[point_id]
    )
    # Removed flush and optimize calls as they don't exist in QdrantClient
    return point_id

# ——————————————————————————————————————————————
# 4. Main execution flow
# ——————————————————————————————————————————————
if __name__ == "__main__":
    user_id = os.getenv("USER_ID", "anonymous")  # Session-based ID in production
    print("📝 Enter your journal entry (press ENTER twice to submit):\n")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    journal_input = "\n".join(lines)

    # 1) Get AI feedback
    print("\n🤖 GPT Feedback:\n")
    ai_reply = get_ai_feedback(journal_input)
    print(ai_reply)

    # 2) Store journal entry embedding
    print("\n📦 Storing your journal entry in the vector database...")
    journal_point_id = store_journal_entry(user_id, journal_input)
    print(f"✅ Stored journal with point ID: {journal_point_id}")