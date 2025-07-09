import openai
import os
from dotenv import load_dotenv
import time
import uuid
import json

# LangChain & Qdrant imports
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# ——————————————————————————————————————————————
# 1. Bootstrap environment and clients
# ——————————————————————————————————————————————
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = openai.Client()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_DIM = 1536

qdrant = QdrantClient(url=QDRANT_URL)
collections = qdrant.get_collections().collections

# Journal collection
COLLECTION_NAME = "journals"
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
vectorstore = Qdrant(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)

# 🔥 NEW: Knowledge base collection
KNOWLEDGE_COLLECTION = "knowledge_base"
if KNOWLEDGE_COLLECTION not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=KNOWLEDGE_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
knowledge_vectorstore = Qdrant(
    client=qdrant,
    collection_name=KNOWLEDGE_COLLECTION,
    embeddings=embeddings
)

# ——————————————————————————————————————————————
# 2. AI Feedback on Journal Entry
# ——————————————————————————————————————————————
def get_ai_feedback(journal_text: str) -> str:
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
    return response.choices[0].message.content or "No response generated."

# ——————————————————————————————————————————————
# 3. Store Journal Entry
# ——————————————————————————————————————————————
def store_journal_entry(user_id: str, journal_text: str) -> str:
    point_id = str(uuid.uuid4())
    vectorstore.add_texts(
        texts=[journal_text],
        metadatas=[{"user_id": user_id, "timestamp": int(time.time()), "type": "journal"}],
        ids=[point_id]
    )
    return point_id

# ——————————————————————————————————————————————
# 4. Retrieve Relevant Journal Entries
# ——————————————————————————————————————————————
def get_relevant_entries(user_id: str, query: str, limit: int = 10) -> list:
    all_user_entries = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="metadata.type", match=MatchValue(value="journal"))
            ]
        ),
        limit=100
    )[0]
    
    if not all_user_entries:
        return []
    
    try:
        results = vectorstore.similarity_search(
            query=query,
            k=min(limit, len(all_user_entries)),
            filter={"user_id": user_id, "type": "journal"}
        )
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Search error: {e}")
        recent_entries = sorted(
            [e for e in all_user_entries if e.payload and isinstance(e.payload, dict)],
            key=lambda x: x.payload.get('metadata', {}).get('timestamp', 0) if x.payload and isinstance(x.payload, dict) else 0,
            reverse=True
        )[:limit]
        return [
            entry.payload.get('page_content', '')
            for entry in recent_entries
            if entry.payload and entry.payload.get('page_content')
        ]

# ——————————————————————————————————————————————
# 5. Personality Analysis + Expert Knowledge
# ——————————————————————————————————————————————
def analyze_personality_and_respond(user_id: str, user_question: str) -> str:
    relevant_entries = get_relevant_entries(user_id, user_question, limit=8)
    if not relevant_entries:
        return "I don't have enough journal entries from you yet to provide personalized insights. Please write a few more journal entries first!"
    
    journal_context = "\n\n---\n\n".join(relevant_entries)

    # 🔥 NEW: Retrieve expert knowledge relevant to the user's question
    try:
        expert_docs = knowledge_vectorstore.similarity_search(query=user_question, k=3)
        expert_context = "\n\n---\n\n".join([doc.page_content for doc in expert_docs])
    except Exception as e:
        expert_context = "No expert knowledge retrieved."
        print(f"Knowledge retrieval error: {e}")

    system_prompt = f"""
You are an AI psychologist and behavioral analyst with expertise in personality psychology, CBT, and emotional intelligence.

Here is expert knowledge that may help answer the user's question:
{expert_context}

Here are the user's journal entries:
{journal_context}

Now the user is asking you:
"{user_question}"

Your task:
1. Analyze their patterns and personality
2. Use relevant expert knowledge to help
3. Give actionable and compassionate advice
"""

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.7,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    content = response.choices[0].message.content
    return content or "Sorry, I couldn't generate a response at this time."

# ——————————————————————————————————————————————
# 6. Interactive Chat Mode
# ——————————————————————————————————————————————
def start_personality_chat(user_id: str):
    print("\n🧠 Welcome to your Personal Behavior Analysis Chat!")
    print("Ask me anything about your personality, behavior patterns, or emotional tendencies.")
    print("I'll analyze your journal entries to give you personalized insights.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in ['quit', 'exit', 'bye']:
            print("🤖 Take care! Keep journaling for better insights!")
            break
        if not user_question:
            continue
        print("\n🤖 Analyzing your patterns...")
        try:
            ai_response = analyze_personality_and_respond(user_id, user_question)
            print(f"\nAI Analyst: {ai_response}\n")
        except Exception as e:
            print(f"Sorry, I encountered an error: {e}")
            print("Please try asking your question differently.\n")

# ——————————————————————————————————————————————
# 7. Main Menu
# ——————————————————————————————————————————————
def main_menu():
    user_id = os.getenv("USER_ID", "anonymous")
    
    while True:
        print("\n" + "="*50)
        print("🌟 SMART JOURNALING & PERSONALITY ANALYZER")
        print("="*50)
        print("1. Write a new journal entry")
        print("2. Chat about my personality & behavior")
        print("3. Exit")
        print("-"*50)
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == "1":
            print("\n📝 Enter your journal entry (press ENTER twice to submit):\n")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            if not lines:
                print("No entry provided.")
                continue
            journal_input = "\n".join(lines)
            print("\n🤖 GPT Feedback:\n")
            ai_reply = get_ai_feedback(journal_input)
            print(ai_reply)
            print("\n📦 Storing your journal entry...")
            journal_point_id = store_journal_entry(user_id, journal_input)
            print(f"✅ Stored journal with ID: {journal_point_id}")
        elif choice == "2":
            start_personality_chat(user_id)
        elif choice == "3":
            print("👋 Goodbye! Keep reflecting and growing!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

# ——————————————————————————————————————————————
# 8. Run App
# ——————————————————————————————————————————————
if __name__ == "__main__":
    main_menu()
