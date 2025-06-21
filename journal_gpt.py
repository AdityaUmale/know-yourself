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
from qdrant_client.http.models import Distance, VectorParams

# ——————————————————————————————————————————————
# 1. Bootstrap environment and clients (same as before)
# ——————————————————————————————————————————————
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = openai.Client()

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "journals"
EMBEDDING_DIM = 1536

qdrant = QdrantClient(url=QDRANT_URL)
collections = qdrant.get_collections().collections

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

# ——————————————————————————————————————————————
# 2. Original AI feedback function (unchanged)
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
    return response.choices[0].message.content

# ——————————————————————————————————————————————
# 3. Store journal entry function (unchanged)
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
# 4. NEW: Retrieve relevant journal entries for a user
# ——————————————————————————————————————————————
def get_relevant_entries(user_id: str, query: str, limit: int = 10) -> list:
    """
    Search for journal entries related to the user's question.
    Returns the most relevant entries based on semantic similarity.
    """
    # First, get all entries for this user
    all_user_entries = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "metadata.user_id", "match": {"value": user_id}},
                {"key": "metadata.type", "match": {"value": "journal"}}
            ]
        },
        limit=100  # Adjust based on how many entries you expect
    )[0]  # scroll returns (points, next_page_offset)
    
    if not all_user_entries:
        return []
    
    # If we have entries, do a semantic search
    try:
        results = vectorstore.similarity_search(
            query=query,
            k=min(limit, len(all_user_entries)),
            filter={"user_id": user_id, "type": "journal"}
        )
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback: return recent entries
        recent_entries = sorted(all_user_entries, 
                              key=lambda x: x.payload.get('metadata', {}).get('timestamp', 0), 
                              reverse=True)[:limit]
        return [entry.payload.get('page_content', '') for entry in recent_entries if entry.payload.get('page_content')]

# ——————————————————————————————————————————————
# 5. NEW: Personality Analysis Chatbot
# ——————————————————————————————————————————————
def analyze_personality_and_respond(user_id: str, user_question: str) -> str:
    """
    Analyze user's personality based on their journal entries and answer their question.
    """
    # Get relevant journal entries
    relevant_entries = get_relevant_entries(user_id, user_question, limit=8)
    
    if not relevant_entries:
        return "I don't have enough journal entries from you yet to provide personalized insights. Please write a few more journal entries first!"
    
    # Create context from journal entries
    journal_context = "\n\n---\n\n".join(relevant_entries)
    
    system_prompt = f"""
You are an AI psychologist and behavioral analyst with expertise in personality psychology, CBT, and emotional intelligence.

You have access to the user's journal entries below. Based on these entries, you can understand their:
- Personality patterns and traits
- Emotional responses and triggers
- Behavioral patterns
- Coping mechanisms
- Growth areas and strengths
- Recurring themes in their life

JOURNAL ENTRIES:
{journal_context}

Now the user is asking you a question about themselves. Your task:
1. Analyze their personality and behavioral patterns from the journal entries
2. Identify relevant patterns that relate to their question
3. Provide personalized insights and suggestions
4. Be compassionate, specific, and actionable

Answer their question based on what you observe in their writing patterns and experiences.
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
    
    return response.choices[0].message.content

# ——————————————————————————————————————————————
# 6. NEW: Interactive Chat Mode
# ——————————————————————————————————————————————
def start_personality_chat(user_id: str):
    """
    Start an interactive chat session where user can ask about their behavior/personality.
    """
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
# 7. Enhanced Main Menu
# ——————————————————————————————————————————————
def main_menu():
    """
    Main application menu with options for journaling and chatbot.
    """
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
            # Original journaling functionality
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
            
            # Get AI feedback
            print("\n🤖 GPT Feedback:\n")
            ai_reply = get_ai_feedback(journal_input)
            print(ai_reply)
            
            # Store journal entry
            print("\n📦 Storing your journal entry...")
            journal_point_id = store_journal_entry(user_id, journal_input)
            print(f"✅ Stored journal with ID: {journal_point_id}")
            
        elif choice == "2":
            # New personality chat functionality
            start_personality_chat(user_id)
            
        elif choice == "3":
            print("👋 Goodbye! Keep reflecting and growing!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

# ——————————————————————————————————————————————
# 8. Run the application
# ——————————————————————————————————————————————
if __name__ == "__main__":
    main_menu()