# journal_gpt.py

import openai
import os
from dotenv import load_dotenv

# Load your OpenAI key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client()

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

if __name__ == "__main__":
    print("📝 Enter your journal entry (press ENTER twice to submit):\n")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    journal_input = "\n".join(lines)

    print("\n🤖 GPT Feedback:\n")
    ai_reply = get_ai_feedback(journal_input)
    print(ai_reply)
