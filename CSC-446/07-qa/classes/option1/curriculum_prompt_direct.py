import json
import os
from openai import OpenAI

# ------------------------------------------------------
# 1. Load the curriculum JSON into memory
# ------------------------------------------------------

with open("../cua_bscs_curriculum.json", "r", encoding="utf-8") as f:
    curriculum_json = json.load(f)

curriculum_text = json.dumps(curriculum_json, indent=2)

# ------------------------------------------------------
# 2. System prompt with JSON included directly
# ------------------------------------------------------

SYSTEM_PROMPT = f"""
You are an academic advising and curriculum-analysis assistant for the
Bachelor of Science in Computer Science program at The Catholic University of America (CUA).

You have access to the full curriculum dataset below. 
Use ONLY this JSON to answer questions.
If the information is not present, say so.

Rules:
1. Do NOT invent courses, credits, or requirements.
2. When the user asks about semesters, list all courses with credits and compute totals.
3. When the user asks about a course, show title, credits, and which semesters it appears in.
4. When helpful, use bullet lists or tables.
5. If the user does not specify the catalog version (2017–2023 or 2023+), ask for clarification.
6. Always answer accurately and concisely.

Here is the curriculum dataset:

=== BEGIN CURRICULUM JSON ===
{curriculum_text}
=== END CURRICULUM JSON ===
"""

# ------------------------------------------------------
# 3. OpenAI client
# ------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------
# 4. Function to ask a question once
# ------------------------------------------------------

def ask_llm(question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",   # or gpt-4.1-mini for cheaper tests
        messages=messages,
        temperature=0,     # deterministic because we rely on facts
    )

    return response.choices[0].message.content


# ------------------------------------------------------
# 5. Simple CLI loop
# ------------------------------------------------------

if __name__ == "__main__":
    print("Curriculum advisor (JSON directly in prompt). Ctrl+C to exit.")

    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        answer = ask_llm(q)
        print("\nAssistant:\n", answer)

