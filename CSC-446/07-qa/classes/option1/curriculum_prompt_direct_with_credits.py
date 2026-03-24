import json
import os
from openai import OpenAI

# ------------------------------------------------------
# 1. Load the curriculum JSON into memory
# ------------------------------------------------------

with open("../cua_bscs_curriculum.json", "r", encoding="utf-8") as f:
    data = json.load(f)

programs = data["programs"]
semesters = data["semesters"]
courses = data["courses"]
semester_courses = data["semester_courses"]

# Build quick lookup maps
course_by_code = {c["code"]: c for c in courses}
semesters_by_id = {s["id"]: s for s in semesters}

# ------------------------------------------------------
# 2. Build a human-readable curriculum text WITH CREDITS
# ------------------------------------------------------

# helpful ordering of terms
term_order = {"Fall": 0, "Spring": 1}

lines = []

for prog in programs:
    prog_id = prog["id"]
    prog_name = prog["name"]
    lines.append(f"Program: {prog_name} (id: {prog_id})")

    # semesters for this program, ordered by (year, term)
    prog_semesters = [
        s for s in semesters if s["program_id"] == prog_id
    ]
    prog_semesters.sort(
        key=lambda s: (s["year"], term_order.get(s["term"], 99))
    )

    for sem in prog_semesters:
        sem_id = sem["id"]
        year = sem["year"]
        term = sem["term"]
        total_credits = sem.get("total_credits")

        lines.append(
            f"\nYear {year} {term} (semester_id: {sem_id}, total credits: {total_credits})"
        )
        lines.append("Courses:")

        # find semester_courses entries for this semester, sorted by position
        sc_list = [sc for sc in semester_courses if sc["semester_id"] == sem_id]
        sc_list.sort(key=lambda sc: sc.get("position", 0))

        for sc in sc_list:
            code = sc["course_code"]
            credits = sc["credits"]
            course = course_by_code.get(code, {})
            title = course.get("title", "").strip() or "(no title in data)"
            lines.append(f"- {code}: {title} — {credits} credits")

    lines.append("\n" + "-" * 60 + "\n")

curriculum_text = "\n".join(lines)

# ------------------------------------------------------
# 3. System prompt that embeds the curriculum text
# ------------------------------------------------------

SYSTEM_PROMPT = f"""
You are an academic advising and curriculum-analysis assistant for the
Bachelor of Science in Computer Science program at The Catholic University of America (CUA).

You have access to a formatted curriculum description below that includes:
- Program IDs (e.g., bscs-2017-2023, bscs-2023+)
- Semesters (Year, term, total credits)
- Courses with their titles and CREDITS

RULES:
1. Use ONLY this curriculum data to answer questions. Do NOT invent courses or credits.
2. Whenever you list courses, ALWAYS include the number of credits, exactly as shown.
3. When the user asks about a semester, list each course with its credits and mention the total credits.
4. When the user asks about a course, include:
   - code
   - title
   - number of credits
   - which semesters (program/year/term) it appears in, if relevant.
5. If the user does not specify the program version (2017–2023 or 2023+), ask for clarification.
6. If some information is not present in the data, clearly say so.

Here is the curriculum data:

=== BEGIN CURRICULUM DATA ===
{curriculum_text}
=== END CURRICULUM DATA ===
"""

# ------------------------------------------------------
# 4. OpenAI client
# ------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------
# 5. Function to ask a question once
# ------------------------------------------------------

def ask_llm(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",   # or "gpt-4.1-mini" if you want cheaper
        messages=messages,
        temperature=0,     # deterministic and factual
    )

    return response.choices[0].message.content

# ------------------------------------------------------
# 6. Simple CLI loop
# ------------------------------------------------------

if __name__ == "__main__":
    print("Curriculum advisor (formatted text with credits). Ctrl+C to exit.")

    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        answer = ask_llm(q)
        print("\nAssistant:\n", answer)

